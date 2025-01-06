# src/pypoprf/utils/vector.py
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from pathlib import Path
import concurrent.futures
import threading
from typing import Dict, Optional, Union, Tuple

from tqdm import tqdm



def vector2raster(gdf: gpd.GeoDataFrame,
                  transform: rasterio.Affine,
                  out_shape: Tuple[int, int],
                  column: str = 'value',
                  all_touched: bool = True,
                  dtype: str = 'float32',
                  nodata: float = -1) -> np.ndarray:
    """
    Convert vector geometries to raster format.

    Args:
        gdf: GeoDataFrame containing geometries
        transform: Affine transform
        out_shape: Shape of output raster
        column: Column name containing values to rasterize
        all_touched: Whether to rasterize all touched pixels
        dtype: Output data type
        nodata: NoData value

    Returns:
        np.ndarray: Rasterized data
    """
    try:
        if column not in gdf.columns:
            raise ValueError(f"Column '{column}' not found in GeoDataFrame")

        # Create shapes iterator
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[column]))

        # Perform rasterization
        rst = features.rasterize(
            shapes,
            out_shape=out_shape,
            fill=nodata,
            transform=transform,
            all_touched=all_touched,
            dtype=dtype
        )

        # Handle zero values
        rst = np.where(rst == 0, nodata, rst)

        return rst

    except Exception as e:
        print(f"Error in vector2raster: {str(e)}")
        raise



def rasterize(source: Union[str, gpd.GeoDataFrame],
              outfile: Union[str, Path],
              by_block: bool = True,
              max_workers: int = 4,
              template: Optional[str] = None,
              resolution: Tuple[float, float] = (0.01, 0.01),
              column: Optional[str] = None,
              dtype: str = 'int16',
              show_progress: bool = False,
              block_size: Tuple[int, int] = (256, 256),
              edit_profile: Dict = {}) -> None:
    """
    Memory-optimized rasterization function
    """
    try:
        t0 = time.time()

        # Load and validate vector data
        if isinstance(source, gpd.GeoDataFrame):
            gdf = source.copy()
        else:
            gdf = gpd.read_file(source)

        print(f"Loaded vector data with {len(gdf)} features")

        # Validate geometry
        if not all(gdf.geometry.is_valid):
            print("Warning: Invalid geometries found. Attempting to fix...")
            gdf.geometry = gdf.geometry.buffer(0)

        # Prepare rasterization column
        if column is None:
            column = 'value'
            gdf['value'] = gdf.index + 1
        elif column not in gdf.columns:
            raise ValueError(f"Column '{column}' not found in data")

        print(f"Using column '{column}' for rasterization")

        # Set up profile
        if template is None:
            bounds = gdf.total_bounds
            transform = rasterio.transform.from_origin(
                bounds[0], bounds[3],
                resolution[0], resolution[1]
            )
            sx = np.round((bounds[2] - bounds[0]) / resolution[0]).astype(int)
            sy = np.round((bounds[3] - bounds[1]) / resolution[1]).astype(int)

            profile = {
                'driver': 'GTiff',
                'dtype': dtype,
                'nodata': -99.0,
                'width': sx,
                'height': sy,
                'crs': gdf.crs,
                'count': 1,
                'transform': transform,
                'blockxsize': block_size[0],
                'blockysize': block_size[1],
                'tiled': True,
                'compress': 'lzw',
                'interleave': 'band'
            }
        else:
            with rasterio.open(template) as src:
                profile = src.profile.copy()

        # Update profile
        profile.update(edit_profile)
        profile['dtype'] = dtype

        print(f"Starting rasterization to {outfile}")

        # Process in single thread if small dataset
        if len(gdf) < 1000 or not by_block:
            print("Processing in single thread mode")
            with rasterio.open(outfile, 'w', **profile) as dst:
                # Process entire raster at once
                res = vector2raster(
                    gdf,
                    profile['transform'],
                    (profile['height'], profile['width']),
                    column=column,
                    dtype=profile['dtype'],
                    nodata=profile['nodata']
                )
                res = res.reshape((1, *res.shape[-2:]))
                dst.write(res)
        else:
            # Process in blocks with reduced number of workers
            print(f"Processing in blocks with {max_workers} workers")
            with rasterio.open(outfile, 'w', **profile) as dst:
                windows = [window for ij, window in dst.block_windows()]

                # Create queue for results
                write_lock = threading.Lock()

                def process_window(window):
                    try:
                        # Get window transform and bounds
                        transform = rasterio.windows.transform(window, dst.transform)
                        bounds = rasterio.windows.bounds(window, dst.transform)
                        out_shape = (window.height, window.width)

                        # Get data for this window
                        window_gdf = gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()

                        if len(window_gdf) < 1:
                            res = np.zeros(out_shape, dtype=profile['dtype']) + profile['nodata']
                        else:
                            res = vector2raster(
                                window_gdf,
                                transform,
                                out_shape,
                                column=column,
                                dtype=profile['dtype'],
                                nodata=profile['nodata']
                            )

                        # Write result
                        res = res.reshape((1, *res.shape[-2:]))
                        with write_lock:
                            dst.write(res, window=window)

                        # Clean up
                        del window_gdf
                        del res

                    except Exception as e:
                        print(f"Error processing window: {str(e)}")
                        raise

                # Process windows in chunks to manage memory
                chunk_size = 100
                for i in range(0, len(windows), chunk_size):
                    chunk_windows = windows[i:i + chunk_size]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        list(tqdm(
                            executor.map(process_window, chunk_windows),
                            total=len(chunk_windows),
                            disable=not show_progress,
                            desc=f"Rasterizing chunk {i // chunk_size + 1}/{len(windows) // chunk_size + 1}"
                        ))

        duration = time.time() - t0
        print(f'Rasterization completed: {outfile}')
        print(f'Finished in {duration:.2f} sec')

    except Exception as e:
        print(f"Error during rasterization: {str(e)}")
        raise