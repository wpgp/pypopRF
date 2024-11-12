# src/popupy/utils/raster.py
import numpy as np
import pandas as pd
import rasterio
import threading
from typing import Dict, List, Tuple, Optional, Any

from rasterio.windows import Window
from tqdm import tqdm
import concurrent.futures

from popupy.utils.matplotlib_utils import with_non_interactive_matplotlib


def progress_bar(iterable: Any,
                show: bool,
                total: int,
                desc: str = "Processing") -> Any:
    """
    Create a progress bar for iteration.

    Args:
        iterable: Iterable object
        show: Whether to show progress bar
        total: Total number of items
        desc: Description for progress bar

    Returns:
        Progress bar wrapped iterable
    """
    if show:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def raster_compare(p1: Dict,
                   p2: Dict) -> List[str]:
    """
    Compare two raster profiles for compatibility.

    Args:
        p1: First raster profile
        p2: Second raster profile

    Returns:
        List of required processing steps
    """
    process = {
        'crs': 'reprojection',
        'width': 'clipping',
        'height': 'clipping'
    }

    required = []
    for p in ['crs', 'width', 'height']:
        if p1[p] != p2[p]:
            print(f'rasters have different [{p}]')
            print(f'required process: {process[p]}')
            required.append(process[p])

    if p1['transform'][0] != p2['transform'][0]:
        print('rasters have different resolutions')
        print('required process: resampling')
        required.append('resampling')

    return required


def get_raster_stats(t: np.ndarray,
                     m: np.ndarray,
                     nodata: Optional[float] = None,
                     skip: Optional[float] = None) -> pd.DataFrame:
    """
    Calculate statistics for a raster within mask regions.

    Args:
        t: Target raster data
        m: Mask raster data
        nodata: No data value
        skip: Value to skip in mask

    Returns:
        DataFrame with statistics
    """
    ids = np.unique(m)
    df_list = []

    for i in ids:
        if i == skip:
            continue

        select = np.logical_and(t != nodata, m == i)
        count = np.sum(select)
        if count < 1:
            continue

        tm = np.where(select, t, np.nan)
        d = pd.DataFrame({
            'id': i,
            'count': count,
            'sum': np.nansum(tm),
            'sum2': np.nansum(tm * tm),
            'min': np.nanmin(tm),
            'max': np.nanmax(tm)
        }, index=[0])

        df_list.append(d)

    if df_list:
        return pd.concat(df_list, ignore_index=True)

    return pd.DataFrame()


def aggregate_table(df: pd.DataFrame, prefix: str = '', min_count: int = 1) -> pd.DataFrame:
    """
    Aggregate statistics from raster data.

    Args:
        df: Input DataFrame with statistics
        prefix: Prefix for output column names
        min_count: Minimum count threshold for valid data

    Returns:
        DataFrame with aggregated statistics
    """
    # Basic validation
    if df.empty:
        return pd.DataFrame()

    # Group by ID and aggregate
    ag1 = df[['id', 'count', 'sum', 'sum2']].groupby('id').sum().reset_index()
    ag2 = df[['id', 'min']].groupby('id').min().reset_index()
    ag3 = df[['id', 'max']].groupby('id').max().reset_index()

    # Mask for valid values
    where = ag1['count'].values > 0

    # Calculate mean with safe division
    avg = np.divide(ag1['sum'].values,
                    ag1['count'].values,
                    out=np.zeros_like(ag1['sum'].values, dtype=float),
                    where=where)

    # Calculate variance with checks
    var_raw = np.divide(ag1['sum2'].values,
                        ag1['count'].values,
                        out=np.zeros_like(ag1['sum2'].values, dtype=float),
                        where=where) - avg * avg
    var = np.maximum(var_raw, 0)  # Replace negative values with 0

    # Safe standard deviation calculation
    std = np.sqrt(var, where=var >= 0)

    # Handle prefix
    if prefix:
        prefix = f"{prefix}_"

    # Create output DataFrame
    out = ag2[['id']].copy()
    out[prefix + 'count'] = ag1['count'].values
    out[prefix + 'sum'] = ag1['sum'].values
    out[prefix + 'min'] = ag2['min'].values
    out[prefix + 'max'] = ag3['max'].values
    out[prefix + 'avg'] = avg
    out[prefix + 'var'] = var
    out[prefix + 'std'] = std

    # Filter by minimum count
    out = out[out[prefix + 'count'] > min_count]

    # Replace NaN with 0 in statistics columns
    stats_cols = [prefix + x for x in ['avg', 'var', 'std']]
    out[stats_cols] = out[stats_cols].fillna(0)

    return out


@with_non_interactive_matplotlib
def raster_stat(infile: str,
                mastergrid: str,
                by_block: bool = True,
                max_workers: int = 4,
                blocksize: Optional[Tuple[int, int]] = None,
                show_progress: bool = True) -> pd.DataFrame:
    """
    Calculate zonal statistics for a raster.

    Args:
        infile: Input raster path
        mastergrid: Master grid raster path
        by_block: Whether to process by blocks
        max_workers: Number of worker processes
        blocksize: Size of processing blocks
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with zonal statistics

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If processing fails
    """
    import time
    t0 = time.time()

    try:
        with rasterio.open(mastergrid, 'r') as mst, rasterio.open(infile, 'r') as tgt:
            nodata = tgt.nodata
            skip = mst.nodata
            lock = threading.Lock()

            def process(window):
                with lock:
                    m = mst.read(window=window)
                    t = tgt.read(window=window)
                d = get_raster_stats(t, m, nodata=nodata, skip=skip)
                return d

            if by_block:
                if blocksize is None:
                    windows = [window for ij, window in tgt.block_windows()]
                else:
                    x0 = np.arange(0, mst.width, blocksize[0])
                    y0 = np.arange(0, mst.height, blocksize[1])
                    grid = np.meshgrid(x0, y0)
                    windows = [rasterio.windows.Window(a, b, blocksize[0], blocksize[1])
                               for (a, b) in zip(grid[0].flatten(), grid[1].flatten())]

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    df = list(progress_bar(executor.map(process, windows),
                                           show_progress,
                                           len(windows),
                                           desc="Dasymetric raster statistics"))

                res = pd.concat(df, ignore_index=True)

            else:
                m = mst.read(1)
                t = tgt.read(1)
                res = get_raster_stats(t, m, nodata=nodata, skip=skip)

    except Exception as e:
        raise RuntimeError(f"Error processing rasters: {str(e)}")

    out_df = aggregate_table(res)
    duration = time.time() - t0
    print(f'Raster statistics is finished in {duration:.2f} sec')

    return out_df


@with_non_interactive_matplotlib
def raster_stat_stack(infiles: Dict[str, str],
                      mastergrid: str,
                      by_block: bool = True,
                      max_workers: int = 4,
                      blocksize: Optional[Tuple[int, int]] = None,
                      show_progress: bool = True) -> pd.DataFrame:
    """
    Calculate zonal statistics for multiple rasters.

    Args:
        infiles: Dictionary of raster names and paths
        mastergrid: Master grid raster path
        by_block: Whether to process by blocks
        max_workers: Number of worker processes
        blocksize: Size of processing blocks
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with combined statistics

    Raises:
        RuntimeError: If processing fails
        FileNotFoundError: If input files don't exist
    """
    import time
    t0 = time.time()

    try:
        with rasterio.open(mastergrid, 'r') as mst:
            # Open all target rasters using context manager
            targets = []
            try:
                targets = [rasterio.open(path, 'r') for path in infiles.values()]
                nodata = [t.nodata for t in targets]
                skip = mst.nodata
                lock = threading.Lock()

                def process(window):
                    with lock:
                        m = mst.read(window=window)
                        t = [tgt.read(window=window) for tgt in targets]
                    d = [get_raster_stats(ta, m, nodata=na, skip=skip)
                         for (ta, na) in zip(t, nodata)]
                    return d

                if by_block:
                    if blocksize is None:
                        # Use first raster to determine blocks
                        windows = [window for ij, window in targets[0].block_windows()]
                    else:
                        x0 = np.arange(0, mst.width, blocksize[0])
                        y0 = np.arange(0, mst.height, blocksize[1])
                        grid = np.meshgrid(x0, y0)
                        windows = [rasterio.windows.Window(a, b, blocksize[0], blocksize[1])
                                   for (a, b) in zip(grid[0].flatten(), grid[1].flatten())]

                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        df = list(progress_bar(
                            executor.map(process, windows),
                            show_progress,
                            len(windows),
                            desc="Feature Extraction"
                        ))
                else:
                    m = mst.read(1)
                    df = [get_raster_stats(t.read(1), m, nodata=na, skip=skip)
                          for t, na in zip(targets, nodata)]

                # Create output DataFrame
                out_df = pd.DataFrame({'id': []})
                for i, key in enumerate(infiles):
                    d = [a[i] for a in df]
                    res = pd.concat(d, ignore_index=True)
                    out = aggregate_table(res, prefix=key)
                    out_df = pd.merge(out, out_df, on='id', how='outer')

            finally:
                # Close all opened rasters
                for t in targets:
                    try:
                        t.close()
                    except Exception:
                        pass  # Ignore errors on closing

    except Exception as e:
        raise RuntimeError(f"Error processing raster stack: {str(e)}")

    duration = time.time() - t0
    print(f'Raster statistics is finished in {duration:.2f} sec')

    return out_df