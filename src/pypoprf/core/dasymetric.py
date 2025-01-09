# src/pypoprf/core/dasymetric.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import concurrent.futures
import threading
from typing import Optional, Tuple, TypedDict, List
from rasterio.windows import Window

from ..config.settings import Settings
from ..utils.logger import get_logger
from ..utils.matplotlib_utils import with_non_interactive_matplotlib
from ..utils.raster import raster_stat, progress_bar

logger = get_logger()

class DasymetricMapper:
    """
    Handle dasymetric mapping for population distribution.

    This class manages the process of dasymetric mapping, which redistributes
    population data from census units to a finer grid based on ancillary data.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the DasymetricMapper.

        Args:
            settings: Settings object containing configuration parameters
        """
        self.settings = settings
        self.output_dir = Path(settings.work_dir) / 'output'
        self.output_dir.mkdir(exist_ok=True)
        logger.debug(f"Output directory set to: {self.output_dir}")

        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._norm_read_lock = threading.Lock()
        self._pred_read_lock = threading.Lock()
        logger.debug("Thread locks initialized")

    @staticmethod
    def _validate_census(census: pd.DataFrame,
                         kwargs: dict,
                         simple: bool = False) -> Tuple[pd.DataFrame, str, str]:
        """
        Validate census data and extract column names.

        Args:
            census: Census DataFrame with population data
            kwargs: Dictionary with parameters including column names
            simple: If True, return simplified DataFrame with only ID and population columns

        Returns:
            Tuple containing:
            - Validated census DataFrame
            - ID column name
            - Population column name

        Raises:
            ValueError: If required columns are not found
        """
        logger.info("Validating census data")

        # Get column names from DataFrame
        cols = census.columns.values
        logger.debug(f"Available columns: {cols.tolist()}")

        # Get column names from kwargs with defaults
        pop_column = kwargs.get('pop_column', 'pop')
        id_column = kwargs.get('id_column', 'id')
        logger.debug(f"Using columns: id={id_column}, population={pop_column}")

        # Validate required columns exist
        missing_cols = []
        if id_column not in cols:
            missing_cols.append(f'ID column "{id_column}"')
        if pop_column not in cols:
            missing_cols.append(f'Population column "{pop_column}"')

        if missing_cols:
            error_msg = f"Missing required columns in census data: {', '.join(missing_cols)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Handle special case where population column is named 'sum'
        if pop_column == 'sum':
            logger.info("Renaming 'sum' column to 'pop'")
            pop_column = 'pop'
            census = census.rename(columns={'sum': 'pop'})

        # Validate population values
        if (census[pop_column] < 0).any():
            logger.error("Found negative population values in census data")
            raise ValueError("Found negative population values in census data")

        # Optionally return simplified DataFrame
        if simple:
            logger.debug("Returning simplified DataFrame with only ID and population columns")
            census = census[[id_column, pop_column]]

        logger.info("Census data validation completed successfully")
        return census, id_column, pop_column

    @staticmethod
    def _check_compatibility(src_profile,
                             tgt_profile,
                             labels: Tuple[str, str]) -> None:

        logger.info(f"Checking compatibility between {labels[0]} and {labels[1]}")

        # Check for compatible CRS
        if tgt_profile['crs'] != src_profile['crs']:
            error_msg = f"CRS mismatch between {labels[1]} and {labels[0]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for compatible dimensions
        if (tgt_profile['width'] != src_profile['width'] or
                tgt_profile['height'] != src_profile['height']):
            error_msg = f"Dimension mismatch between {labels[1]} and {labels[0]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for compatible transforms
        if tgt_profile['transform'] != src_profile['transform']:
            error_msg = f"Transform mismatch between {labels[1]} and {labels[0]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Compatibility check passed successfully")

    def _validate_inputs(self,
                         prediction_path: str,
                         mastergrid_path: str,
                         constrain_path: Optional[str] = None) -> None:
        """
        Validate input files and their compatibility.

        Args:
            prediction_path: Path to prediction raster
            mastergrid_path: Path to mastergrid raster
            constrain_path: Path to constraining raster

        Raises:
            ValueError: If files are incompatible or contain invalid data
        """
        logger.info("Starting input validation")

        # Check file existence
        input_files = {
            'Prediction': prediction_path,
            'Mastergrid': mastergrid_path
        }

        if constrain_path:
            input_files['Constrain'] = constrain_path

        for file_type, path in input_files.items():
            if not Path(path).exists():
                error_msg = f"{file_type} raster not found at: {path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

        # Load and check prediction raster
        logger.info("Validating prediction raster")
        with rasterio.open(prediction_path) as pred:
            pred_profile = pred.profile
            pred_data = pred.read(1)
            pred_nodata = pred.nodata

            # Validate prediction values
            valid_pred = pred_data[pred_data != pred_nodata]
            if len(valid_pred) == 0:
                error_msg = "Prediction raster contains no valid data"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("Prediction raster statistics:")
            logger.info(f"- Shape: {pred_data.shape}")
            logger.info(f"- Valid pixels: {len(valid_pred)}")
            logger.info(f"- Value range: [{valid_pred.min():.2f}, {valid_pred.max():.2f}]")
            logger.info(f"- NoData value: {pred_nodata}")

        # Load and check mastergrid
        logger.info("Validating mastergrid")
        with rasterio.open(mastergrid_path) as mst:
            mst_profile = mst.profile
            mst_data = mst.read(1)
            mst_nodata = mst.nodata

            # Check compatibility
            self._check_compatibility(mst_profile, pred_profile, labels=('mastergrid', 'prediction'))

            # Analyze mastergrid content
            unique_zones = np.unique(mst_data[mst_data != mst_nodata])
            if len(unique_zones) == 0:
                error_msg = "Mastergrid contains no valid zones"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("Mastergrid statistics:")
            logger.info(f"- Shape: {mst_data.shape}")
            logger.info(f"- Unique zones: {len(unique_zones)}")
            logger.info(f"- Zone ID range: [{unique_zones.min()}, {unique_zones.max()}]")
            logger.info(f"- NoData value: {mst_nodata}")

        # Load and check mastergrid
        if constrain_path:
            logger.info("Validating constraining raster")
            with rasterio.open(constrain_path) as con:
                con_profile = con.profile
                con_data = con.read(1)
                con_nodata = con.nodata

                # Check compatibility
                self._check_compatibility(con_profile, pred_profile, labels=('constraining', 'prediction'))

                # Analyze mastergrid content
                valid_con = np.unique(con_data[con_data != con_nodata])
                if len(valid_con) == 0:
                    error_msg = "Constraining raster contains no valid data"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.info("Constraining raster statistics:")
                logger.info(f"- Shape: {con_data.shape}")
                logger.info(f"- Valid pixels: {len(valid_con)}")
                logger.info(f"- NoData value: {con_nodata}")

        logger.info("Input validation completed successfully")

    def _load_census(self,
                     census_path: str,
                     **kwargs) -> Tuple[pd.DataFrame, str, str]:
        """
        Load and validate census data.

        Args:
            census_path: Path to census data file
            **kwargs: Additional arguments passed to census validation

        Returns:
            Tuple containing:
            - Processed census DataFrame
            - ID column name
            - Population column name

        Raises:
            ValueError: If census data is invalid or cannot be loaded
        """
        logger.info("Loading census data")

        # Load census data based on file extension
        file_ext = Path(census_path).suffix.lower()
        try:
            if file_ext == '.csv':
                logger.debug(f"Loading CSV file: {census_path}")
                census = pd.read_csv(census_path)
            else:
                error_msg = f"Unsupported census file format: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load census data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Census data loaded: {len(census)} rows")
        logger.debug(f"Available columns: {census.columns.tolist()}")

        # Validate census data
        try:
            census, id_column, pop_column = self._validate_census(census, kwargs)
        except ValueError as e:
            error_msg = f"Census validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Basic data quality checks
        total_pop = census[pop_column].sum()
        if total_pop <= 0:
            error_msg = "Total population must be greater than 0"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Census data summary:")
        logger.info(f"- Total population: {total_pop:,}")
        logger.info(f"- Number of zones: {len(census)}")
        logger.info(f"- Using columns: id={id_column}, population={pop_column}")

        return census, id_column, pop_column

    @with_non_interactive_matplotlib
    def _calculate_normalization(self,
                                 census: pd.DataFrame,
                                 prediction_path: str,
                                 id_column: str,
                                 pop_column: str) -> pd.DataFrame:
        """Calculate normalization factors with detailed diagnostics."""

        logger.info("Calculating normalization factors...")

        # Calculate zonal statistics
        sum_prob = raster_stat(prediction_path,
                               self.settings.constrain,
                               by_block=self.settings.by_block,
                               max_workers=self.settings.max_workers,
                               block_size=self.settings.block_size)

        stats_summary = f"""
            Number of zones: {len(sum_prob)}

            Sample zones:
            {sum_prob[['id', 'sum']].head().to_string()}

            Distribution:
            {sum_prob['sum'].describe().to_string()}
            """
        logger.info(stats_summary)

        # Merge Results
        pre_merge_pop = census[pop_column].sum()

        merged = pd.merge(census,
                          sum_prob[['id', 'sum']],
                          left_on=id_column,
                          right_on='id',
                          how='outer')

        unmatched_census = merged[merged['sum'].isna()]
        unmatched_stats = merged[merged[pop_column].isna()]

        merge_summary = f"""
            Initial Population: {pre_merge_pop:,}
            Census Rows: {len(census)}
            Statistics Rows: {len(sum_prob)}
            Merged Rows: {len(merged)}

            Unmatched Data:
            - Census zones: {len(unmatched_census)} {unmatched_census['id'].tolist()}
            - Statistics zones: {len(unmatched_stats)} {unmatched_stats['id'].tolist()}
            """
        logger.info(merge_summary)

        # Normalization Results
        valid = merged['sum'].values > 0
        merged['norm'] = np.divide(merged[pop_column],
                                   merged['sum'],
                                   where=valid,
                                   out=np.full_like(merged['sum'], np.nan))
        total_pop_check = (merged['sum'] * merged['norm']).sum()

        norm_summary = f"""
            Factor Statistics:
            {merged['norm'].describe().to_string()}

            Quality Check:
            - Valid normalizations: {valid.sum()}
            - Zero sums: {len(merged[merged['sum'] == 0])}
            - Invalid norms: {len(merged[merged['norm'].isna()])}

            Population Verification:
            - Original: {pre_merge_pop:,}
            - After normalization: {total_pop_check:,.0f}
            - Difference: {abs(total_pop_check - pre_merge_pop):,} ({abs(total_pop_check - pre_merge_pop) / pre_merge_pop:.2%})
            """
        logger.info(norm_summary)

        if abs(total_pop_check - pre_merge_pop) / pre_merge_pop > 0.01:  # 1% threshold
            logger.warning("Population difference after normalization exceeds 1%")

        invalid_norms = len(merged[merged['norm'].isna()])
        if invalid_norms > 0:
            logger.warning(f"Found {invalid_norms} invalid normalization factors")

        return merged

    # @with_non_interactive_matplotlib
    # def _create_norm_raster(self,
    #                         census_df: pd.DataFrame,
    #                         mastergrid_path: str,
    #                         output_path: str) -> None:
    #     """Create normalized raster with diagnostics."""
    #
    #     with rasterio.open(mastergrid_path) as src:
    #         data = src.read(1)
    #         unique_ids = np.unique(data[data != src.nodata])
    #         print(f"\nMastergrid ID analysis:")
    #         print(f"Unique IDs in mastergrid: {len(unique_ids)}")
    #         print(f"ID range: [{unique_ids.min()}, {unique_ids.max()}]")
    #         print(f"Sample IDs: {unique_ids[:10]}")
    #
    #     print("\nChecking census data before rasterization:")
    #     print(f"Unique IDs in census: {len(census_df['id'].unique())}")
    #     print(f"Non-null norm values: {census_df['norm'].notna().sum()}")
    #
    #     # Create ID to norm mapping
    #     norm_dict = census_df.set_index('id')['norm'].to_dict()
    #     print("\nNormalization mapping:")
    #     print(f"Number of mapping entries: {len(norm_dict)}")
    #     print("Sample of norm values:")
    #     sample_ids = list(norm_dict.keys())[:5]
    #     for id_val in sample_ids:
    #         print(f"ID {id_val}: {norm_dict[id_val]:.4f}")
    #
    #     print("\nChecking normalization dictionary:")
    #     print("Norm values distribution:")
    #     norm_values = np.array(list(norm_dict.values()))
    #     print(pd.Series(norm_values).describe())
    #
    #     print("\nNorm dictionary analysis:")
    #     print(f"ID range in dictionary: [{min(norm_dict.keys())}, {max(norm_dict.keys())}]")
    #     print(f"Sample dictionary IDs: {sorted(list(norm_dict.keys()))[:10]}")
    #
    #     # Track statistics per window
    #     window_stats = []
    #
    #     with rasterio.open(mastergrid_path) as src:
    #         profile = src.profile.copy()
    #         profile.update({
    #             'dtype': 'float32',
    #             'nodata': -99,
    #             'blockxsize': self.settings.block_size[0],
    #             'blockysize': self.settings.block_size[1]
    #         })
    #
    #         def process_window(window):
    #             with self._read_lock:
    #                 mst_data = src.read(1, window=window)
    #
    #             # Initialize output array
    #             norm_data = np.full_like(mst_data, profile['nodata'], dtype='float32')
    #
    #             # Get unique IDs and apply normalization
    #             unique_ids = np.unique(mst_data)
    #             unique_ids_no_nodata = unique_ids[unique_ids != src.nodata]
    #
    #             # Track window statistics
    #             stats: WindowStats = {
    #                 'window': window,
    #                 'unique_ids_total': len(unique_ids),
    #                 'unique_ids_valid': len(unique_ids_no_nodata),
    #                 'sample_ids': unique_ids_no_nodata[:5].tolist() if len(unique_ids_no_nodata) > 0 else [],
    #                 'nodata_count': 0,
    #                 'not_in_dict_count': 0,
    #                 'nan_norm_count': 0,
    #                 'mapped_ids': 0,
    #                 'invalid_ids': 0,
    #                 'valid_pixels': 0,
    #                 'min_value': None,
    #                 'max_value': None
    #             }
    #
    #             # Process each unique ID
    #             for id_val in unique_ids:
    #                 if id_val != src.nodata:
    #                     if id_val in norm_dict:
    #                         norm = norm_dict[id_val]
    #                         if not np.isnan(norm):
    #                             norm_data[mst_data == id_val] = norm
    #                             stats['mapped_ids'] += 1
    #                         else:
    #                             stats['nan_norm_count'] += 1
    #                             stats['invalid_ids'] += 1
    #                     else:
    #                         stats['not_in_dict_count'] += 1
    #                         stats['invalid_ids'] += 1
    #                 else:
    #                     stats['nodata_count'] += 1
    #                     stats['invalid_ids'] += 1
    #
    #             # Calculate statistics for valid data
    #             valid_mask = norm_data != profile['nodata']
    #             stats['valid_pixels'] = np.sum(valid_mask)
    #             if stats['valid_pixels'] > 0:
    #                 stats['min_value'] = float(np.min(norm_data[valid_mask]))
    #                 stats['max_value'] = float(np.max(norm_data[valid_mask]))
    #             else:
    #                 stats['min_value'] = None
    #                 stats['max_value'] = None
    #
    #             window_stats.append(stats)
    #             return window, norm_data
    #
    #         print("\nCreating normalized raster...")
    #         with rasterio.open(output_path, 'w', **profile) as dst:
    #             if self.settings.by_block:
    #                 windows = [window for ij, window in dst.block_windows()]
    #
    #                 with concurrent.futures.ThreadPoolExecutor(
    #                         max_workers=self.settings.max_workers
    #                 ) as executor:
    #                     futures = [
    #                         executor.submit(process_window, window)
    #                         for window in windows
    #                     ]
    #
    #                     for future in progress_bar(
    #                             concurrent.futures.as_completed(futures),
    #                             self.settings.show_progress,
    #                             len(futures),
    #                             desc="Creating normalized raster"
    #                     ):
    #                         window, norm_data = future.result()
    #                         with self._write_lock:
    #                             dst.write(norm_data, 1, window=window)
    #
    #         print("\nAggregated Window Processing Statistics:")
    #         total_ids = sum(s['unique_ids_valid'] for s in window_stats)
    #         total_mapped = sum(s['mapped_ids'] for s in window_stats)
    #         total_invalid = sum(s['invalid_ids'] for s in window_stats)
    #         total_nodata = sum(s['nodata_count'] for s in window_stats)
    #         total_not_in_dict = sum(s['not_in_dict_count'] for s in window_stats)
    #         total_nan_norm = sum(s['nan_norm_count'] for s in window_stats)
    #
    #         print(f"\nID Statistics:")
    #         print(f"Total unique IDs found: {total_ids}")
    #         print(f"Total mapped IDs: {total_mapped}")
    #         print(f"Total invalid IDs: {total_invalid}")
    #         print(f"- NoData values: {total_nodata}")
    #         print(f"- IDs not in dictionary: {total_not_in_dict}")
    #         print(f"- NaN norm values: {total_nan_norm}")
    #
    #         # Sample of IDs from first window
    #         first_window_ids = window_stats[0]['sample_ids']
    #         if first_window_ids:
    #             print(f"\nSample IDs from first window: {first_window_ids}")

    def _process_window(self,
                        window: Window,
                        prob_ds: rasterio.DatasetReader,
                        norm_ds: rasterio.DatasetReader,
                        dst_ds: rasterio.DatasetReader,
                        nodata: float) -> dict:
        """Process window with statistics tracking."""
        stats = {'window': window}

        with self._read_lock:
            norm_data = norm_ds.read(window=window)
            pred_data = prob_ds.read(window=window)
            logger.debug("Window data read successfully")

        # Track input statistics
        valid_norm = norm_data != norm_ds.nodata
        valid_pred = pred_data != prob_ds.nodata
        stats.update({
            'valid_norm_pixels': np.sum(valid_norm),
            'valid_pred_pixels': np.sum(valid_pred),
            'norm_range': [
                float(np.min(norm_data[valid_norm])) if np.any(valid_norm) else None,
                float(np.max(norm_data[valid_norm])) if np.any(valid_norm) else None
            ],
            'pred_range': [
                float(np.min(pred_data[valid_pred])) if np.any(valid_pred) else None,
                float(np.max(pred_data[valid_pred])) if np.any(valid_pred) else None
            ]
        })

        # Calculate result
        result = np.round(norm_data * pred_data).astype('int32')

        # Apply masks
        mask_invalid = np.logical_or(norm_data == norm_ds.nodata,
                                     pred_data == prob_ds.nodata)
        result = np.where(mask_invalid, nodata, result)

        valid_result = result != nodata
        if np.any(valid_result):
            stats.update({
                'valid_output_pixels': int(np.sum(valid_result)),
                'output_sum': int(np.sum(result[valid_result])),
                'output_range': [
                    int(np.min(result[valid_result])),
                    int(np.max(result[valid_result]))
                ]
            })
            logger.debug(f"Window processed: {stats['valid_output_pixels']} valid pixels")
        else:
            stats.update({
                'valid_output_pixels': 0,
                'output_sum': 0,
                'output_range': [None, None]
            })
            logger.warning(f"Window {window} produced no valid output pixels")

        with self._write_lock:
            dst_ds.write(result, window=window)
            logger.debug("Window result written successfully")

        return stats

    @with_non_interactive_matplotlib
    def _create_normalized_raster(self,
                                  normalized_data: pd.DataFrame) -> str:
        """
        Create raster of normalization factors.

        Args:
            normalized_data: DataFrame with normalization factors

        Returns:
            Path to normalized raster
        """
        logger.info("Creating normalized census raster...")

        # Prepare output path
        output_path = self.output_dir / 'normalized_census.tif'
        logger.debug(f"Output path set to: {output_path}")

        # Get profile from mastergrid
        with rasterio.open(self.settings.mastergrid) as src:
            profile = src.profile.copy()
            profile.update({
                'dtype': 'float32',
                'nodata': -99,
                'blockxsize': self.settings.block_size[0],
                'blockysize': self.settings.block_size[1],
            })
            logger.debug("Raster profile created from mastergrid")

        # Create normalized raster
        with rasterio.open(self.settings.mastergrid) as mst, \
                rasterio.open(str(output_path), 'w', **profile) as dst:

            def process_window(window):
                # Read mastergrid data for window
                with self._read_lock:
                    mst_data = mst.read(1, window=window)

                # Create output array
                output = np.full_like(mst_data, profile['nodata'], dtype='float32')

                # Map normalization factors to zones
                valid_mappings = 0
                for idx, row in normalized_data.iterrows():
                    zone_id = row['id']
                    norm_factor = row['norm']
                    if not np.isnan(norm_factor):
                        mask = mst_data == zone_id
                        output[mask] = norm_factor
                        valid_mappings += 1


                # Write window
                with self._write_lock:
                    dst.write(output[np.newaxis, :, :], window=window)

            if self.settings.by_block:
                logger.info("Processing by blocks")
                windows = [window for ij, window in dst.block_windows()]
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.settings.max_workers
                ) as executor:
                    list(progress_bar(
                        executor.map(process_window, windows),
                        self.settings.show_progress,
                        len(windows),
                        desc="Creating normalized raster"
                    ))
            else:
                logger.info("Processing entire raster at once")
                process_window(rasterio.windows.Window(0, 0, mst.width, mst.height))

        logger.info(f"Normalized census raster created successfully: {output_path}")
        return str(output_path)

    @with_non_interactive_matplotlib
    def _create_dasymetric_raster(self,
                                  prediction_path: str,
                                  norm_raster_path: str) -> str:
        """
        Create final dasymetric population raster.

        Args:
            prediction_path: Path to prediction raster
            norm_raster_path: Path to normalization raster

        Returns:
            Path to final dasymetric raster
        """
        logger.info("Creating final dasymetric population raster...")

        # Prepare output path
        output_path = self.output_dir / 'dasymetric.tif'
        logger.debug(f"Output path set to: {output_path}")

        # Get profile from prediction raster
        with rasterio.open(prediction_path) as src:
            profile = src.profile.copy()
            profile.update({
                'dtype': 'int32',  # Population counts should be integers
                'nodata': -99,
                'blockxsize': self.settings.block_size[0],
                'blockysize': self.settings.block_size[1],
            })
            logger.debug("Raster profile created from prediction raster")

        # Create dasymetric raster
        with rasterio.open(prediction_path) as pred, \
                rasterio.open(norm_raster_path) as norm, \
                rasterio.open(str(output_path), 'w', **profile) as dst:

            def process_window(window):
                # Read input data for window
                with self._pred_read_lock:
                    pred_data = pred.read(1, window=window)
                with self._norm_read_lock:
                    norm_data = norm.read(1, window=window)

                # Calculate population
                # Multiply prediction by normalization factor and round to integers
                population = np.round(pred_data * norm_data).astype('int32')

                # Handle nodata
                nodata_mask = (pred_data == pred.nodata) | (norm_data == profile['nodata'])
                population[nodata_mask] = profile['nodata']

                # Write window
                with self._write_lock:
                    dst.write(population[np.newaxis, :, :], window=window)

            if self.settings.by_block:
                logger.info("Processing by blocks")
                windows = [window for ij, window in dst.block_windows()]
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.settings.max_workers
                ) as executor:
                    list(progress_bar(
                        executor.map(process_window, windows),
                        self.settings.show_progress,
                        len(windows),
                        desc="Creating dasymetric raster"
                    ))
            else:
                logger.info("Processing entire raster at once")
                process_window(rasterio.windows.Window(0, 0, pred.width, pred.height))

        logger.info(f"Dasymetric population raster created successfully: {output_path}")
        return str(output_path)

    def map(self,
            prediction_path: str) -> str:
        """
        Perform dasymetric mapping using prediction raster and census data.

        Args:
            prediction_path: Path to prediction raster from model

        Returns:
            Path to final dasymetric population raster
        """

        t0 = time.time()

        # Load and validate inputs
        self._validate_inputs(prediction_path, self.settings.mastergrid, self.settings.constrain)

        # Load census data
        census, id_column, pop_column = self._load_census(
            self.settings.census['path'],
            **self.settings.census
        )

        # Calculate normalization factors
        normalized_data = self._calculate_normalization(
            census,
            prediction_path,
            id_column,
            pop_column
        )

        # Create normalized raster
        norm_raster_path = self._create_normalized_raster(normalized_data)

        # Create final dasymetric raster
        final_raster_path = self._create_dasymetric_raster(
            prediction_path,
            norm_raster_path
        )

        duration = time.time() - t0
        logger.info(f'Dasymetric mapping completed in {duration:.2f} seconds')
        logger.info(f'Output saved to: {final_raster_path}')

        return final_raster_path
