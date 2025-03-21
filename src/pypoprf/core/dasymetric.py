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
from ..utils.raster import raster_stat
from ..utils.raster_processing import parallel

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

    def _validate_agesex(census: pd.DataFrame,
                         id_column: str) -> Tuple[pd.DataFrame, str, str]:
        """
        Validate census data and extract column names.

        Args:
            census: Census DataFrame with population data
            id_column: Column name containing region IDs

        Returns:
            Tuple containing:
            - Validated census DataFrame
            - ID column name
            - Population column name

        Raises:
            ValueError: If required columns are not found
        """
        logger.info("Validating census data with age-sex structure")

        # Get column names from DataFrame
        cols = census.columns.values
        logger.debug(f"Available columns: {cols.tolist()}")

        # Get column names from kwargs with defaults
        pop_columns = [c for c in cols if c[0] in ['f','F','m','M']]
        #logger.debug(f"Using columns: id={id_column}, population={pop_column}")

        # Validate required columns exist
        missing_cols = []
        if id_column not in cols:
            missing_cols.append(f'ID column "{id_column}"')
            error_msg = f"Missing required columns in census data: {', '.join(missing_cols)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Census data validation completed successfully")
        return census, id_column, pop_columns
    
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

    def _load_agesex(self,
                     census_path: str, 
                     id_column: str) -> Tuple[pd.DataFrame, str, str]:
        """
        Load and validate census data with age-sex structure.

        Args:
            census_path: Path to census data file
            id_column: Column name containing region IDs

        Returns:
            Tuple containing:
            - Processed census DataFrame
            - ID column name
            - Population column name

        Raises:
            ValueError: If census data is invalid or cannot be loaded
        """
        logger.info("Loading census data with age-sex structure")

        # Load census data based on file extension
        file_ext = Path(census_path).suffix.lower()
        try:
            if file_ext == '.csv':
                logger.debug(f"Loading CSV file: {census_path}")
                census = pd.read_csv(census_path)
                cols = census.columns.values
                pop_columns = [c for c in cols if c[0] in ['f','F','m','M']]

            else:
                error_msg = f"Unsupported census file format: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load census data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Census data loaded: {len(census)} rows")
        
        # Basic data quality checks
        logger.info("Census data summary:")
        logger.info(f"- Number of zones: {len(census)}")

        return census, id_column, pop_columns

    @with_non_interactive_matplotlib
    def _calculate_normalization(self,
                                 census: pd.DataFrame,
                                 prediction_path: str,
                                 id_column: str,
                                 pop_column: str,
                                 constrained: bool = False) -> pd.DataFrame:
        """Calculate normalization factors with detailed diagnostics."""

        logger.info("Calculating normalization factors...")

        if (constrained):
            mastergrid = self.settings.constrain
        else:
            mastergrid = self.settings.mastergrid

        # Calculate zonal statistics
        sum_prob = raster_stat(prediction_path,
                               mastergrid,
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
        logger.debug(stats_summary)
        logger.info(f"Number of zones found: {len(sum_prob)}")
        logger.info(
            f"Sample zones (top 5) - ID: {sum_prob['id'].head().tolist()}, Sum: {sum_prob['sum'].head().round(2).tolist()}")

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
        logger.debug(merge_summary)
        logger.info(f"Initial total population: {pre_merge_pop:,.0f}")
        logger.info(f"Census rows: {len(census)}, Statistics rows: {len(sum_prob)}, Merged rows: {len(merged)}")

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
        logger.debug(norm_summary)
        logger.info(f"Valid normalizations: {valid.sum()} of {len(merged)} zones")
        logger.info(f"Zones with zero sums: {len(merged[merged['sum'] == 0])}")

        if abs(total_pop_check - pre_merge_pop) / pre_merge_pop > 0.01:  # 1% threshold
            logger.warning("Population difference after normalization exceeds 1%")

        invalid_norms = len(merged[merged['norm'].isna()])
        if invalid_norms > 0:
            logger.warning(f"Found {invalid_norms} invalid normalization factors")

        return merged

    @with_non_interactive_matplotlib
    def _create_normalized_raster(self,
                                  normalized_data: pd.DataFrame, 
                                  constrained: bool, 
                                  suffix: str = '') -> str:
        """
        Create raster of normalization factors.

        Args:
            normalized_data: DataFrame with normalization factors

        Returns:
            Path to normalized raster
        """
        logger.info("Creating normalized census raster...")

        # Prepare output path
        if constrained:
            mastergrid = self.settings.constrain
            output_path = self.output_dir / 'normalized_census_constrained.tif'
        else:
            mastergrid = self.settings.mastergrid
            output_path = self.output_dir / 'normalized_census.tif'

        if suffix != '':
            #output_path = output_path.replace('.tif', f'_{suffix}.tif')
            output_path = Path(str(output_path).replace('.tif', f'_{suffix}.tif'))
        logger.debug(f"Output path set to: {output_path}")

        # Get profile from mastergrid
        with rasterio.open(mastergrid) as src:
            profile = src.profile.copy()
            profile.update({
                'dtype': 'float32',
                'nodata': -99,
                'blockxsize': self.settings.block_size[0],
                'blockysize': self.settings.block_size[1],
            })
            logger.debug("Raster profile created from mastergrid")

        # Create normalized raster
        with rasterio.open(mastergrid) as mst, \
                rasterio.open(str(output_path), 'w', **profile) as dst:

            def process(window):
                # Read mastergrid data for window
                with self._read_lock:
                    mst_data = mst.read(1, window=window)

                # Create output array
                output = np.full_like(mst_data, profile['nodata'], dtype=profile['dtype'])

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

                return valid_mappings

            if self.settings.by_block:
                logger.info("Processing by blocks")
                block_windows = list(dst.block_windows())
                windows = []
                for block_window in block_windows:
                    idx, window = block_window
                    windows.append(window)

                parallel(
                    windows=windows,
                    process_func=process,
                    max_workers=self.settings.max_workers,
                    show_progress=self.settings.show_progress,
                    desc="Creating normalized raster"
                )
            else:
                logger.info("Processing entire raster at once")
                process(rasterio.windows.Window(0, 0, mst.width, mst.height))

        logger.info(f"Normalized census raster created successfully: {output_path}")
        return str(output_path)

    @with_non_interactive_matplotlib
    def _create_dasymetric_raster(self,
                                  prediction_path: str,
                                  norm_raster_path: str,
                                  constrained: bool, 
                                  suffix: str = '') -> str:
        """
        Create final dasymetric population raster.

        Args:
            prediction_path: Path to prediction raster
            norm_raster_path: Path to normalization raster
            constrained: Constrain the population

        Returns:
            Path to final dasymetric raster
        """
        logger.info("Creating final dasymetric population raster...")

        # Prepare output path
        if (constrained):
            output_path = self.output_dir / 'dasymetric_constrained.tif'
        else:
            output_path = self.output_dir / 'dasymetric.tif'
        if suffix:
            output_path = Path(str(output_path).replace('.tif', f'_{suffix}.tif'))
            #output_path = output_path.replace('.tif', f'_{suffix}.tif')

        logger.debug(f"Output path set to: {output_path}")

        # Get profile from prediction raster
        with rasterio.open(prediction_path) as src:
            profile = src.profile.copy()
            profile.update({
                'dtype': 'float32',  # Population counts should be integers
                'nodata': -99,
                'blockxsize': self.settings.block_size[0],
                'blockysize': self.settings.block_size[1],
            })
            logger.debug("Raster profile created from prediction raster")

        # Create dasymetric raster
        with rasterio.open(prediction_path) as pred, \
                rasterio.open(norm_raster_path) as norm, \
                rasterio.open(str(output_path), 'w', **profile) as dst:

            def process(window):
                # Read input data for window
                with self._pred_read_lock:
                    pred_data = pred.read(1, window=window)
                with self._norm_read_lock:
                    norm_data = norm.read(1, window=window)

                # Calculate population
                # Multiply prediction by normalization factor
                #population = np.round(pred_data * norm_data).astype('int32')
                population = pred_data * norm_data

                # Handle nodata
                nodata_mask = (pred_data == pred.nodata) | (norm_data == profile['nodata'])
                population[nodata_mask] = profile['nodata']

                # Write window
                with self._write_lock:
                    dst.write(population[np.newaxis, :, :], window=window)

            if self.settings.by_block:
                logger.info("Processing by blocks")
                block_windows = list(dst.block_windows())
                windows = []
                for block_window in block_windows:
                    idx, window = block_window
                    windows.append(window)

                parallel(
                    windows=windows,
                    process_func=process,
                    max_workers=self.settings.max_workers,
                    show_progress=self.settings.show_progress,
                    desc="Performing dasymetric redistribution"
                )
            else:
                logger.info("Processing entire raster at once")
                process(rasterio.windows.Window(0, 0, pred.width, pred.height))

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
            pop_column,
            constrained=False
        )

        # Create normalized raster
        norm_raster_path = self._create_normalized_raster(
            normalized_data, 
            constrained=False
        )

        # Create final dasymetric raster
        final_raster_path = self._create_dasymetric_raster(
            prediction_path,
            norm_raster_path, 
            constrained=False
        )

        # If constraining layer is provided
        if (self.settings.constrain):
            normalized_data_c = self._calculate_normalization(
                census,
                prediction_path,
                id_column,
                pop_column,
                constrained=True
            )
            norm_raster_path_c = self._create_normalized_raster(
                normalized_data_c, 
                constrained=True
            )
            final_raster_path = self._create_dasymetric_raster(
                prediction_path,
                norm_raster_path_c, 
                constrained=True
            )

        duration = time.time() - t0
        logger.info(f'Dasymetric mapping completed in {duration:.2f} seconds')
        logger.info(f'Output saved to: {final_raster_path}')

        return final_raster_path

    def map_agesex(self,
            prediction_path: str,
            agesex_path: str) -> str:
        """
        Perform dasymetric mapping using prediction raster and data with age-sex structure.

        Args:
            prediction_path: Path to prediction raster from model
            agesex_path: Path to census data with age-sex structure

        Returns:
            Path to final dasymetric population raster
        """

        t0 = time.time()

        # Load and validate inputs
        self._validate_inputs(prediction_path, self.settings.mastergrid, self.settings.constrain)
        
        # Load census data
        census, id_column, pop_columns = self._load_agesex(agesex_path, self.settings.census['id_column'])
        norm = census[[id_column]].copy()
        norm['one'] = 1
        
        # Perform zonal statistics to get normalization factor
        normalized_data = self._calculate_normalization(
            norm,
            prediction_path,
            id_column,
            'one',
            constrained=False
        )

        for pop_column in pop_columns:
            normalized = normalized_data.copy()
            normalized['norm'] *= census[pop_column].values

            # Create normalized raster
            norm_raster_path = self._create_normalized_raster(
                normalized, 
                constrained=False
            )

            # Create final dasymetric raster
            final_raster_path = self._create_dasymetric_raster(
                prediction_path,
                norm_raster_path, 
                constrained=False,
                suffix=pop_column
            )

        # If constraining layer is provided
        print('Check', self.settings.constrain)
        if (self.settings.constrain):
            normalized_data_c = self._calculate_normalization(
                norm,
                prediction_path,
                id_column,
                'one',
                constrained=True
            )

            for pop_column in pop_columns:
                normalized = normalized_data_c.copy()
                normalized['norm'] *= census[pop_column].values

                # Create normalized raster
                norm_raster_path = self._create_normalized_raster(
                    normalized, 
                    constrained=True
                )

                # Create final dasymetric raster
                final_raster_path = self._create_dasymetric_raster(
                    prediction_path,
                    norm_raster_path, 
                    constrained=True,
                    suffix=pop_column
                )

        duration = time.time() - t0
        logger.info(f'Dasymetric mapping completed in {duration:.2f} seconds')
        logger.info(f'Output saved to: {final_raster_path}')

        return final_raster_path
