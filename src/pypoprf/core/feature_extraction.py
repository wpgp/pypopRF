# src/pypoprf/core/feature_extraction.py
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import pandas as pd
import geopandas as gpd

from ..utils.raster import raster_stat_stack
from ..config.settings import Settings


class FeatureExtractor:
    """Extract and process features for population modeling."""

    # Supported file formats
    SUPPORTED_VECTOR_FORMATS = {'.shp', '.json', '.geojson', '.gpkg'}
    SUPPORTED_TABLE_FORMATS = {'.csv'}

    def __init__(self, settings: Settings):
        """
        Initialize feature extractor.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.features = None

    def load_table(self, path: str) -> pd.DataFrame:
        """
        Load data table from file.

        Args:
            path: Path to data file

        Returns:
            pd.DataFrame: Loaded data table
            gpd.GeoDataFrame: For spatial data formats

        Raises:
            ValueError: If file format is not supported or file cannot be read
            FileNotFoundError: If file does not exist
        """
        path = Path(path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()

        try:
            if ext in self.SUPPORTED_VECTOR_FORMATS:
                return gpd.read_file(path)
            elif ext in self.SUPPORTED_TABLE_FORMATS:
                return pd.read_csv(path)
            else:
                raise ValueError(f'Unsupported file format: {ext}')

        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")

    def dump_table(self, df: pd.DataFrame, path: str, drop: Optional[List[str]] = None) -> None:
        """
        Save data table to file.

        Args:
            df: DataFrame to save
            path: Output file path
            drop: Optional list of columns to drop before saving

        Raises:
            ValueError: If file format is not supported or saving fails
            TypeError: If input arguments are invalid
        """
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if drop is not None:
            if isinstance(drop, str):
                drop = [drop]  # Convert single string to list
            elif not isinstance(drop, list):
                raise TypeError("drop must be a string or list of strings")

            # Filter existing columns
            cols = df.columns.values
            drop = [d for d in drop if d in cols]
            df = df.drop(columns=drop)

        # Handle file saving
        path = Path(path)
        ext = path.suffix.lower()

        try:
            if ext in self.SUPPORTED_VECTOR_FORMATS:
                if not isinstance(df, gpd.GeoDataFrame):
                    raise ValueError("DataFrame must be GeoDataFrame for spatial formats")
                df.to_file(str(path), index=False)
            elif ext in self.SUPPORTED_TABLE_FORMATS:
                df.to_csv(path, index=False)
            else:
                raise ValueError(f'Unsupported file format: {ext}')

        except Exception as e:
            raise ValueError(f"Failed to save file: {str(e)}")

    def validate_census(self,
                        census: pd.DataFrame,
                        simple: bool = False) -> Tuple[pd.DataFrame, str, str]:
        """Validate census data and return processed dataframe with column names."""
        cols = census.columns.values
        pop_column = self.settings.census['pop_column']
        id_column = self.settings.census['id_column']

        if id_column not in cols:
            raise ValueError('id_column not found in census data')
        if pop_column not in cols:
            raise ValueError('pop_column not found in census data')

        if pop_column == 'sum':
            pop_column = 'pop'
            census = census.rename(columns={'sum': 'pop'})

        if simple:
            census = census[[id_column, pop_column]]

        return census, id_column, pop_column

    def extract(self,
                save: Optional[str] = None,
                avg_only: bool = True) -> pd.DataFrame:
        """
        Extract features from raster data.

        Args:
            save: Optional path to save features
            avg_only: If True, keep only average statistics

        Returns:
            pd.DataFrame: Extracted features

        Raises:
            ValueError: If required data is missing
        """

        # Extract features from raster stack
        res = raster_stat_stack(
            self.settings.covariate,
            self.settings.mastergrid,
            by_block=self.settings.by_block,
            max_workers=self.settings.max_workers,
            blocksize=self.settings.blocksize
        )

        # Load and validate census data
        census = self.load_table(self.settings.census['path'])
        census, id_column, pop_column = self.validate_census(census, simple=True)

        # Merge with census data
        res = pd.merge(
            res,
            census,
            left_on='id',
            right_on=id_column,
            how='inner'
        )

        # Get count for density calculation
        for c in res.columns.values:
            if c.endswith('count'):
                count = res[c].values
                break

        # Filter columns if avg_only is True
        if avg_only:
            cols = ['id', 'pop']
            cols.extend([c for c in res.columns if c.endswith('avg')])
            res = res[cols]

        # Calculate population density
        res['dens'] = np.divide(res['pop'], count, where=(count > 0))

        # Save features
        output_path = Path(self.settings.work_dir) / 'output' / 'features.csv'
        self.dump_table(res, str(output_path))

        # Save to additional location if specified
        if save:
            save_path = Path(self.settings.work_dir) / save
            self.dump_table(res, str(save_path))

        print(f'Extracted features: {res.columns.values.tolist()}')

        self.features = res
        return res
