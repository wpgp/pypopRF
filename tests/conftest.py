# tests/conftest.py
import pytest
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data that persists across all tests."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def sample_raster(test_data_dir):
    """Create a sample raster file for testing."""
    raster_path = test_data_dir / 'test.tif'

    # Create sample raster data
    data = np.random.rand(100, 100)

    with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            transform=rasterio.transform.from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0])
    ) as dst:
        dst.write(data, 1)
        dst.nodata = -99

    return raster_path


@pytest.fixture(scope="session")
def sample_census(test_data_dir):
    """Create a sample census file for testing."""
    census_path = test_data_dir / 'census.csv'

    # Create sample census data
    data = pd.DataFrame({
        'id': range(1, 11),
        'pop': np.random.randint(1000, 10000, 10),
        'other': np.random.rand(10)
    })

    data.to_csv(census_path, index=False)
    return census_path


@pytest.fixture(scope="session")
def base_settings(test_data_dir, sample_raster, sample_census):
    """Create base settings for tests."""
    from pypoprf.config.settings import Settings

    return Settings(
        work_dir=str(test_data_dir),
        covariates={'test': str(sample_raster)},
        census_data=str(sample_census),
        census_pop_column='pop',
        census_id_column='id'
    )