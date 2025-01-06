# tests/core/test_dasymetric.py
import numpy as np
import pandas as pd
import pytest
import rasterio
from pathlib import Path

from pypoprf.core.dasymetric import DasymetricMapper
from pypoprf.config.settings import Settings


@pytest.fixture
def sample_prediction(tmp_path):
    """Create sample prediction raster for testing."""
    pred_path = tmp_path / 'prediction.tif'

    # Create sample prediction data (probability surface)
    data = np.random.uniform(0, 1, (100, 100))

    with rasterio.open(
            pred_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype='float32',
            transform=rasterio.transform.from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0])
    ) as dst:
        dst.write(data.astype('float32'), 1)
        dst.nodata = -99

    return pred_path


@pytest.fixture
def sample_mastergrid(tmp_path):
    """Create sample mastergrid raster for testing."""
    grid_path = tmp_path / 'mastergrid.tif'

    # Create sample zone IDs (1-5 repeated across the grid)
    data = np.random.randint(1, 6, (100, 100))

    with rasterio.open(
            grid_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype='int32',
            transform=rasterio.transform.from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0])
    ) as dst:
        dst.write(data.astype('int32'), 1)
        dst.nodata = -99

    return grid_path


@pytest.fixture
def sample_census(tmp_path):
    """Create sample census data for testing."""
    census_path = tmp_path / 'census.csv'

    # Create sample census data for 5 zones
    data = pd.DataFrame({
        'id': range(1, 6),
        'pop': np.random.randint(1000, 10000, 5)
    })

    data.to_csv(census_path, index=False)
    return census_path


@pytest.fixture
def mapper_instance(tmp_path, sample_mastergrid, sample_census):
    """Create a configured DasymetricMapper instance."""
    settings = Settings(
        work_dir=str(tmp_path),
        mastergrid=str(sample_mastergrid),
        census_data=str(sample_census),
        census_pop_column='pop',
        census_id_column='id',
        covariates={'dummy': 'dummy.tif'}  # Needed for settings validation
    )
    return DasymetricMapper(settings)


def test_mapper_initialization(mapper_instance):
    """Test mapper initialization."""
    assert isinstance(mapper_instance, DasymetricMapper)
    assert mapper_instance.output_dir.exists()


def test_input_validation(mapper_instance, sample_prediction):
    """Test input validation."""
    # Valid inputs
    mapper_instance._validate_inputs(
        str(sample_prediction),
        str(mapper_instance.settings.mastergrid)
    )

    # Invalid prediction path
    with pytest.raises(FileNotFoundError):
        mapper_instance._validate_inputs(
            'nonexistent.tif',
            str(mapper_instance.settings.mastergrid)
        )


def test_census_validation(mapper_instance, sample_census):
    """Test census data validation."""
    census, id_col, pop_col = mapper_instance._load_census(
        sample_census,
        pop_column='pop',
        id_column='id'
    )

    assert isinstance(census, pd.DataFrame)
    assert id_col == 'id'
    assert pop_col == 'pop'
    assert all(census[pop_col] >= 0)


def test_normalization_calculation(mapper_instance, sample_prediction):
    """Test normalization factor calculation."""
    # Load census data
    census, id_col, pop_col = mapper_instance._load_census(
        mapper_instance.settings.census['path'],
        **mapper_instance.settings.census
    )

    # Calculate normalization
    normalized = mapper_instance._calculate_normalization(
        census,
        sample_prediction,
        id_col,
        pop_col
    )

    assert isinstance(normalized, pd.DataFrame)
    assert 'norm' in normalized.columns
    assert all(normalized['norm'].notna())


def test_dasymetric_mapping(mapper_instance, sample_prediction):
    """Test complete dasymetric mapping process."""
    result_path = mapper_instance.map(str(sample_prediction))

    assert Path(result_path).exists()

    # Verify output raster
    with rasterio.open(result_path) as src:
        data = src.read(1)
        assert data.shape == (100, 100)
        assert src.nodata == -99
        assert not np.all(data == src.nodata)
        assert data.dtype == np.int32  # Population should be integer


def test_parallel_processing(mapper_instance, sample_prediction):
    """Test parallel processing capabilities."""
    # Test with different numbers of workers
    for n_workers in [1, 2, 4]:
        mapper_instance.settings.max_workers = n_workers
        result_path = mapper_instance.map(str(sample_prediction))
        assert Path(result_path).exists()


def test_normalization_raster_creation(mapper_instance, sample_prediction):
    """Test creation of normalization raster."""
    # Run mapping to generate normalization raster
    mapper_instance.map(str(sample_prediction))

    # Check if normalization raster was created
    norm_path = mapper_instance.output_dir / 'normalized_census.tif'
    assert norm_path.exists()

    # Verify normalization raster content
    with rasterio.open(norm_path) as src:
        data = src.read(1)
        assert data.shape == (100, 100)
        assert src.nodata == -99
        valid_data = data[data != src.nodata]
        assert np.all(valid_data >= 0)  # Normalization factors should be non-negative


def test_error_handling(mapper_instance):
    """Test error handling in dasymetric mapping."""
    # Test with invalid prediction path
    with pytest.raises(FileNotFoundError):
        mapper_instance.map('nonexistent.tif')

    # Test with invalid census data
    invalid_census = pd.DataFrame({'invalid': range(5)})
    mapper_instance.settings.census['path'] = 'invalid.csv'
    with pytest.raises(ValueError):
        mapper_instance._validate_census(invalid_census, {})


def test_population_preservation(mapper_instance, sample_prediction):
    """Test if total population is preserved after redistribution."""
    # Get original total population
    census = pd.read_csv(mapper_instance.settings.census['path'])
    original_total = census[mapper_instance.settings.census['pop_column']].sum()

    # Run dasymetric mapping
    result_path = mapper_instance.map(str(sample_prediction))

    # Calculate total in result
    with rasterio.open(result_path) as src:
        data = src.read(1)
        redistributed_total = np.sum(data[data != src.nodata])

    # Check if totals match (allowing for small rounding differences)
    assert abs(original_total - redistributed_total) / original_total < 0.01
