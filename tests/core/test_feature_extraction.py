# tests/core/test_feature_extraction.py
import numpy as np
import pandas as pd
import pytest
import rasterio
from pathlib import Path
from pypoprf.core.feature_extraction import FeatureExtractor
from pypoprf.config.settings import Settings


@pytest.fixture
def sample_raster(tmp_path):
    """Create a sample raster file for testing."""
    raster_path = tmp_path / 'test.tif'

    # Create sample raster data
    data = np.random.rand(100, 100)
    try:
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

            assert raster_path.exists(), "Raster file was not created"
            return raster_path
    except Exception as e:
        print(f"Error creating test raster: {e}")
        raise


@pytest.fixture
def sample_census(tmp_path):
    """Create a sample census file for testing."""
    census_path = tmp_path / 'census.csv'
    try:
        # Create sample census data
        data = pd.DataFrame({
            'id': range(1, 11),
            'pop': np.random.randint(1000, 10000, 10),
            'other': np.random.rand(10)
        })

        data.to_csv(census_path, index=False)
        assert census_path.exists(), "Census file was not created"
        return census_path
    except Exception as e:
        print(f"Error creating census: {e}")
        raise


@pytest.fixture
def feature_extractor(tmp_path, sample_raster, sample_census):
    """Create a configured FeatureExtractor instance."""
    settings = Settings(
        work_dir=str(tmp_path),
        covariates={'test': str(sample_raster)},
        census_data=str(sample_census),
        census_pop_column='pop',
        census_id_column='id'
    )
    return FeatureExtractor(settings)


def test_load_table(feature_extractor, sample_census):
    """Test loading data tables."""
    # Test CSV loading
    df = feature_extractor.load_table(sample_census)
    assert isinstance(df, pd.DataFrame)
    assert 'id' in df.columns
    assert 'pop' in df.columns

    # Test error handling
    with pytest.raises(FileNotFoundError):
        feature_extractor.load_table('nonexistent.csv')

    with pytest.raises(ValueError):
        feature_extractor.load_table('test.txt')  # Unsupported format


def test_dump_table(feature_extractor, tmp_path):
    """Test saving data tables."""
    # Create test data
    data = pd.DataFrame({
        'id': range(5),
        'value': np.random.rand(5),
        'drop_me': np.random.rand(5)
    })

    # Test basic save
    output_path = tmp_path / 'output.csv'
    feature_extractor.dump_table(data, output_path)
    assert output_path.exists()

    # Test saving with column dropping
    feature_extractor.dump_table(data, output_path, drop=['drop_me'])
    loaded = pd.read_csv(output_path)
    assert 'drop_me' not in loaded.columns


def test_validate_census(feature_extractor):
    """Test census data validation."""
    # Valid census data
    valid_census = pd.DataFrame({
        'id': range(5),
        'pop': np.random.randint(1000, 10000, 5),
        'extra': np.random.rand(5)
    })

    result, id_col, pop_col = feature_extractor.validate_census(valid_census)
    assert isinstance(result, pd.DataFrame)
    assert id_col == 'id'
    assert pop_col == 'pop'

    # Invalid census data
    invalid_census = pd.DataFrame({
        'invalid_id': range(5),
        'invalid_pop': np.random.rand(5)
    })

    with pytest.raises(ValueError):
        feature_extractor.validate_census(invalid_census)


def test_extract_features(feature_extractor, tmp_path):
    """Test feature extraction process."""
    # Run feature extraction
    features = feature_extractor.extract()

    # Verify output
    assert isinstance(features, pd.DataFrame)
    assert 'id' in features.columns
    assert 'pop' in features.columns
    assert 'dens' in features.columns
    assert any(col.endswith('avg') for col in features.columns)

    # Test saving features
    output_path = tmp_path / 'features.csv'
    features = feature_extractor.extract(save=str(output_path))
    assert output_path.exists()


def test_extract_error_handling(feature_extractor):
    """Test error handling during feature extraction."""
    # Modify settings to cause errors
    feature_extractor.settings.covariate['invalid'] = 'nonexistent.tif'

    with pytest.raises(Exception):
        feature_extractor.extract()


def test_feature_calculation(feature_extractor):
    """Test feature calculation correctness."""
    features = feature_extractor.extract()

    # Verify density calculation
    assert all(features['dens'] >= 0)  # Densities
