# tests/config/test_settings.py
import pytest
import yaml
from pathlib import Path
from pypoprf.config.settings import Settings


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration file for testing."""
    config = {
        'work_dir': str(tmp_path),
        'data_dir': 'data',
        'covariates': {
            'cnt': 'building_count.tif',
            'srf': 'building_surface.tif'
        },
        'mastergrid': 'mastergrid.tif',
        'census_data': 'census.csv',
        'census_pop_column': 'population',
        'census_id_column': 'id',
        'block_size': [512, 512],
        'max_workers': 4,
        'show_progress': True
    }

    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config_path


def test_settings_initialization(tmp_path):
    """Test basic Settings initialization."""
    settings = Settings(
        work_dir=str(tmp_path),
        covariates={'test': 'test.tif'},
        census_data='census.csv',
        census_pop_column='pop',
        census_id_column='id'
    )

    assert settings.work_dir == tmp_path
    assert settings.covariate == {'test': str(tmp_path / 'data' / 'test.tif')}
    assert settings.census['pop_column'] == 'pop'


def test_settings_from_file(sample_config):
    """Test loading settings from config file."""
    settings = Settings.from_file(sample_config)

    assert isinstance(settings, Settings)
    assert settings.work_dir == Path(sample_config).parent
    assert len(settings.covariate) == 2
    assert settings.block_size == (512, 512)


def test_settings_validation():
    """Test settings validation."""
    # Test missing required fields
    with pytest.raises(ValueError):
        Settings(work_dir=".")  # Missing covariates

    with pytest.raises(ValueError):
        Settings(
            work_dir=".",
            covariates={'test': 'test.tif'}
        )  # Missing census settings


def test_settings_path_resolution(tmp_path):
    """Test path resolution in settings."""
    settings = Settings(
        work_dir=str(tmp_path),
        covariates={'test': 'data/test.tif'},
        census_data='data/census.csv',
        census_pop_column='pop',
        census_id_column='id'
    )

    # Check path resolution
    assert str(settings.data_dir) == str(tmp_path / 'data')
    assert settings.covariate['test'] == str(tmp_path / 'data' / 'test.tif')
    assert settings.census['path'] == str(tmp_path / 'data' / 'census.csv')


def test_settings_string_representation(sample_config):
    """Test string representation of settings."""
    settings = Settings.from_file(sample_config)
    str_repr = str(settings)

    # Check key information is present in string representation
    assert 'pypopRF Settings:' in str_repr
    assert 'Work Directory:' in str_repr
    assert 'Covariates:' in str_repr
    assert 'Census:' in str_repr
    assert 'Processing:' in str_repr


def test_validate_config_file(sample_config):
    """Test configuration file validation."""
    # Valid config
    Settings.validate_config_file(sample_config)

    # Invalid config (missing required field)
    invalid_config = Path(sample_config).parent / 'invalid_config.yaml'
    with open(invalid_config, 'w') as f:
        yaml.dump({'work_dir': '.'}, f)

    with pytest.raises(ValueError):
        Settings.validate_config_file(invalid_config)