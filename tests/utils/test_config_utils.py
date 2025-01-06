# tests/utils/test_config_utils.py
import pytest
from pathlib import Path
import yaml
from pypoprf.utils.config_utils import create_config_template


def test_create_config_template(tmp_path):
    """Test creation of configuration template file."""
    # Setup
    output_path = tmp_path / "config.yaml"
    data_dir = "test_data"
    prefix = "test_"

    # Execute
    create_config_template(output_path, data_dir, prefix)

    # Verify file exists
    assert output_path.exists()

    # Verify content
    with open(output_path) as f:
        config = yaml.safe_load(f)

    # Check required fields
    assert config['work_dir'] == "."
    assert config['data_dir'] == data_dir
    assert 'covariates' in config
    assert isinstance(config['covariates'], dict)
    assert all(f"{prefix}" in v for v in config['covariates'].values())
    assert config['mastergrid'] == f"{prefix}mastergrid.tif"
    assert config['census_data'] == f"{prefix}admin3.csv"


def test_create_config_template_invalid_args():
    """Test error handling for invalid arguments."""
    with pytest.raises(ValueError):
        create_config_template(123)  # Invalid path type

    with pytest.raises(ValueError):
        create_config_template("path", data_dir=123)  # Invalid data_dir type

    with pytest.raises(ValueError):
        create_config_template("path", prefix=123)  # Invalid prefix type

