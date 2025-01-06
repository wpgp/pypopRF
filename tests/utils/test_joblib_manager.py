# tests/utils/test_joblib_manager.py
import os
import pytest
from pypoprf.utils.joblib_manager import joblib_resources


def test_joblib_resources():
    """Test joblib resources context manager."""
    original_temp = os.environ.get('JOBLIB_TEMP_FOLDER')

    with joblib_resources() as temp_dir:
        # Verify temp directory is created and set
        assert os.path.exists(temp_dir)
        assert os.environ['JOBLIB_TEMP_FOLDER'] == temp_dir

    # Verify cleanup
    assert not os.path.exists(temp_dir)
    assert os.environ.get('JOBLIB_TEMP_FOLDER') == original_temp
