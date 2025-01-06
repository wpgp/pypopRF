# tests/utils/test_raster.py
import numpy as np
import pandas as pd
import pytest
from pypoprf.utils.raster import get_raster_stats, aggregate_table, progress_bar


def test_get_raster_stats():
    """Test raster statistics calculation."""
    # Create test data
    target = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([[1, 1, 2], [2, 2, 2]])
    nodata = -99

    # Calculate stats
    result = get_raster_stats(target, mask, nodata=nodata)

    # Verify results
    assert isinstance(result, pd.DataFrame)
    assert set(result['id'].unique()) == {1, 2}  # Check unique IDs
    assert all(result['count'] > 0)  # Check counts are positive


def test_aggregate_table():
    """Test statistics aggregation."""
    # Create test data
    data = pd.DataFrame({
        'id': [1, 1, 2, 2],
        'count': [10, 20, 15, 25],
        'sum': [100, 200, 150, 250],
        'sum2': [1000, 2000, 1500, 2500],
        'min': [1, 2, 1, 2],
        'max': [10, 20, 15, 25]
    })

    # Aggregate
    result = aggregate_table(data, prefix='test')

    # Verify results
    assert 'test_count' in result.columns
    assert 'test_avg' in result.columns
    assert 'test_std' in result.columns
    assert len(result) == len(data['id'].unique())


def test_progress_bar():
    """Test progress bar functionality."""
    items = range(10)

    # Test with progress bar shown
    result = list(progress_bar(items, True, 10, "Test"))
    assert result == list(items)

    # Test without progress bar
    result = list(progress_bar(items, False, 10, "Test"))
    assert result == list(items)
