# tests/utils/test_matplotlib_utils.py
import matplotlib.pyplot as plt
import pytest
from pypoprf.utils.matplotlib_utils import non_interactive_backend, with_non_interactive_matplotlib


def test_non_interactive_backend():
    """Test non-interactive backend context manager."""
    with non_interactive_backend():
        plt.figure()
        plt.plot([1, 2, 3])
        # Verify backend is non-interactive
        assert not plt.isinteractive()

    # Verify cleanup
    assert len(plt.get_fignums()) == 0


@with_non_interactive_matplotlib
def test_decorated_function():
    """Test matplotlib decorator."""
    plt.figure()
    plt.plot([1, 2, 3])
    assert not plt.isinteractive()

# After function exits, plot should be cleaned up
assert len(plt.get_fignums()) == 0
