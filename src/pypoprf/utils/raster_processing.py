from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Callable

from rasterio.windows import Window
from tqdm import tqdm

from .logger import get_logger

logger = get_logger()


def progress_bar(iterable: Any,
                 show: bool,
                 total: int,
                 desc: str = "Processing") -> Any:
    """
    Create a progress bar for iteration.

    Args:
        iterable: Iterable object
        show: Whether to show progress bar
        total: Total number of items
        desc: Description for progress bar

    Returns:
        Progress bar wrapped iterable
    """
    if show:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def parallel(windows: List[Window],
             process_func: Callable,
             max_workers: int,
             show_progress: bool = True,
             desc: str = "Processing") -> List[Any]:
    """
    Generic parallel processing function.

    Args:
        windows: List of windows to process
        process_func: Function to process each window
        max_workers: Maximum number of worker threads
        show_progress: Whether to show progress bar
        desc: Description for progress bar

    Returns:
        List of results from processing each window
    """

    logger.debug(f"Processing {len(windows)} windows")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(progress_bar(
            executor.map(process_func, windows),
            show_progress,
            len(windows),
            desc=desc
        ))

    return results
