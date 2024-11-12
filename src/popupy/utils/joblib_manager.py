import tempfile
import shutil
import os
from typing import Generator
from contextlib import contextmanager


@contextmanager
def joblib_resources() -> Generator[str, None, None]:
    """
    Context manager for handling joblib temporary resources.

    Creates a temporary directory for joblib to use and ensures cleanup
    after processing is complete.

    Yields:
        str: Path to temporary directory
    """

    temp_folder = tempfile.mkdtemp(prefix='popupy_joblib_')
    old_temp = os.environ.get('JOBLIB_TEMP_FOLDER')

    try:
        os.environ['JOBLIB_TEMP_FOLDER'] = temp_folder
        yield temp_folder

    finally:
        # Restore original environment
        if old_temp is not None:
            os.environ['JOBLIB_TEMP_FOLDER'] = old_temp
        else:
            os.environ.pop('JOBLIB_TEMP_FOLDER', None)

        try:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"Failed to cleanup temp folder {temp_folder}: {e}")
