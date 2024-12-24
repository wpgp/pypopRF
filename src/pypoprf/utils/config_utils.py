# src/pypoprf/utils/config_utils.py
from pathlib import Path
from typing import Dict, Any
import yaml


def create_config_template(output_path: Path,
                           data_dir: str = "data",
                           prefix: str = "test_") -> None:
    """
    Create a configuration file template with proper paths.

    Args:
        output_path: Path where config file will be saved
        data_dir: Name of directory for input data files
        prefix: Prefix added to input filenames

    Raises:
        ValueError: If arguments are invalid
        OSError: If file cannot be written
    """
    # Validate inputs
    if not isinstance(output_path, (str, Path)):
        raise ValueError("output_path must be string or Path")
    output_path = Path(output_path)

    if not isinstance(data_dir, str):
        raise ValueError("data_dir must be string")
    if not isinstance(prefix, str):
        raise ValueError("prefix must be string")

    # Define default configuration
    config: Dict[str, Any] = {
        'work_dir': ".",
        'data_dir': data_dir,
        'covariates': {
            'cnt': f"{prefix}buildingCount.tif",
            'srf': f"{prefix}buildingSurface.tif",
            'vol': f"{prefix}buildingVolume.tif"
        },
        'mastergrid': f"{prefix}mastergrid.tif",
        'mask': f"{prefix}mask.tif",
        'constrain': f"{prefix}constrain.tif",
        'census_data': f"{prefix}admin3.csv",
        'census_pop_column': "pop",
        'census_id_column': "id",
        'output_dir': "output",
        'block_size': [512, 512],
        'max_workers': 8,
        'show_progress': True
    }

    header_comment = """# pypopRF configuration file
#
# Configuration options:
#   work_dir: Root directory for project
#   data_dir: Directory containing input files
#   covariates: Dictionary of covariate raster files
#   mastergrid: Path to mastergrid file
#   mask: Path to mask file
#   constrain: Path to file for constraining population dist.
#   census_data: Path to census data file
#   census_pop_column: Population column in census data
#   census_id_column: ID column in census data
#   output_dir: Directory for output files
#   block_size: Processing block size [width, height]
#   max_workers: Number of parallel workers
#   show_progress: Whether to show progress bars
#
# Example covariates:
#   covariates:
#     key1: "path/to/raster1.tif"
#     key2: "path/to/raster2.tif"
#
"""

    try:
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write config file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header_comment)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    except Exception as e:
        raise OSError(f"Failed to write config file: {str(e)}")
