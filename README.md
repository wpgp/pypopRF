# PopuPy

PopuPy is a Python package for population prediction and dasymetric mapping using machine learning techniques. It provides a comprehensive toolkit for processing geospatial data, training models, and generating high-resolution population distribution maps.

## Features

- Feature extraction from multiple geospatial covariates
- Random Forest-based population prediction with automatic feature selection
- Parallel processing support for large datasets
- Dasymetric mapping for high-resolution population redistribution
- Visualization tools for analysis and validation
- Command-line interface for easy project management

## Installation and Setup

### Quick Installation

```bash
pip install popupy
```

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/wpgp/pypopRF.git
cd PopuPyTool
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows (cmd.exe)
venv\Scripts\activate.bat

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1
```

3. Install in development mode:
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install with additional extras
pip install -e ".[dev,docs]"
```

### System Dependencies

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc
```

#### MacOS
```bash
brew install gdal
```

#### Windows
- Install OSGeo4W from https://trac.osgeo.org/osgeo4w/
- Add GDAL paths to system environment variables

## Quick Start

### Initialize a New Project

```bash
# Create a new project with default covariates
popupy init my_project

# Create a project with custom covariates
popupy init my_project --covariates population --covariates elevation --covariates slope
```

Initialization flags:
- `project_dir`: (Required) Directory path for the new project
- `--data-dir`: Name of directory for input data files (default: "data")
- `--prefix`: Prefix added to input filenames (default: "test_")
- `--covariates`, `-cov`: Names of covariate datasets. Can be specified multiple times.

### Run Population Modeling

```bash
# Run with default settings
popupy run -c config.yaml

# Run with verbose output and skip visualization
popupy run -c config.yaml -v --no-viz
```

Running flags:
- `-c`, `--config`: (Required) Path to configuration YAML file
- `-v`, `--verbose`: Enable detailed output logging
- `--no-viz`: Skip visualization generation

## Project Structure

After initialization:
```
my_project/
├── config.yaml          # Configuration file
├── data/               # Input data directory
│   ├── mastergrid.tif
│   ├── covariate1.tif
│   ├── covariate2.tif
│   └── census.csv
└── output/             # Results directory
    ├── features.csv
    ├── model.pkl.gz
    ├── scaler.pkl.gz
    ├── prediction.tif
    ├── normalized_census.tif
    ├── dasymetric.tif
    └── visualization.png
```

## Configuration

Example `config.yaml`:
```yaml
# Working directory configuration
work_dir: "."
data_dir: "data"

# Input data paths
covariates:
  cnt: "buildingCount.tif"
  srf: "buildingSurface.tif"
  vol: "buildingVolume.tif"
mastergrid: "mastergrid.tif"
census_data: "census.csv"

# Census data columns
census_pop_column: "pop"
census_id_column: "id"

# Processing parameters
block_size: [512, 512]
max_workers: 8
show_progress: true
```

## Input Data Requirements

### Covariates
- Format: GeoTIFF
- Must share the same:
  - Coordinate Reference System (CRS)
  - Resolution
  - Extent
  - Dimensions

### Census Data
- Format: CSV
- Required columns:
  - Population count
  - Zone ID (matching mastergrid zones)

### Mastergrid
- Format: GeoTIFF
- Contains zone IDs matching census data
- Aligned with covariates

## Output Files

- `features.csv`: Extracted features for model training
- `model.pkl.gz`: Trained Random Forest model
- `scaler.pkl.gz`: Fitted feature scaler
- `feature_selection.png`: Feature importance visualization
- `prediction.tif`: Raw population probability surface
- `normalized_census.tif`: Normalized census populations
- `dasymetric.tif`: Final high-resolution population distribution
- `visualization.png`: Multi-panel visualization of results

## Python API Usage

```python
from popupy.config import Settings
from popupy.core import FeatureExtractor, Model, DasymetricMapper

# Initialize settings
settings = Settings.from_file('config.yaml')

# Extract features
extractor = FeatureExtractor(settings)
features = extractor.extract()

# Train model and predict
model = Model(settings)
model.train(features)
prediction_path = model.predict()

# Perform dasymetric mapping
mapper = DasymetricMapper(settings)
result_path = mapper.map(prediction_path)
```

## Docker Support

### Build Image
```bash
docker build -t popupy .
```

### Run Container
```bash
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    popupy run -c config.yaml
```

## Development

### Running Tests
```bash
# Run tests with coverage
pytest tests/ --cov=popupy
```

### Code Quality
```bash
# Format code
black src/
isort src/

# Run linters
ruff check src/
mypy src/
```

### Building Documentation
```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html
```

## Troubleshooting

### Common Issues

1. GDAL Import Errors:
   - Ensure GDAL is installed system-wide
   - Check Python GDAL bindings match system version

2. Memory Issues:
   - Adjust block_size in config.yaml
   - Reduce max_workers for parallel processing

3. CRS Mismatches:
   - Ensure all input rasters share same CRS
   - Use gdalwarp to reproject if needed

### Getting Help

- Documentation: https://popupy.readthedocs.io/
- Issues: https://github.com/wpgp/pypopRF/issues
- Discussions: https://github.com/wpgp/pypopRF/discussions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PopuPy in your research, please cite:

```bibtex
@software{popupy2024,
  author = {WorldPop SDI},
  title = {PopuPy: Population Prediction and Dasymetric Mapping Tool},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/wpgp/pypopRF}
}
```

## Acknowledgments

- Developed by WorldPop SDI