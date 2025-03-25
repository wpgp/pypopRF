# pypopRF

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://wpgp.github.io/pypopRF/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/886682636.svg)](https://doi.org/10.5281/zenodo.15085988)


pypopRF is a Python package for population prediction and dasymetric mapping using machine learning techniques. It provides a comprehensive toolkit for processing geospatial data, training models, and generating high-resolution population distribution maps.

## Features

- Feature extraction from multiple geospatial covariates
- Random Forest-based population prediction with automatic feature selection
- Parallel processing support for large datasets
- Dasymetric mapping for high-resolution population redistribution
- Visualization tools for analysis and validation
- Command-line interface for easy project management

## Quick Installation

```bash
pip install pypoprf
```

## Documentation

Full documentation is available at [https://wpgp.github.io/pypopRF/](https://wpgp.github.io/pypopRF/)

The documentation includes:
- Detailed installation instructions
- Usage guide and examples
- Input data requirements
- Configuration options
- Troubleshooting guide

## Basic Usage

### Initialize Project

```bash
# Create a new project
pypoprf init my_project
```

### Run Analysis

```bash
# Run with configuration file
pypoprf run -c my_project/config.yaml
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/wpgp/pypopRF.git
cd pypopRF
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows
```

3. Install in development mode:
```bash
pip install -e ".[dev,docs]"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use pypopRF in your research, please cite:

```bibtex
@software{pypoprf2025,
  author = {Priyatikanto R., Nosatiuk B., Zhang W., McKeen T., Vataga E., Tejedor-Garavito N, Bondarenko M.},
  title = {pypopRF: Population Prediction and Dasymetric Mapping Tool},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/wpgp/pypopRF}
}
```

## Acknowledgments

- Developed by WorldPop SDI
