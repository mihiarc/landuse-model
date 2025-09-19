# Land Use Modeling

Python package for analyzing land use transitions, climate impacts, and policy effects using discrete choice models.

## Overview

This package provides tools for:
- Estimating multinomial logit models for land use transitions
- Calculating marginal effects and elasticities
- Analyzing climate change impacts on land use decisions
- Exploring spatial and temporal patterns in land use data
- Visualizing results through maps and charts

## Installation

### Using uv (recommended)

```bash
# Create virtual environment and install in development mode
uv venv
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

1. Create a configuration file:
```bash
uv run python -m landuse.main create-config
```

2. Edit the configuration file with your data paths

3. Run the full analysis pipeline:
```bash
uv run python -m landuse.main full config.json
```

## Project Structure

```
landuse-modeling/
├── src/
│   └── landuse/
│       ├── __init__.py
│       ├── main.py              # Main entry point and CLI
│       ├── climate_impact.py    # Climate change impact analysis
│       ├── logit_estimation.py  # Discrete choice model estimation
│       ├── marginal_effects.py  # Marginal effects and elasticities
│       └── data_exploration.py  # Data exploration utilities
├── tests/                        # Test suite
├── data/                         # Data directory (not in git)
├── archive/                      # Original R scripts
├── pyproject.toml               # Project configuration
└── README.md
```

## Original R Scripts

This package is a Python port of R scripts for land use modeling. The original R scripts are preserved in the `archive/` directory.
