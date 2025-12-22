# Land Use Change Model

Discrete choice model for analyzing land use transitions across the contiguous United States, based on Mihiar (2018) and Mihiar & Lewis (2023).

## Overview

This package implements the econometric land use change model that estimates transition probabilities between major land use categories (cropland, pasture, forest, urban, CRP) using:
- Multinomial/nested logit discrete choice models
- National Resources Inventory (NRI) plot-level data
- Land Capability Class (LCC) as primary land quality measure
- Net returns from companion rent estimation models

## Related Repositories

This model is part of a multi-repo system:

| Repository | Purpose |
|------------|---------|
| **landuse-modeling** (this repo) | Land use transition discrete choice model |
| [ag-rents](../ag-rents) | Agricultural land rent estimation (cropland, pasture) |
| [urban-rents](../urban-rents) | Urban/developed land rent estimation |
| [forest-rents](../forest-rents) | Forest land rent estimation |

The rent models provide county-level net returns that serve as inputs to this land use change model.

## Installation

```bash
# Clone repository
git clone <repository-url>
cd landuse-modeling

# Create virtual environment and install
uv venv
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Generate test data
uv run python -m landuse.data_generator

# Run estimation
uv run python -m landuse.main full config.json

# Run tests
uv run pytest tests/
```

## Project Structure

```
landuse-modeling/
├── src/landuse/              # Core Python package
│   ├── logit_estimation.py   # Discrete choice model estimation
│   ├── climate_impact.py     # Climate scenario analysis
│   ├── marginal_effects.py   # Elasticity calculations
│   ├── crp_enrollment.py     # CRP-specific modeling
│   ├── data_converter.py     # NRI data processing
│   └── main.py               # CLI entry point
├── tests/                    # Test suite
├── docs/                     # Technical documentation
├── scripts/                  # Utility and analysis scripts
├── data/                     # Test and sample data
├── reference/                # NRI layout and specs
├── archive/                  # Original R scripts
└── examples/                 # Usage examples
```

## Theoretical Foundation

See [CLAUDE.md](CLAUDE.md) for detailed documentation on:
- Model architecture and econometric specification
- Data sources and variables
- Key findings from original research
- Planned updates and extensions

## Key References

- Mihiar, C. (2018). "An Econometric Analysis of the Impact of Climate Change on Forest Land Value and Broad Land-use Change." PhD Dissertation, Oregon State University.
- Mihiar, C.M., and D.J. Lewis (2023). "An empirical analysis of U.S. land-use change under multiple climate change scenarios." Journal of the Agricultural and Applied Economics Association, 2: 597-611.
- Lubowski, R.N., A.J. Plantinga, and R.N. Stavins (2006). "Land-use change and carbon sinks." Journal of Environmental Economics and Management, 51(2): 135-152.

## License

MIT
