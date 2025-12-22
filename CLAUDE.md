# Land Use Modeling Project

This project replicates and extends the econometric land use change modeling framework developed in Mihiar (2018) and published in Mihiar & Lewis (2023).

## Theoretical Foundation

### Source Literature

**Primary References:**
- Mihiar, C. (2018). "Three essays on the economics of climate change, land use and carbon sequestration." PhD Dissertation, Oregon State University. [ScholarsArchive@OSU](https://ir.library.oregonstate.edu/concern/graduate_thesis_or_dissertations/d791sk216)
- Mihiar, C.M., and D.J. Lewis (2023). "An empirical analysis of U.S. land-use change under multiple climate change scenarios." Journal of the Agricultural and Applied Economics Association, 2: 597-611. [DOI: 10.1002/jaa2.82](https://onlinelibrary.wiley.com/doi/10.1002/jaa2.82)
- Mihiar, C., and D.J. Lewis (2021). "Climate, adaptation, and the value of forestland: A national Ricardian analysis of the United States." Land Economics, 97(4): 911-932.

**Foundational Papers:**
- Lubowski, R.N., A.J. Plantinga, and R.N. Stavins (2006). "Land-use change and carbon sinks: Econometric estimation of the carbon sequestration supply function." Journal of Environmental Economics and Management, 51(2): 135-152.
- Lubowski, R.N., A.J. Plantinga, and R.N. Stavins (2008). "What drives land-use change in the United States? A national analysis of landowner decisions." Land Economics, 84(4): 529-550.

## Model Architecture

### Overview

The model integrates two econometric approaches:

1. **Ricardian Estimation**: Links climate variables to net economic returns for each land use (crops, pasture, forest, urban development)
2. **Discrete Choice Model**: Estimates plot-level land use transition probabilities based on returns and land characteristics

### Land Use Categories

The model tracks transitions between five major land use types:
- **Cropland (CR)**: Agricultural row crops
- **Pasture (PS)**: Grazing land and hay
- **Forest (FR)**: Timberland and woodland
- **Urban/Developed (UR)**: Residential, commercial, industrial
- **Conservation Reserve Program (CRP)**: Federally enrolled conservation land

### Data Sources

**National Resources Inventory (NRI)**:
- Plot-level land use observations at 5-year intervals
- ~800,000 sample points across non-federal US lands
- Variables: land use, Land Capability Class (LCC), expansion factors (weights)
- Time series: 1982-present

**Geographic and Economic Data**:
- County-level FIPS codes for spatial aggregation
- Regional classifications (North, South, etc.)
- Net returns by land use type and county

## Econometric Specification

### Discrete Choice Framework

Land use change is modeled as a discrete choice where landowners select the use that maximizes expected net returns. For each starting land use j, estimate a separate model:

```
P(end_use = k | start_use = j) = f(net_returns_k, LCC, regional_fixed_effects)
```

### Multinomial Logit Specification

The probability that plot i transitions from use j to use k:

```
P_ijk = exp(V_ijk) / Σ_m exp(V_ijm)

where:
V_ijk = β_k * X_i + γ_k * NR_ik
```

- `X_i`: Plot characteristics (LCC, location)
- `NR_ik`: Net returns to use k in county of plot i
- `β_k`, `γ_k`: Parameters to estimate

### Nested Logit Extension

Following Lubowski et al. (2006), the model can be extended to a nested logit structure to relax Independence of Irrelevant Alternatives (IIA):

```
P_ijk = P(k|nest) * P(nest)
```

Typical nesting structure:
- Nest 1: Agricultural uses (crop, pasture)
- Nest 2: Non-agricultural uses (forest, urban, CRP)

### Model by Starting Use

Four separate models are estimated:
1. **Crop Start**: Land beginning in cropland
2. **Pasture Start**: Land beginning in pasture
3. **Forest Start**: Land beginning in forest
4. **CRP Start**: Land enrolled in conservation programs

## Key Variables

### Land Capability Class (LCC)

Primary physical land quality measure from USDA soil surveys:
- **Class 1-2**: Prime agricultural land
- **Class 3-4**: Moderate limitations
- **Class 5-8**: Severe limitations for agriculture

### Net Returns

County-level economic returns by use:
- `nr_cr`: Crop net returns ($/acre)
- `nr_ps`: Pasture net returns ($/acre)
- `nr_fr`: Forest net returns ($/acre)
- `nr_ur`: Urban net returns ($/acre)

Net returns are lagged 2 periods to capture landowner decision timing.

### Climate Variables (for projections)

- Temperature: Mean annual, seasonal variation
- Precipitation: Annual total, growing season
- Derived indices: Growing degree days, Palmer Drought Severity Index

## Key Findings from Original Research

### Projected Changes by 2070

| Land Use | Projected Change |
|----------|------------------|
| Developed | +51% (~0.82M acres/year) |
| Cropland | -5.6% |
| Pasture | -7.9% |
| Forest | -2.3% |

### Climate Scenario Effects

| Climate Scenario | Favored Land Use |
|------------------|------------------|
| Drier + Warmer | Forest |
| Wetter + Cooler | Developed |
| Wetter + Warmer | Cropland |

### Spatial Patterns

- 62% of new development occurs in currently nonmetropolitan counties
- Forest gains concentrated in marginal agricultural regions
- Agricultural intensification on prime lands

## Implementation Notes

### Current Simplifications

This implementation supports both:
1. **LCC-only model**: Uses Land Capability Class as sole predictor (default)
2. **Full model**: Includes net returns as additional predictors

### Regional Stratification

Models can be estimated for:
- All regions combined
- Eastern regions (North + South)
- Northern region only
- Southern region only

### Weighting

NRI expansion factors (`xfact`) are used to weight observations to represent the full population of non-federal US land.

## Updates and Extensions

This project aims to update the original modeling framework with:

1. **Extended time series**: Incorporate NRI data through 2022
2. **Updated climate projections**: CMIP6 scenarios
3. **Additional land uses**: Separate solar/renewable energy development
4. **Improved net returns**: Updated agricultural and forestry return estimates
5. **Enhanced spatial resolution**: County-level projections
6. **Policy scenarios**: Conservation program effects, carbon markets

## Running the Model

```bash
# Install dependencies
uv pip install -e ".[dev]"

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
├── src/landuse/
│   ├── logit_estimation.py    # Core discrete choice models
│   ├── climate_impact.py      # Climate scenario analysis
│   ├── marginal_effects.py    # Elasticity calculations
│   ├── crp_enrollment.py      # CRP-specific modeling
│   └── data_converter.py      # NRI data processing
├── tests/                     # Test suite
├── archive/                   # Original R scripts
└── examples/                  # Usage examples
```

## Code Style

- Use `uv` for package management and virtual environments
- Use `pydantic` for data validation
- Use `rich` for terminal output
- Always run tests with real API calls (no mocking)
- Descriptive variable names
- Regular git commits
