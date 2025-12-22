# Land Use Model - LCC-Only Simplification

## Overview
The land use model has been simplified to use only Land Capability Class (LCC) as a predictor variable, removing the dependency on net returns data. This makes the model more accessible when economic data is unavailable or unreliable.

## Key Changes

### 1. Model Formula
- **Previous**: `enduse ~ lcc + nr_cr + nr_ps + nr_fr + nr_ur`
- **New (default)**: `enduse ~ lcc`
- **Optional**: Can still use full model with `use_net_returns=True`

### 2. Modified Functions

#### `prepare_estimation_data()`
- Added `use_net_returns` parameter (default: False)
- Net returns data is now optional
- Creates categorical LCC variables for better model fit
- Merges net returns only when explicitly requested

#### `estimate_land_use_transitions()`
- Added `use_net_returns` parameter (default: False)
- Automatically selects appropriate formula based on parameter
- Provides clear logging of which model type is being used

#### `main()`
- Added `use_net_returns` parameter for command-line usage
- Reports model type in output messages

### 3. Land Capability Classes
LCC ranges from 1 (best) to 8 (worst) for agricultural suitability:
- **Classes 1-2**: Best agricultural land (good)
- **Classes 3-4**: Moderate agricultural potential (moderate)
- **Classes 5-8**: Poor agricultural land, better for forest/conservation (poor)

## Usage Examples

### Python API
```python
from landuse.logit_estimation import estimate_land_use_transitions

# LCC-only model (default)
models = estimate_land_use_transitions(
    start_crop, start_pasture, start_forest,
    nr_data=nr_data,
    georef=georef,
    years=[2010, 2011, 2012],
    use_net_returns=False  # Default
)

# Full model with net returns
models = estimate_land_use_transitions(
    start_crop, start_pasture, start_forest,
    nr_data=nr_data,
    georef=georef,
    years=[2010, 2011, 2012],
    use_net_returns=True
)
```

### Example Script
Run the provided example:
```bash
uv run python examples/lcc_only_model_example.py
```

This script:
1. Generates test data if needed
2. Estimates LCC-only models for different regions
3. Displays model statistics
4. Optionally compares with full model

## Model Interpretation

### LCC Coefficients
- **Positive coefficient**: Higher LCC (worse land) increases probability of that land use
- **Negative coefficient**: Higher LCC decreases probability of that land use

### Expected Patterns
- **Crop land**: Typically negative LCC coefficient (prefers good land)
- **Pasture**: Mixed coefficients depending on region
- **Forest**: Typically positive LCC coefficient (occupies poorer land)
- **Urban**: Less sensitive to LCC

## Benefits of LCC-Only Model

1. **Simplicity**: Fewer parameters, easier to estimate
2. **Data availability**: LCC data is widely available from soil surveys
3. **Stability**: Less sensitive to economic fluctuations
4. **Interpretability**: Direct relationship between land quality and use

## When to Use Each Model

### Use LCC-Only When:
- Net returns data is unavailable or unreliable
- Focus is on physical land suitability
- Long-term structural patterns are of interest
- Simpler model is preferred for policy analysis

### Use Full Model When:
- Economic factors are important
- Short-term market responses matter
- Net returns data is high quality
- Maximum predictive accuracy is needed

## Testing
Tests have been updated to support both model types:
```bash
uv run pytest tests/test_logit_estimation.py
```

## Files Modified
- `src/landuse/logit_estimation.py`: Core model changes
- `tests/test_logit_estimation.py`: Updated tests
- `examples/lcc_only_model_example.py`: New example script
- `src/landuse/data_generator.py`: Test data generation (unchanged)