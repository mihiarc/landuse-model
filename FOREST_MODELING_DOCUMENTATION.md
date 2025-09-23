# Forest Land Use Modeling Documentation

## Overview
This document describes the forest land use modeling implementation using NRI BROAD codes and the LCC-only logit model.

## Land Use Codes (NRI BROAD)

Based on the 2017 NRI CSV File Layout, the correct land use codes are:

| Code | Description | Model Use |
|------|-------------|-----------|
| 1 | Cultivated cropland | ✓ Primary |
| 2 | Noncultivated cropland | - |
| 3 | Pastureland | ✓ Primary |
| 4 | Rangeland | - |
| **5** | **Forest land** | **✓ Primary** |
| 6 | Minor land | - |
| 7 | Urban and built-up land | ✓ Primary |
| 8 | Rural transportation | - |
| 9 | Small water areas | - |
| 10 | Large water areas | - |
| 11 | Federal land | - |
| 12 | Conservation Reserve Program (CRP) | - |

**Important**: Forest is code **5**, not 3 or 7!

## Land Capability Class (LCC)

- LCC ranges from 1-8
- **Higher LCC = Better quality land** for agricultural use
- Expected relationships:
  - Crops prefer high LCC land (6-8)
  - Pasture uses medium LCC land (4-6)
  - Forest typically occupies lower LCC land (1-4)
  - Urban development is less sensitive to LCC

## Model Structure

### LCC-Only Model Formula
```
enduse ~ lcc
```

This simplified model predicts land use transitions based solely on land capability class.

### Expected Parameter Signs

For transitions TO each land use:

1. **To Crop** (code 1):
   - Expected: **Positive** LCC coefficient
   - Interpretation: Higher probability of crop use on better (higher LCC) land

2. **To Pasture** (code 3):
   - Expected: **Mixed/Moderate positive** LCC coefficient
   - Interpretation: Flexible land use, slight preference for better land

3. **To Forest** (code 5):
   - Expected: **Negative** LCC coefficient
   - Interpretation: Higher probability of forest on poorer (lower LCC) land

4. **To Urban** (code 7):
   - Expected: **Weak/Mixed** LCC coefficient
   - Interpretation: Less sensitive to agricultural land quality

## Transition Patterns

### From Forest Land
Typical transition probabilities based on land quality:

- **High LCC land (6-8)**: More likely to convert to agriculture
  - Forest → Crop: ~15%
  - Forest → Pasture: ~10%
  - Forest → Forest: ~70%
  - Forest → Urban: ~5%

- **Low LCC land (1-3)**: Forest tends to persist
  - Forest → Crop: ~2%
  - Forest → Pasture: ~3%
  - Forest → Forest: ~94%
  - Forest → Urban: ~1%

### To Forest Land
Factors favoring conversion to forest:
- Low LCC (poor agricultural land)
- Abandonment of marginal cropland
- Conservation programs
- Natural succession from pasture

## Implementation Details

### Data Generator Updates
The `data_generator.py` has been updated to:
1. Use correct NRI BROAD codes (1, 3, 5, 7)
2. Implement realistic transition probabilities based on LCC
3. Generate test data that reflects economic theory

### Model Estimation
The `logit_estimation.py` module:
1. Supports both LCC-only and full models with net returns
2. Handles NRI BROAD codes correctly
3. Estimates multinomial logit models for each starting land use

### Testing
The forest modeling has been tested with:
- Correct code verification (forest = 5)
- Transition pattern analysis
- Parameter sign evaluation
- Consistency with economic theory

## Usage Examples

### Estimating Forest Transitions
```python
from landuse.logit_estimation import estimate_land_use_transitions

# Estimate LCC-only model including forest transitions
models = estimate_land_use_transitions(
    start_crop, start_pasture, start_forest,
    nr_data=nr_data,
    georef=georef,
    years=[2010, 2011, 2012],
    use_net_returns=False  # LCC-only model
)
```

### Analyzing Forest Persistence
```python
# Check forest → forest transitions
forest_model = models['foreststart_all']
lcc_params = forest_model.params.loc['lcc']

# Forest persistence should show negative LCC coefficient
# (forest persists on lower quality land)
```

## Key Findings from Testing

1. **Code Verification**: ✓ Forest correctly uses code 5
2. **Data Generation**: ✓ Realistic transition patterns based on LCC
3. **Model Convergence**: ✓ All models converge successfully
4. **Parameter Signs**: Mixed results due to synthetic data limitations

## Notes on Model Output

The multinomial logit model may internally recode categories (0, 1, 2, ...) for estimation. The mapping between internal codes and NRI BROAD codes should be maintained through proper data handling.

## Future Enhancements

1. **Include additional NRI codes**: Rangeland (4), CRP (12)
2. **Spatial factors**: Distance to urban areas, protected lands
3. **Policy variables**: Conservation programs, carbon credits
4. **Climate factors**: Temperature, precipitation changes
5. **Economic factors**: Timber prices, carbon markets

## References

- 2017 NRI CSV File Layout (nri17_csv_layout_050521.xlsx)
- USDA Natural Resources Conservation Service
- Land Capability Classification System