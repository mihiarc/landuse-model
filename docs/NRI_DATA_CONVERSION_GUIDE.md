# NRI 2017 Data Conversion Guide

## Overview

This guide explains how to convert the NRI 2017 dataset (`nri17_csv.csv`) into the format required for land use modeling.

## Data Structure

### NRI 2017 Dataset
- **Size**: 3.2 GB
- **Columns**: 1,317 total
- **Key columns**:
  - `fips`: County FIPS code
  - `xfact`: Expansion factor (survey weight)
  - `landu{YY}`: Land use for year YY (e.g., landu10 for 2010)
  - `lcc{YY}`: Land capability class for year YY
- **Years available**: 1979-2017 (not all years)

### Land Use Code Mapping

NRI codes are mapped to our system as follows:

| NRI Codes | Our Code | Land Use Type |
|-----------|----------|---------------|
| 11-15, 140 | 1 | Cropland |
| 210, 341, 342 | 2 | Pasture/Range |
| 401, 410 | 3 | Forest |
| 613, 640, 650, 700-910 | 4 | Urban/Developed |

## Conversion Process

### Quick Start

```bash
# Convert sample (50,000 rows)
uv run python convert_nri_data.py --sample 50000 --output data/processed_sample

# Convert full dataset (may take time)
uv run python convert_nri_data.py --output data/processed_full

# Custom chunk size for memory management
uv run python convert_nri_data.py --chunk-size 5000 --output data/processed
```

### What the Conversion Does

1. **Extracts Transitions**: Compares land use between years (e.g., 2010→2012)
2. **Maps Land Use Codes**: Converts NRI codes to our 4-category system
3. **Handles LCC**: Cleans and standardizes land capability classes
4. **Creates Geographic Reference**: Assigns regions based on state FIPS
5. **Generates Files**:
   - `start_crop.csv`: Transitions starting from cropland
   - `start_pasture.csv`: Transitions starting from pasture
   - `start_forest.csv`: Transitions starting from forest
   - `forest_georef.csv`: Geographic reference
   - `nr_clean_5year_normals.csv`: Net returns (currently synthetic)

## Output Statistics

From a 50,000 row sample:
- **Transitions created**: 37,820
- **Counties covered**: 99
- **Transition matrix** (showing stability):
  ```
  From\To   Crop  Pasture Forest Urban
  Crop      99.2%   0.3%   0.0%  0.5%
  Pasture    0.2%  99.6%   0.0%  0.2%
  Forest     0.0%  12.9%  87.0%  0.1%
  Urban      0.0%   0.3%   0.0% 99.7%
  ```

## Data Requirements Still Needed

### 1. Net Returns Data
Currently using synthetic data. You need actual economic data with:
- Crop net returns ($/acre)
- Pasture net returns ($/acre)
- Forest net returns ($/acre)
- Urban land values ($/acre)

### 2. Climate Impact Data (Optional)
For climate change analysis:
- Changes in net returns under climate scenarios
- By county and land use type

## Running the Analysis

After conversion:

```bash
# Test with converted data
uv run python -m landuse.main full data/processed_sample/config.json

# Or run individual components
uv run python -m landuse.main explore data/processed_sample/start_crop.csv
uv run python -m landuse.main estimate \
    data/processed_sample/forest_georef.csv \
    data/processed_sample/nr_clean_5year_normals.csv \
    data/processed_sample/start_crop.csv \
    data/processed_sample/start_pasture.csv \
    data/processed_sample/start_forest.csv
```

## Memory Management

For the full 3.2GB file:

1. **Use chunking**: Default chunk size is 10,000 rows
2. **Process by state**: Add state filtering to reduce memory
3. **Select years**: Focus on specific transition periods

## Validation

Check your converted data:

```python
import pandas as pd

# Check transition counts
crop = pd.read_csv('data/processed_sample/start_crop.csv')
print(f"Crop transitions: {len(crop)}")
print(f"Counties: {crop['fips'].nunique()}")
print(f"Transition types: {crop.groupby(['startuse', 'enduse']).size()}")

# Check geographic coverage
georef = pd.read_csv('data/processed_sample/forest_georef.csv')
print(f"Regions: {georef['region'].value_counts()}")
```

## Troubleshooting

### Out of Memory
- Reduce chunk size: `--chunk-size 5000`
- Process sample first: `--sample 10000`
- Use state-by-state processing

### Missing Transitions
- Check land use code mapping in `NRI_LANDUSE_MAPPING`
- Verify years have data (not all years present in NRI)
- Some transitions may be rare or non-existent

### Invalid LCC Values
- NRI uses various formats (1, 2e, 3w, etc.)
- Script extracts first digit and defaults to 4 if missing
- Review `clean_lcc()` function for adjustments

## Next Steps

1. **Add Real Net Returns**: Replace synthetic data with actual economic data
2. **Validate Mappings**: Verify land use code interpretations with NRI documentation
3. **Scale Testing**: Test with increasingly larger samples before full processing
4. **Add Climate Data**: Incorporate climate change scenarios if available

## Files Created

```
data/processed_sample/
├── config.json                 # Pipeline configuration
├── forest_georef.csv           # Geographic reference
├── nr_clean_5year_normals.csv  # Net returns (synthetic)
├── start_crop.csv              # Crop transitions
├── start_forest.csv            # Forest transitions
└── start_pasture.csv           # Pasture transitions
```

Ready for analysis with: `uv run python -m landuse.main full data/processed_sample/config.json`