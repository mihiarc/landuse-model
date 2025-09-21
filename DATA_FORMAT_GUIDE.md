# Land Use Modeling Data Format Guide

## Overview

This guide describes the expected data format for the land use modeling estimation system. Use the provided data generator and converter utilities to prepare your data.

## Required Data Files

### 1. Geographic Reference (`forest_georef.csv`)

**Purpose**: Maps counties to regions for geographic analysis.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| fips | integer | 5-digit county FIPS code | 1001 |
| county_fips | integer | Alternative name (optional) | 1001 |
| subregion | string | Subregion code | "NE", "SE" |
| region | string | Major region code | "NO", "SO", "WE", "MW" |

### 2. Net Returns Data (`nr_clean_5year_normals.csv`)

**Purpose**: Economic returns for different land uses by county and year.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| fips | integer | County FIPS code | 1001 |
| year | integer | Year of observation | 2010 |
| nr_cr | float | Crop net returns ($/acre) | 250.50 |
| nr_ps | float | Pasture net returns ($/acre) | 80.25 |
| nr_fr | float | Forest net returns ($/acre) | 40.00 |
| nr_ur | float | Urban net returns ($/acre) | 5000.00 |

**Optional columns**: `nrmean_*` (5-year means), `nrchange_*` (changes)

### 3. Land Use Transition Files

Three files split by starting land use:
- `start_crop.csv` - Transitions starting from cropland
- `start_pasture.csv` - Transitions starting from pasture
- `start_forest.csv` - Transitions starting from forest

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| fips | integer | County FIPS code | 1001 |
| year | integer | Year of transition | 2012 |
| riad_id | string | Unique plot identifier | "1001_001" |
| startuse | integer | Starting land use code | 1 |
| enduse | integer | Ending land use code | 2 |
| lcc | integer | Land capability class (1-8) | 3 |
| xfact | float | Expansion factor (survey weight) | 100.5 |

**Land Use Codes**:
- 1 = Cropland
- 2 = Pasture/Grassland
- 3 = Forest
- 4 = Urban/Developed

### 4. Climate Impact Data (Optional)

Directory: `cc_impacts/`

- `crop_climate_change_impact.csv`
- `forest_climate_change_impact.csv`
- `urban_climate_change_impact.csv`

Each file contains:
- `GEOID` or `fips`: County identifier
- `nr` or `nracre`: Current net returns
- `impact`: Climate-induced change in net returns

## Important Notes

### Data Type Requirements
- **FIPS codes**: Must be integers (no leading zeros in CSV)
- **Years**: Integer years (e.g., 2010, not "2010")
- **Land use codes**: Integers 1-4
- **Net returns**: Numeric values in dollars per acre
- **Weights**: Positive numeric values

### Time Lag Relationship
⚠️ **Critical**: Net returns data is lagged by 2 years
- Net returns year + 2 = Transition year
- Example: 2010 net returns → 2012 transitions

### Missing Values
- The system will handle missing values but may drop observations
- Ensure complete data for key variables: fips, year, enduse, lcc, xfact

## Using the Data Tools

### 1. Generate Test Data
```bash
uv run python -m landuse.data_generator test_data
```

### 2. Convert Existing Data
```python
from landuse.data_converter import DataConverter, convert_existing_data

# Create conversion config
config = {
    "georef": {
        "file": "counties.csv",
        "fips_col": "FIPS"
    },
    "net_returns": {
        "file": "returns.csv",
        "columns": {
            "nr_cr": "crop_revenue",
            "nr_ps": "pasture_revenue",
            "nr_fr": "forest_revenue",
            "nr_ur": "urban_value"
        }
    },
    "transitions": {
        "file": "landuse_changes.csv",
        "start_col": "from_use",
        "end_col": "to_use",
        "lcc_col": "land_quality",
        "weight_col": "sample_weight"
    }
}

convert_existing_data("input_dir", "output_dir", config)
```

### 3. Validate Data Format
```python
from landuse.data_generator import validate_data_format

is_valid, errors = validate_data_format("data_dir")
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

## Running the Estimation

Once data is properly formatted:

```bash
# Create config file
cat > config.json << EOF
{
    "georef_file": "data/forest_georef.csv",
    "nr_data_file": "data/nr_clean_5year_normals.csv",
    "start_crop_file": "data/start_crop.csv",
    "start_pasture_file": "data/start_pasture.csv",
    "start_forest_file": "data/start_forest.csv",
    "years": [2010, 2011, 2012],
    "output_dir": "results"
}
EOF

# Run estimation
uv run python -m landuse.main full config.json
```

## Troubleshooting

### Common Issues

1. **"The column label 'fips' is not unique"**
   - Check for duplicate column names in georef file
   - Remove `county_fips` if it duplicates `fips`

2. **"All arrays must be of the same length"**
   - Ensure all columns in each file have the same number of rows
   - Check for incomplete data processing

3. **Missing net returns after merge**
   - Verify the 2-year lag relationship
   - Check that FIPS codes match between files

4. **Model convergence failures**
   - Check for sufficient variation in land use transitions
   - Ensure adequate sample size per county
   - Verify net returns are reasonable values

## Example Data Structure

```
data/
├── forest_georef.csv          # 100 counties
├── nr_clean_5year_normals.csv # 500 obs (100 counties × 5 years)
├── start_crop.csv             # ~3000 transitions
├── start_pasture.csv          # ~3000 transitions
├── start_forest.csv           # ~3000 transitions
└── cc_impacts/
    ├── crop_climate_change_impact.csv
    ├── forest_climate_change_impact.csv
    └── urban_climate_change_impact.csv
```

## Support

For issues or questions:
- Check generated `test_data/data_format_spec.json` for detailed specifications
- Review test data examples in `test_data/` directory
- See GitHub issues for known problems and solutions