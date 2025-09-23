"""
Generate test datasets that reveal the expected data format for land use modeling.
This module creates sample data with all required fields and proper structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


def generate_geographic_reference() -> pd.DataFrame:
    """
    Generate geographic reference data (georef).

    Expected format:
    - fips: County FIPS code (integer, 5 digits)
    - county_fips: Alternative name for FIPS (for compatibility)
    - subregion: Subregion code (e.g., 'NE', 'SE', 'MW')
    - region: Major region code (e.g., 'NO', 'SO', 'WE')
    """
    np.random.seed(42)

    n_counties = 100

    # Generate realistic FIPS codes (state + county)
    states = [1, 5, 6, 12, 13, 17, 36, 48]  # Sample state codes
    counties = []

    for state in states:
        for county in range(1, min(20, n_counties // len(states)) + 1, 2):
            fips = state * 1000 + county
            counties.append(fips)

    # Ensure we have exactly n_counties by repeating or extending
    while len(counties) < n_counties:
        # Add more counties with higher county codes
        for state in states:
            if len(counties) >= n_counties:
                break
            county_code = (len(counties) % 100) * 2 + 21
            counties.append(state * 1000 + county_code)

    counties = counties[:n_counties]

    # Regional assignments
    regions = ['NO', 'SO', 'WE', 'MW']
    subregions = ['NE', 'SE', 'NW', 'SW', 'NC', 'SC', 'MW', 'CE']

    georef = pd.DataFrame({
        'fips': counties,
        'county_fips': counties,  # Duplicate for compatibility
        'subregion': np.random.choice(subregions, n_counties),
        'region': np.random.choice(regions, n_counties, p=[0.3, 0.3, 0.2, 0.2])
    })

    return georef


def generate_net_returns_data(years: list = [2008, 2009, 2010, 2011, 2012]) -> pd.DataFrame:
    """
    Generate net returns data.

    Expected format:
    - fips: County FIPS code
    - year: Year of observation
    - nr_cr: Net returns for crop ($/acre)
    - nr_ps: Net returns for pasture ($/acre)
    - nr_fr: Net returns for forest ($/acre)
    - nr_ur: Net returns for urban ($/acre)
    - (optional) nrmean_*, nrchange_* columns
    """
    np.random.seed(42)

    georef = generate_geographic_reference()
    data = []

    for year in years:
        for fips in georef['fips']:
            # Generate correlated net returns with realistic ranges
            base_productivity = np.random.normal(100, 20)

            nr_cr = np.random.normal(250 + base_productivity, 50)  # Crop: $150-350/acre
            nr_ps = np.random.normal(80 + base_productivity * 0.3, 20)  # Pasture: $40-120/acre
            nr_fr = np.random.normal(40 + base_productivity * 0.2, 15)  # Forest: $10-70/acre
            nr_ur = np.random.normal(5000 + base_productivity * 10, 500)  # Urban: $4000-6000/acre

            # Add some negative values for less productive areas
            if np.random.random() < 0.1:
                nr_cr = np.random.normal(-50, 20)

            data.append({
                'fips': fips,
                'year': year,
                'nr_cr': nr_cr,
                'nr_ps': nr_ps,
                'nr_fr': nr_fr,
                'nr_ur': nr_ur,
                # Optional columns for 5-year means and changes
                'nrmean_cr': nr_cr * np.random.normal(1, 0.05),
                'nrmean_ps': nr_ps * np.random.normal(1, 0.05),
                'nrmean_fr': nr_fr * np.random.normal(1, 0.05),
                'nrmean_ur': nr_ur * np.random.normal(1, 0.05),
                'nrchange_cr': np.random.normal(0, 10),
                'nrchange_ps': np.random.normal(0, 5),
                'nrchange_fr': np.random.normal(0, 3),
                'nrchange_ur': np.random.normal(0, 50)
            })

    return pd.DataFrame(data)


def generate_land_use_transitions(start_use: str = 'crop') -> pd.DataFrame:
    """
    Generate land use transition data for a specific starting use.

    Expected format:
    - fips: County FIPS code
    - year: Year of observation
    - riad_id: Plot/observation ID within county
    - startuse: Starting land use (NRI BROAD: 1=crop, 3=pasture, 5=forest, 7=urban)
    - enduse: Ending land use (same coding)
    - lcc: Land capability class (1-8, HIGHER is better quality land)
    - xfact: Expansion factor (survey weight)
    - nr.ps: Net returns for pasture at this location (optional, can be merged)
    """
    np.random.seed(42)

    georef = generate_geographic_reference()
    years = [2010, 2011, 2012]

    # Land use codes (NRI BROAD codes)
    # 1=Cultivated cropland, 3=Pastureland, 5=Forest land, 7=Urban, 12=CRP
    use_codes = {'crop': 1, 'pasture': 3, 'forest': 5, 'urban': 7, 'crp': 12}
    start_code = use_codes[start_use]

    data = []

    for year in years:
        for fips in georef['fips'][:50]:  # Subset for manageable size
            n_plots = np.random.randint(10, 30)  # Plots per county

            for plot in range(n_plots):
                # Land characteristics
                # CORRECTED: Higher LCC = better quality land
                lcc = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],
                                      p=[0.03, 0.07, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1])

                # Transition probabilities depend on land quality
                # CORRECTED: Higher LCC = better land for agriculture
                if start_use == 'crop':
                    if lcc >= 6:  # Good land (high LCC) stays in crop
                        probs = [0.85, 0.08, 0.05, 0.02]  # [crop, pasture, forest, urban]
                    else:  # Poor land (low LCC) more likely to convert to forest
                        probs = [0.60, 0.15, 0.20, 0.05]
                elif start_use == 'forest':
                    if lcc >= 6:  # Good land (high LCC) more likely to convert to ag
                        probs = [0.15, 0.10, 0.70, 0.05]  # Some conversion to crop/pasture
                    else:  # Poor land (low LCC) stays forest
                        probs = [0.02, 0.03, 0.94, 0.01]  # Forest persists on poor land
                elif start_use == 'pasture':
                    if lcc >= 7:  # Very good land might convert to crop
                        probs = [0.20, 0.65, 0.10, 0.05]
                    elif lcc <= 3:  # Poor land might revert to forest
                        probs = [0.05, 0.60, 0.30, 0.05]
                    else:  # Medium quality land
                        probs = [0.10, 0.75, 0.10, 0.05]
                else:  # urban
                    probs = [0.01, 0.01, 0.01, 0.97]  # Urban rarely converts

                # Use NRI BROAD codes for end use
                enduse = np.random.choice([1, 3, 5, 7], p=probs)

                # Survey weights (expansion factors)
                xfact = np.random.exponential(100) + 10

                data.append({
                    'fips': fips,
                    'year': year,
                    'riad_id': f"{fips}_{plot:03d}",
                    'startuse': start_code,
                    'enduse': enduse,
                    'lcc': lcc,
                    'xfact': xfact,
                    # Optional: can be merged from net returns
                    'nr.ps': np.random.normal(80, 20) if hasattr(pd, '__version__') else None
                })

    return pd.DataFrame(data)


def generate_climate_impact_data() -> Dict[str, pd.DataFrame]:
    """
    Generate climate impact data.

    Expected format:
    - GEOID or fips: County identifier
    - nr/nracre: Current net returns
    - impact: Climate-induced change in net returns
    - urnr: Urban net returns (for urban impact file)
    """
    np.random.seed(42)

    georef = generate_geographic_reference()

    # Crop climate impacts
    crop_impact = pd.DataFrame({
        'GEOID': georef['fips'].astype(str).str.zfill(5),
        'fips': georef['fips'],
        'nr': np.random.normal(250, 50, len(georef)),
        'impact': np.random.normal(-20, 15, len(georef))  # Mostly negative impacts
    })

    # Forest climate impacts
    forest_impact = pd.DataFrame({
        'GEOID': georef['fips'].astype(str).str.zfill(5),
        'fips': georef['fips'],
        'nracre': np.random.normal(40, 15, len(georef)),
        'impact': np.random.normal(5, 10, len(georef))  # Some positive impacts
    })

    # Urban climate impacts
    urban_impact = pd.DataFrame({
        'fips': georef['fips'],
        'urnr': np.random.normal(5000, 500, len(georef)),
        'impact': np.random.normal(-100, 50, len(georef))
    })

    return {
        'crop': crop_impact,
        'forest': forest_impact,
        'urban': urban_impact
    }


def generate_population_data() -> pd.DataFrame:
    """
    Generate population data for metro analysis.

    Expected format:
    - fips: County FIPS code
    - year: Year
    - population: Population count
    - metro_area or cbsa: Metropolitan area code (optional)
    """
    np.random.seed(42)

    georef = generate_geographic_reference()
    years = list(range(2000, 2021))

    data = []

    # Assign some counties to metro areas
    metro_areas = ['19100', '31080', '16980', '47900', None, None]  # Some rural

    for fips in georef['fips']:
        base_pop = np.random.lognormal(10, 1.5)  # Log-normal for realistic distribution
        growth_rate = np.random.normal(0.01, 0.02)  # 1% average growth
        metro = np.random.choice(metro_areas)

        for i, year in enumerate(years):
            population = base_pop * (1 + growth_rate) ** i
            population = max(100, int(population))  # Minimum 100 people

            data.append({
                'fips': fips,
                'year': year,
                'population': population,
                'metro_area': metro,
                'cbsa': metro  # Alternative name
            })

    return pd.DataFrame(data)


def save_test_data(output_dir: str = "test_data"):
    """
    Generate and save all test datasets.

    Creates:
    - forest_georef.csv: Geographic reference
    - nr_clean_5year_normals.csv: Net returns data
    - start_crop.csv: Crop starting transitions
    - start_pasture.csv: Pasture starting transitions
    - start_forest.csv: Forest starting transitions
    - population.csv: Population data
    - cc_impacts/: Climate impact files
    - data_format_spec.json: Format specification
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating test datasets...")

    # Generate and save geographic reference
    georef = generate_geographic_reference()
    georef.to_csv(output_path / "forest_georef.csv", index=False)
    print(f"✓ Created forest_georef.csv with {len(georef)} counties")

    # Generate and save net returns
    nr_data = generate_net_returns_data()
    nr_data.to_csv(output_path / "nr_clean_5year_normals.csv", index=False)
    print(f"✓ Created nr_clean_5year_normals.csv with {len(nr_data)} observations")

    # Generate and save land use transitions
    for start_use in ['crop', 'pasture', 'forest']:
        transitions = generate_land_use_transitions(start_use)
        transitions.to_csv(output_path / f"start_{start_use}.csv", index=False)
        print(f"✓ Created start_{start_use}.csv with {len(transitions)} transitions")

    # Generate and save population data
    pop_data = generate_population_data()
    pop_data.to_csv(output_path / "population.csv", index=False)
    print(f"✓ Created population.csv with {len(pop_data)} observations")

    # Generate and save climate impacts
    cc_dir = output_path / "cc_impacts"
    cc_dir.mkdir(exist_ok=True)

    climate_impacts = generate_climate_impact_data()
    for impact_type, impact_data in climate_impacts.items():
        impact_data.to_csv(cc_dir / f"{impact_type}_climate_change_impact.csv", index=False)
        print(f"✓ Created {impact_type}_climate_change_impact.csv")

    # Create format specification document
    format_spec = {
        "geographic_reference": {
            "file": "forest_georef.csv",
            "required_columns": {
                "fips": "integer, 5-digit county FIPS code",
                "county_fips": "alternative name for fips (optional)",
                "subregion": "string, subregion code",
                "region": "string, major region code (NO, SO, WE, MW)"
            }
        },
        "net_returns": {
            "file": "nr_clean_5year_normals.csv",
            "required_columns": {
                "fips": "integer, county FIPS code",
                "year": "integer, year of observation",
                "nr_cr": "float, crop net returns ($/acre)",
                "nr_ps": "float, pasture net returns ($/acre)",
                "nr_fr": "float, forest net returns ($/acre)",
                "nr_ur": "float, urban net returns ($/acre)"
            },
            "optional_columns": {
                "nrmean_*": "5-year mean net returns",
                "nrchange_*": "change in net returns"
            }
        },
        "land_transitions": {
            "files": ["start_crop.csv", "start_pasture.csv", "start_forest.csv"],
            "required_columns": {
                "fips": "integer, county FIPS code",
                "year": "integer, year of observation",
                "riad_id": "string, unique plot identifier",
                "startuse": "integer, starting land use (NRI BROAD: 1=crop, 3=pasture, 5=forest, 7=urban)",
                "enduse": "integer, ending land use (same coding)",
                "lcc": "integer, land capability class (1-8, higher=better quality)",
                "xfact": "float, expansion factor (survey weight)"
            }
        },
        "climate_impacts": {
            "directory": "cc_impacts/",
            "files": {
                "crop_climate_change_impact.csv": {
                    "GEOID": "string, county GEOID",
                    "nr": "float, current net returns",
                    "impact": "float, climate-induced change"
                },
                "forest_climate_change_impact.csv": {
                    "GEOID": "string, county GEOID",
                    "nracre": "float, current net returns per acre",
                    "impact": "float, climate-induced change"
                },
                "urban_climate_change_impact.csv": {
                    "fips": "integer, county FIPS",
                    "urnr": "float, urban net returns",
                    "impact": "float, climate-induced change"
                }
            }
        },
        "notes": {
            "data_types": "FIPS codes should be integers but may need zero-padding for display",
            "missing_values": "Handle NaN values before estimation",
            "weights": "xfact is used for weighted estimation",
            "time_lag": "Net returns data year + 2 = transition year (lagged relationship)"
        }
    }

    with open(output_path / "data_format_spec.json", "w") as f:
        json.dump(format_spec, f, indent=2)
    print("\n✓ Created data_format_spec.json with complete format documentation")

    # Create a sample configuration file
    config = {
        "output_dir": "results",
        "run_exploration": True,
        "run_estimation": True,
        "run_marginal_effects": True,
        "run_climate_impact": False,  # Disable for test run
        "georef_file": f"{output_dir}/forest_georef.csv",
        "nr_data_file": f"{output_dir}/nr_clean_5year_normals.csv",
        "start_crop_file": f"{output_dir}/start_crop.csv",
        "start_pasture_file": f"{output_dir}/start_pasture.csv",
        "start_forest_file": f"{output_dir}/start_forest.csv",
        "population_data": f"{output_dir}/population.csv",
        "years": [2010, 2011, 2012],
        "cc_impacts_dir": f"{output_dir}/cc_impacts"
    }

    with open(output_path / "test_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✓ Created test_config.json for running the pipeline")

    return output_path


def validate_data_format(data_dir: str) -> Tuple[bool, list]:
    """
    Validate that data files match expected format.

    Returns:
    --------
    Tuple[bool, list]
        (is_valid, list of error messages)
    """
    data_path = Path(data_dir)
    errors = []

    # Check geographic reference
    georef_file = data_path / "forest_georef.csv"
    if georef_file.exists():
        georef = pd.read_csv(georef_file)
        required_cols = ['fips', 'region']
        missing = set(required_cols) - set(georef.columns)
        if missing:
            errors.append(f"forest_georef.csv missing columns: {missing}")
    else:
        errors.append("forest_georef.csv not found")

    # Check net returns
    nr_file = data_path / "nr_clean_5year_normals.csv"
    if nr_file.exists():
        nr = pd.read_csv(nr_file)
        required_cols = ['fips', 'year', 'nr_cr', 'nr_ps', 'nr_fr', 'nr_ur']
        missing = set(required_cols) - set(nr.columns)
        if missing:
            errors.append(f"nr_clean_5year_normals.csv missing columns: {missing}")
    else:
        errors.append("nr_clean_5year_normals.csv not found")

    # Check transition files
    for start_use in ['crop', 'pasture', 'forest']:
        trans_file = data_path / f"start_{start_use}.csv"
        if trans_file.exists():
            trans = pd.read_csv(trans_file)
            required_cols = ['fips', 'year', 'enduse', 'lcc', 'xfact']
            missing = set(required_cols) - set(trans.columns)
            if missing:
                errors.append(f"start_{start_use}.csv missing columns: {missing}")
        else:
            errors.append(f"start_{start_use}.csv not found")

    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "test_data"

    # Generate test data
    test_path = save_test_data(output_dir)

    # Validate the generated data
    print("\nValidating generated data format...")
    is_valid, errors = validate_data_format(test_path)

    if is_valid:
        print("✅ All test data files are correctly formatted!")
        print(f"\nYou can now run the estimation pipeline with:")
        print(f"  uv run python -m landuse.main full {test_path}/test_config.json")
    else:
        print("❌ Validation errors found:")
        for error in errors:
            print(f"  - {error}")