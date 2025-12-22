#!/usr/bin/env python3
"""
Test forest land use modeling with correct NRI BROAD codes.
Forest = Code 5 (not 3 or 7!)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from landuse.data_generator import save_test_data, generate_land_use_transitions
from landuse.logit_estimation import (
    load_data,
    estimate_land_use_transitions,
    prepare_estimation_data
)
from landuse.nri_codes import NRI_BROAD_CODES, MODELING_CODES

def test_forest_transitions():
    """Test forest land use transitions with correct NRI BROAD codes."""

    print("=" * 80)
    print("FOREST LAND USE MODELING TEST")
    print("Using NRI BROAD Codes:")
    for code, name in MODELING_CODES.items():
        print(f"  Code {code}: {name}")
    print("=" * 80)

    # Generate fresh test data with correct codes
    test_data_dir = Path("test_data_forest")
    if test_data_dir.exists():
        import shutil
        shutil.rmtree(test_data_dir)

    print("\n1. Generating test data with correct NRI BROAD codes...")
    save_test_data(str(test_data_dir))

    # Load the data
    print("\n2. Loading generated data...")
    georef, nr_data, start_crop, start_pasture, start_forest = load_data(
        str(test_data_dir / "forest_georef.csv"),
        str(test_data_dir / "nr_clean_5year_normals.csv"),
        str(test_data_dir / "start_crop.csv"),
        str(test_data_dir / "start_pasture.csv"),
        str(test_data_dir / "start_forest.csv")
    )

    # Check the land use codes in the data
    print("\n3. Verifying land use codes in generated data...")

    for name, df in [("Crop", start_crop), ("Pasture", start_pasture), ("Forest", start_forest)]:
        print(f"\n{name} starting data:")
        print(f"  Unique startuse values: {sorted(df['startuse'].unique())}")
        print(f"  Unique enduse values: {sorted(df['enduse'].unique())}")

        # Verify forest is code 5
        if name == "Forest":
            assert 5 in df['startuse'].unique(), "Forest should be code 5!"
            print(f"  ✓ Confirmed: Forest startuse = 5")

        # Check transition patterns
        transitions = df.groupby(['startuse', 'enduse']).size().reset_index(name='count')
        transitions['start_name'] = transitions['startuse'].map(MODELING_CODES)
        transitions['end_name'] = transitions['enduse'].map(MODELING_CODES)

        print(f"\n  Transition summary for {name}:")
        for _, row in transitions.iterrows():
            pct = row['count'] / len(df) * 100
            print(f"    {row['start_name']:8s} → {row['end_name']:8s}: {row['count']:4d} ({pct:5.1f}%)")

    # Estimate models with LCC-only
    print("\n4. Estimating LCC-only models with forest transitions...")
    models = estimate_land_use_transitions(
        start_crop, start_pasture, start_forest,
        nr_data=nr_data,
        georef=georef,
        years=[2010, 2011, 2012],
        use_net_returns=False
    )

    # Analyze forest-related transitions
    print("\n5. Analyzing forest transition parameters...")
    print("-" * 60)

    forest_results = []

    for model_name, model in models.items():
        if not model_name.endswith('_data') and model is not None:
            # Look for forest-related parameters
            if 'forest' in model_name.lower():
                print(f"\nModel: {model_name}")
                print(f"  Observations: {model.nobs:.0f}")

                # Get LCC coefficients
                if hasattr(model, 'params'):
                    lcc_params = model.params.loc['lcc']

                    print("  LCC coefficients by end use:")
                    for outcome, coef in lcc_params.items():
                        end_name = MODELING_CODES.get(outcome, f"Code {outcome}")

                        # Determine expected sign for forest transitions
                        if outcome == 5:  # Forest
                            expected = "Negative (forest on poorer/lower LCC land)"
                            consistent = coef < 0
                        elif outcome in [1, 3]:  # Crop or Pasture
                            expected = "Positive (ag on better/higher LCC land)"
                            consistent = coef > 0
                        else:
                            expected = "Variable"
                            consistent = True

                        sign = "+" if coef > 0 else "-"
                        check = "✓" if consistent else "✗"

                        print(f"    → {end_name:8s}: {coef:7.4f} ({sign}) Expected: {expected} {check}")

                        forest_results.append({
                            'model': model_name,
                            'end_use': end_name,
                            'coefficient': coef,
                            'consistent': consistent
                        })

    # Summary
    print("\n" + "=" * 80)
    print("FOREST MODELING SUMMARY")
    print("=" * 80)

    if forest_results:
        df_results = pd.DataFrame(forest_results)

        # Check forest persistence
        forest_to_forest = df_results[
            (df_results['model'].str.contains('forest')) &
            (df_results['end_use'] == 'Forest')
        ]

        if not forest_to_forest.empty:
            mean_coef = forest_to_forest['coefficient'].mean()
            print(f"\nForest persistence (forest→forest) LCC coefficient: {mean_coef:.4f}")
            if mean_coef < 0:
                print("✓ CORRECT: Negative coefficient indicates forest persists on poorer land (lower LCC)")
            else:
                print("✗ UNEXPECTED: Positive coefficient (may be due to synthetic data)")

        # Check forest to agriculture conversions
        forest_to_ag = df_results[
            (df_results['model'].str.contains('forest')) &
            (df_results['end_use'].isin(['Crop', 'Pasture']))
        ]

        if not forest_to_ag.empty:
            mean_coef = forest_to_ag['coefficient'].mean()
            print(f"\nForest→Agriculture conversion LCC coefficient: {mean_coef:.4f}")
            if mean_coef > 0:
                print("✓ CORRECT: Positive coefficient indicates conversion to ag on better land (higher LCC)")
            else:
                print("✗ UNEXPECTED: Negative coefficient (may be due to synthetic data)")

        # Overall consistency
        consistency_rate = df_results['consistent'].mean() * 100
        print(f"\nOverall consistency with theory: {consistency_rate:.1f}%")

        if consistency_rate > 75:
            print("✓ Strong consistency with land use theory")
        elif consistency_rate > 50:
            print("⚠ Moderate consistency with land use theory")
        else:
            print("✗ Weak consistency (likely due to synthetic data limitations)")

    print("\nKey Insights:")
    print("- Forest is correctly coded as 5 (NRI BROAD code)")
    print("- Higher LCC = better quality land for agriculture")
    print("- Forest should prefer lower LCC land (negative coefficients)")
    print("- Agriculture should prefer higher LCC land (positive coefficients)")
    print("=" * 80)

if __name__ == "__main__":
    test_forest_transitions()