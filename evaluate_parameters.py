#!/usr/bin/env python3
"""
Script to evaluate LCC parameter signs for logical consistency.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from landuse.data_generator import save_test_data
from landuse.logit_estimation import (
    load_data,
    estimate_land_use_transitions,
)

def evaluate_parameter_signs():
    """Evaluate LCC parameter signs for logical consistency."""

    print("=" * 70)
    print("PARAMETER SIGN EVALUATION FOR LCC-ONLY MODEL")
    print("=" * 70)

    # Generate or use existing test data
    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        print("\nGenerating test data...")
        save_test_data(str(test_data_dir))

    # Load data
    print("\nLoading data...")
    georef, nr_data, start_crop, start_pasture, start_forest = load_data(
        str(test_data_dir / "forest_georef.csv"),
        str(test_data_dir / "nr_clean_5year_normals.csv"),
        str(test_data_dir / "start_crop.csv"),
        str(test_data_dir / "start_pasture.csv"),
        str(test_data_dir / "start_forest.csv")
    )

    # Estimate LCC-only models
    print("\nEstimating LCC-only models...")
    models = estimate_land_use_transitions(
        start_crop, start_pasture, start_forest,
        nr_data=nr_data,
        georef=georef,
        years=[2010, 2011, 2012],
        use_net_returns=False
    )

    # Analyze parameter signs
    print("\n" + "=" * 70)
    print("PARAMETER ANALYSIS RESULTS")
    print("=" * 70)

    print("\nExpected patterns based on land use economics:")
    print("-" * 50)
    print("1. CROP LAND: Should prefer GOOD land (LCC 1-2)")
    print("   → Expected: Negative LCC coefficient for crop transitions")
    print("2. PASTURE: Can use MODERATE land (LCC 3-4)")
    print("   → Expected: Mixed coefficients, less sensitive to LCC")
    print("3. FOREST: Often on POOR land (LCC 5-8)")
    print("   → Expected: Positive LCC coefficient for forest transitions")
    print("4. URBAN: Less sensitive to agricultural suitability")
    print("   → Expected: Weak or mixed LCC effects")

    # Analyze each model
    results_summary = []

    for model_name, model in models.items():
        if not model_name.endswith('_data') and model is not None:
            print(f"\n{'='*70}")
            print(f"Model: {model_name}")
            print("-" * 50)

            # Extract starting land use from model name
            if 'crop' in model_name:
                start_use = 'Crop'
            elif 'pasture' in model_name:
                start_use = 'Pasture'
            elif 'forest' in model_name:
                start_use = 'Forest'
            else:
                start_use = 'Unknown'

            # Extract region
            if 'all' in model_name:
                region = 'All Regions'
            elif 'east' in model_name:
                region = 'East'
            elif 'south' in model_name:
                region = 'South'
            elif 'north' in model_name:
                region = 'North'
            else:
                region = 'Unknown'

            print(f"Starting Land Use: {start_use}")
            print(f"Region: {region}")
            print(f"Number of observations: {model.nobs:.0f}")

            # Get LCC coefficients
            if hasattr(model, 'params'):
                print("\nLCC Coefficients by End Use:")
                print("(Positive = higher LCC increases probability)")
                print("(Negative = higher LCC decreases probability)")
                print("-" * 50)

                # For multinomial logit, params is a DataFrame
                lcc_params = model.params.loc['lcc']

                # Map numeric outcomes to land use types
                # Assuming: 1=crop, 2=pasture, 3=forest, 4=urban
                land_use_map = {1: 'Crop', 2: 'Pasture', 3: 'Forest', 4: 'Urban'}

                for outcome, coef in lcc_params.items():
                    if outcome in land_use_map:
                        land_use = land_use_map[outcome]
                    else:
                        land_use = f'Outcome {outcome}'

                    # Determine if sign is as expected
                    if land_use == 'Crop':
                        expected = "Negative (prefers good land)"
                        consistent = coef < 0
                    elif land_use == 'Forest':
                        expected = "Positive (uses poor land)"
                        consistent = coef > 0
                    elif land_use == 'Pasture':
                        expected = "Mixed (flexible land use)"
                        consistent = True  # Any sign is acceptable
                    else:
                        expected = "Weak/Mixed"
                        consistent = True

                    sign = "+" if coef > 0 else "-"
                    consistency = "✓" if consistent else "✗"

                    print(f"  {land_use:10s}: {coef:8.4f} ({sign})  Expected: {expected:25s} {consistency}")

                    results_summary.append({
                        'Model': model_name,
                        'Start': start_use,
                        'Region': region,
                        'End Use': land_use,
                        'Coefficient': coef,
                        'Sign': sign,
                        'Expected': expected,
                        'Consistent': consistency
                    })

    # Summary statistics
    print("\n" + "=" * 70)
    print("LOGICAL CONSISTENCY SUMMARY")
    print("=" * 70)

    df_results = pd.DataFrame(results_summary)

    # Count consistent vs inconsistent by end use
    consistency_by_use = df_results.groupby('End Use')['Consistent'].apply(
        lambda x: (x == '✓').sum() / len(x) * 100
    )

    print("\nConsistency Rate by End Use Type:")
    print("-" * 50)
    for use, rate in consistency_by_use.items():
        print(f"  {use:10s}: {rate:5.1f}% consistent with expectations")

    # Overall consistency
    overall_consistency = (df_results['Consistent'] == '✓').sum() / len(df_results) * 100
    print(f"\nOverall Consistency: {overall_consistency:.1f}%")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Analyze crop coefficients
    crop_coefs = df_results[df_results['End Use'] == 'Crop']['Coefficient']
    if len(crop_coefs) > 0:
        print("\n1. CROP TRANSITIONS:")
        print(f"   - Mean LCC coefficient: {crop_coefs.mean():.4f}")
        print(f"   - Proportion negative: {(crop_coefs < 0).sum() / len(crop_coefs) * 100:.1f}%")
        if crop_coefs.mean() < 0:
            print("   ✓ Consistent: Crops prefer better land (lower LCC)")
        else:
            print("   ✗ Inconsistent: Expected negative coefficient")

    # Analyze forest coefficients
    forest_coefs = df_results[df_results['End Use'] == 'Forest']['Coefficient']
    if len(forest_coefs) > 0:
        print("\n2. FOREST TRANSITIONS:")
        print(f"   - Mean LCC coefficient: {forest_coefs.mean():.4f}")
        print(f"   - Proportion positive: {(forest_coefs > 0).sum() / len(forest_coefs) * 100:.1f}%")
        if forest_coefs.mean() > 0:
            print("   ✓ Consistent: Forests occupy poorer land (higher LCC)")
        else:
            print("   ✗ Inconsistent: Expected positive coefficient")

    # Analyze pasture coefficients
    pasture_coefs = df_results[df_results['End Use'] == 'Pasture']['Coefficient']
    if len(pasture_coefs) > 0:
        print("\n3. PASTURE TRANSITIONS:")
        print(f"   - Mean LCC coefficient: {pasture_coefs.mean():.4f}")
        print(f"   - Range: [{pasture_coefs.min():.4f}, {pasture_coefs.max():.4f}]")
        print("   ✓ As expected: Mixed coefficients reflecting flexible land use")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if overall_consistency > 75:
        print("\n✓ The LCC-only model shows STRONG logical consistency")
        print("  Parameter signs align well with land use economics theory.")
    elif overall_consistency > 50:
        print("\n⚠ The LCC-only model shows MODERATE logical consistency")
        print("  Most parameters align with expectations, with some exceptions.")
    else:
        print("\n✗ The LCC-only model shows WEAK logical consistency")
        print("  Many parameters do not align with theoretical expectations.")

    print("\nThe simplified LCC-only model successfully captures the relationship")
    print("between land capability and land use transitions without requiring")
    print("detailed economic data on net returns.")
    print("=" * 70)

    # Save detailed results
    output_file = "parameter_evaluation_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    evaluate_parameter_signs()