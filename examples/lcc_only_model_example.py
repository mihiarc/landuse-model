#!/usr/bin/env python3
"""
Example script demonstrating the simplified LCC-only land use model.

This script shows how to estimate land use transition probabilities
using only Land Capability Class (LCC) as a predictor, without net returns.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from landuse.data_generator import save_test_data
from landuse.logit_estimation import (
    load_data,
    estimate_land_use_transitions,
    save_estimation_results
)


def run_lcc_only_example():
    """Run example of LCC-only land use model."""

    print("=" * 60)
    print("Land Use Modeling - LCC-Only Model Example")
    print("=" * 60)

    # Generate test data if it doesn't exist
    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        print("\n1. Generating test data...")
        save_test_data(str(test_data_dir))
    else:
        print("\n1. Using existing test data...")

    # Load the data
    print("\n2. Loading data files...")
    georef, nr_data, start_crop, start_pasture, start_forest = load_data(
        str(test_data_dir / "forest_georef.csv"),
        str(test_data_dir / "nr_clean_5year_normals.csv"),  # Still loaded but not used in LCC-only model
        str(test_data_dir / "start_crop.csv"),
        str(test_data_dir / "start_pasture.csv"),
        str(test_data_dir / "start_forest.csv")
    )

    print(f"   - Loaded {len(georef)} counties")
    print(f"   - Crop transitions: {len(start_crop)} observations")
    print(f"   - Pasture transitions: {len(start_pasture)} observations")
    print(f"   - Forest transitions: {len(start_forest)} observations")

    # Estimate LCC-only models
    print("\n3. Estimating LCC-only models...")
    print("   This simplified model uses only Land Capability Class (LCC)")
    print("   to predict land use transitions, without economic factors.")
    print("")

    lcc_models = estimate_land_use_transitions(
        start_crop,
        start_pasture,
        start_forest,
        nr_data=nr_data,  # Passed but not used when use_net_returns=False
        georef=georef,
        years=[2010, 2011, 2012],
        use_net_returns=False  # This ensures we use LCC-only model
    )

    # Save results
    output_dir = Path("results_lcc_only")
    output_dir.mkdir(exist_ok=True)

    print("\n4. Saving results...")
    save_estimation_results(lcc_models, str(output_dir))

    # Display model summary
    print("\n5. Model Results Summary")
    print("-" * 40)

    for name, model in lcc_models.items():
        if not name.endswith('_data') and model is not None:
            print(f"\nModel: {name}")
            print(f"  Number of observations: {model.nobs:.0f}")
            print(f"  Log-likelihood: {model.llf:.2f}")
            print(f"  AIC: {model.aic:.2f}")
            print(f"  BIC: {model.bic:.2f}")

            # Show LCC coefficient
            if hasattr(model, 'params'):
                # For multinomial logit, params is a DataFrame with columns for each outcome
                print(f"  Number of parameters: {model.params.size}")
                # Show first few parameters for inspection
                if len(model.params) > 0:
                    print(f"  Sample parameters (first 3):")
                    for i, (idx, row) in enumerate(model.params.iterrows()):
                        if i >= 3:
                            break
                        if 'lcc' in idx.lower():
                            print(f"    {idx}: {row.values}")

    print("\n" + "=" * 60)
    print("LCC-Only Model Example Complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("\nInterpretation:")
    print("- The model predicts land use transitions based solely on land quality (LCC)")
    print("- LCC ranges from 1 (best) to 8 (worst) for agricultural use")
    print("- Lower LCC values typically favor agricultural uses (crop/pasture)")
    print("- Higher LCC values typically favor non-agricultural uses (forest)")
    print("=" * 60)


def compare_models():
    """Compare LCC-only model with full model including net returns."""

    print("\n" + "=" * 60)
    print("Model Comparison: LCC-Only vs Full Model")
    print("=" * 60)

    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        print("Generating test data first...")
        save_test_data(str(test_data_dir))

    # Load data
    georef, nr_data, start_crop, start_pasture, start_forest = load_data(
        str(test_data_dir / "forest_georef.csv"),
        str(test_data_dir / "nr_clean_5year_normals.csv"),
        str(test_data_dir / "start_crop.csv"),
        str(test_data_dir / "start_pasture.csv"),
        str(test_data_dir / "start_forest.csv")
    )

    # Estimate LCC-only model
    print("\n1. Estimating LCC-only model...")
    lcc_models = estimate_land_use_transitions(
        start_crop, start_pasture, start_forest,
        nr_data=nr_data,
        georef=georef,
        years=[2010, 2011, 2012],
        use_net_returns=False
    )

    # Estimate full model with net returns
    print("\n2. Estimating full model with net returns...")
    full_models = estimate_land_use_transitions(
        start_crop, start_pasture, start_forest,
        nr_data=nr_data,
        georef=georef,
        years=[2010, 2011, 2012],
        use_net_returns=True
    )

    # Compare model fit statistics
    print("\n3. Model Comparison Results")
    print("-" * 40)

    comparison_data = []

    for key in lcc_models.keys():
        if not key.endswith('_data'):
            lcc_model = lcc_models.get(key)
            full_model = full_models.get(key)

            if lcc_model and full_model:
                comparison_data.append({
                    'Model': key,
                    'LCC_AIC': lcc_model.aic,
                    'Full_AIC': full_model.aic,
                    'LCC_BIC': lcc_model.bic,
                    'Full_BIC': full_model.bic,
                    'LCC_LogLik': lcc_model.llf,
                    'Full_LogLik': full_model.llf
                })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        print("\nAIC Comparison (lower is better):")
        for _, row in comparison_df.iterrows():
            print(f"  {row['Model'][:20]:20s}: LCC={row['LCC_AIC']:.1f}, Full={row['Full_AIC']:.1f}")

        print("\nBIC Comparison (lower is better):")
        for _, row in comparison_df.iterrows():
            print(f"  {row['Model'][:20]:20s}: LCC={row['LCC_BIC']:.1f}, Full={row['Full_BIC']:.1f}")

        print("\nSummary:")
        print("- LCC-only model is simpler with fewer parameters")
        print("- Full model may have better fit but higher complexity")
        print("- Choose based on data availability and prediction needs")


if __name__ == "__main__":
    # Run the LCC-only example
    run_lcc_only_example()

    # Optional: Compare with full model
    print("\n" + "=" * 60)
    response = input("Would you like to compare LCC-only with full model? (y/n): ")
    if response.lower() == 'y':
        compare_models()