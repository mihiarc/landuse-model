"""
Generate test data for CRP enrollment modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json


def generate_crp_transitions(n_counties: int = 100,
                            years: list = [2010, 2011, 2012, 2013, 2014]) -> pd.DataFrame:
    """
    Generate realistic CRP enrollment transition data.

    CRP enrollment patterns based on LCC:
    - LCC 3-5 (marginal land): Highest enrollment probability
    - LCC 1-2 (poor land): Low enrollment (may already be out of production)
    - LCC 6-8 (good land): Low enrollment (too valuable for crops)
    """
    np.random.seed(42)

    data = []

    # Generate FIPS codes
    states = [1, 5, 6, 12, 13, 17, 36, 48]
    counties = []
    for state in states:
        for county in range(1, min(20, n_counties // len(states)) + 1, 2):
            fips = state * 1000 + county
            counties.append(fips)
    counties = counties[:n_counties]

    for year in years:
        for fips in counties:
            n_plots = np.random.randint(20, 50)  # More plots for better statistics

            for plot in range(n_plots):
                # Land characteristics
                # Distribution favoring mid-range LCC (where CRP is most common)
                lcc = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],
                                      p=[0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05])

                # Starting use (CRP typically comes from cropland)
                # Weight towards cropland for CRP analysis
                startuse = np.random.choice([1, 3, 5, 12],  # crop, pasture, forest, crp
                                          p=[0.60, 0.20, 0.10, 0.10])

                # CRP enrollment logic based on LCC and current use
                if startuse == 1:  # Cropland
                    if lcc in [3, 4, 5]:  # Marginal land
                        crp_prob = 0.15 + np.random.normal(0, 0.05)
                    elif lcc in [1, 2]:  # Poor land
                        crp_prob = 0.05 + np.random.normal(0, 0.02)
                    else:  # Good land (6, 7, 8)
                        crp_prob = 0.02 + np.random.normal(0, 0.01)
                elif startuse == 3:  # Pasture
                    if lcc in [3, 4, 5]:
                        crp_prob = 0.08 + np.random.normal(0, 0.03)
                    else:
                        crp_prob = 0.02 + np.random.normal(0, 0.01)
                elif startuse == 12:  # Already in CRP
                    # CRP contracts typically last 10-15 years
                    # Small probability of exiting
                    crp_prob = 0.85  # Stays in CRP
                else:  # Forest or other
                    crp_prob = 0.01

                crp_prob = max(0, min(1, crp_prob))

                # Determine end use
                if startuse == 12:  # Already CRP
                    if np.random.random() > 0.85:  # 15% exit CRP
                        # Return to previous use (mostly crop)
                        enduse = np.random.choice([1, 3], p=[0.7, 0.3])
                    else:
                        enduse = 12  # Stay in CRP
                else:
                    if np.random.random() < crp_prob:
                        enduse = 12  # Enter CRP
                    else:
                        enduse = startuse  # Stay in current use

                # Survey weights
                xfact = np.random.exponential(100) + 10

                # CRP payment rate ($/acre) - varies by land quality
                if lcc <= 3:
                    payment_rate = np.random.normal(45, 10)
                elif lcc <= 5:
                    payment_rate = np.random.normal(65, 15)
                else:
                    payment_rate = np.random.normal(40, 10)
                payment_rate = max(20, payment_rate)

                data.append({
                    'fips': fips,
                    'year': year,
                    'riad_id': f"{fips}_{plot:03d}_{year}",
                    'startuse': startuse,
                    'enduse': enduse,
                    'lcc': lcc,
                    'xfact': xfact,
                    'in_crp': 1 if enduse == 12 else 0,
                    'crp_payment': payment_rate if enduse == 12 else 0,
                    'is_marginal': 1 if lcc in [3, 4, 5] else 0,
                    'is_cropland': 1 if startuse == 1 else 0
                })

    return pd.DataFrame(data)


def generate_crp_analysis_dataset(output_dir: str = "crp_test_data") -> Dict[str, pd.DataFrame]:
    """
    Generate complete dataset for CRP enrollment analysis.

    Returns:
    --------
    Dict containing:
    - 'transitions': CRP transition data
    - 'summary': Summary statistics
    - 'by_lcc': Enrollment rates by LCC
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating CRP enrollment test data...")

    # Generate transition data
    transitions = generate_crp_transitions()
    transitions.to_csv(output_path / "crp_transitions.csv", index=False)
    print(f"✓ Created crp_transitions.csv with {len(transitions)} observations")

    # Calculate summary statistics
    summary_stats = {
        'total_observations': len(transitions),
        'unique_counties': transitions['fips'].nunique(),
        'years': sorted(transitions['year'].unique().tolist()),
        'crp_enrollment_rate': (transitions['in_crp'].mean() * 100),
        'from_cropland_to_crp': len(transitions[(transitions['startuse'] == 1) &
                                               (transitions['enduse'] == 12)]),
        'from_pasture_to_crp': len(transitions[(transitions['startuse'] == 3) &
                                              (transitions['enduse'] == 12)]),
        'stays_in_crp': len(transitions[(transitions['startuse'] == 12) &
                                       (transitions['enduse'] == 12)]),
        'exits_crp': len(transitions[(transitions['startuse'] == 12) &
                                    (transitions['enduse'] != 12)])
    }

    # Calculate enrollment by LCC
    enrollment_by_lcc = []
    for lcc in range(1, 9):
        lcc_data = transitions[transitions['lcc'] == lcc]
        if len(lcc_data) > 0:
            enrollment_rate = (lcc_data['in_crp'].mean() * 100)
            from_crop_rate = ((lcc_data[lcc_data['startuse'] == 1]['enduse'] == 12).mean() * 100)

            enrollment_by_lcc.append({
                'lcc': lcc,
                'n_obs': len(lcc_data),
                'enrollment_rate': enrollment_rate,
                'from_crop_to_crp_rate': from_crop_rate,
                'avg_payment': lcc_data[lcc_data['in_crp'] == 1]['crp_payment'].mean()
                              if any(lcc_data['in_crp'] == 1) else 0
            })

    enrollment_df = pd.DataFrame(enrollment_by_lcc)
    enrollment_df.to_csv(output_path / "crp_enrollment_by_lcc.csv", index=False)
    print("✓ Created crp_enrollment_by_lcc.csv")

    # Create summary report
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(output_path / "crp_summary.csv", index=False)
    print("✓ Created crp_summary.csv")

    # Print summary
    print("\n" + "=" * 60)
    print("CRP ENROLLMENT TEST DATA SUMMARY")
    print("=" * 60)
    print(f"Total observations: {summary_stats['total_observations']:,}")
    print(f"Counties: {summary_stats['unique_counties']}")
    print(f"Years: {summary_stats['years'][0]}-{summary_stats['years'][-1]}")
    print(f"Overall CRP enrollment rate: {summary_stats['crp_enrollment_rate']:.1f}%")
    print(f"\nTransitions to CRP:")
    print(f"  From cropland: {summary_stats['from_cropland_to_crp']}")
    print(f"  From pasture: {summary_stats['from_pasture_to_crp']}")
    print(f"  Stays in CRP: {summary_stats['stays_in_crp']}")
    print(f"  Exits CRP: {summary_stats['exits_crp']}")

    print("\nEnrollment by Land Capability Class:")
    print("LCC | Enrollment Rate | From Crop to CRP | Avg Payment")
    print("-" * 55)
    for _, row in enrollment_df.iterrows():
        print(f" {row['lcc']}  | {row['enrollment_rate']:>13.1f}% | {row['from_crop_to_crp_rate']:>14.1f}% | ${row['avg_payment']:>9.2f}")

    # Create format specification
    format_spec = {
        "crp_transitions": {
            "description": "Land use transitions including CRP enrollment",
            "required_columns": {
                "fips": "County FIPS code",
                "year": "Year of observation",
                "startuse": "Starting land use (NRI BROAD codes)",
                "enduse": "Ending land use (12 = CRP)",
                "lcc": "Land Capability Class (1-8, higher=better)",
                "in_crp": "Binary indicator for CRP enrollment",
                "crp_payment": "CRP payment rate ($/acre) if enrolled",
                "is_marginal": "Binary indicator for marginal land (LCC 3-5)",
                "is_cropland": "Binary indicator for starting cropland"
            }
        },
        "key_patterns": {
            "optimal_lcc": "LCC 3-5 shows highest CRP enrollment",
            "source": "Most CRP comes from cropland",
            "persistence": "CRP contracts typically last 10-15 years"
        }
    }

    with open(output_path / "data_format_spec.json", "w") as f:
        json.dump(format_spec, f, indent=2)
    print("\n✓ Created data_format_spec.json")

    return {
        'transitions': transitions,
        'summary': summary_df,
        'by_lcc': enrollment_df
    }


if __name__ == "__main__":
    # Generate test data
    datasets = generate_crp_analysis_dataset()
    print("\n✅ CRP test data generation complete!")