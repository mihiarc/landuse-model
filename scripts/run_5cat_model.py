#!/usr/bin/env python
"""
End-to-end script for 5-Category Land Use Transition Model

This script orchestrates the full pipeline for the 5-category model which
combines irrigated and non-irrigated cropland into a single "Cropland" category.

Categories:
    CR (1): Cropland (combined irrigated + non-irrigated)
    PS (2): Pasture
    RG (3): Rangeland
    FR (4): Forest
    UR (5): Urban

Usage:
    uv run python scripts/run_5cat_model.py
    uv run python scripts/run_5cat_model.py --sample 100000 --output output/test_5cat
    uv run python scripts/run_5cat_model.py --region SE
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# RPA Subregion names
RPA_SUBREGIONS = {
    'NE': 'Northeast',
    'LS': 'Lake States',
    'CB': 'Corn Belt',
    'NP': 'Northern Plains',
    'AP': 'Appalachian',
    'SE': 'Southeast',
    'DL': 'Delta',
    'SP': 'Southern Plains',
    'MT': 'Mountain',
    'PC': 'Pacific Coast',
}

# 5-category region specs (combined cropland)
# Based on coefficient sign analysis - much fewer violations than 6-category
REGION_SPECS_5CAT = {
    'SE': {'crop': 'lcc + nr_ur', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},
    'DL': {'crop': 'lcc + nr_ur', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},
    'NP': {'crop': 'lcc + nr_ur', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},
    'CB': {'crop': 'lcc + nr_ur', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},
    'AP': {'crop': 'lcc + nr_ur', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},
    'LS': {'crop': 'lcc + nr_ur', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},
    'NE': {'crop': 'lcc + nr_ur', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},
    'MT': {'crop': 'lcc', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},  # crop has violation
    'PC': {'crop': 'lcc + nr_ur', 'pasture': 'lcc', 'range': 'lcc', 'forest': 'lcc'},  # pasture ~0 coef
    'SP': {'crop': 'lcc + nr_ur', 'pasture': 'lcc + nr_ur', 'range': 'lcc', 'forest': 'lcc'},
}


def create_combined_crop_returns(nr_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined cropland net returns from irrigated and non-irrigated.

    Uses area-weighted average or simple average of irrigated and non-irrigated
    cropland returns to get a single cropland return value.
    """
    nr = nr_data.copy()

    # Check if we have separate irrigated/non-irrigated columns
    if 'nr_cr_irr' in nr.columns and 'nr_cr_dry' in nr.columns:
        # Use average of irrigated and non-irrigated (could use weighted avg if we had acreage)
        nr['nr_cr'] = nr[['nr_cr_irr', 'nr_cr_dry']].mean(axis=1)
    elif 'nr_cr' not in nr.columns:
        # If no crop returns at all, use a default
        console.print("[yellow]Warning: No cropland returns found, using default value[/]")
        nr['nr_cr'] = 0.0

    return nr


def run_pipeline(
    nri_file: Path,
    ag_rents_file: Path,
    urban_rents_file: Path,
    output_dir: Path,
    sample_size: int = None,
    start_year: int = 12,
    end_year: int = 17,
    rent_min_year: int = 2010,
    rent_max_year: int = 2015,
    use_net_returns: bool = True,
    skip_extraction: bool = False,
    skip_rent_merge: bool = False,
    region_filter: str = None,
    check_signs_only: bool = False
):
    """
    Run the complete 5-category model pipeline.
    """
    output_dir = Path(output_dir)
    processed_dir = output_dir / 'processed'
    results_dir = output_dir / 'results'

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold blue]5-Category Land Use Transition Model[/]\n"
        "CR (combined) | Pasture | Range | Forest | Urban\n\n"
        "[dim]Crop/Pasture: LCC + Net Returns | Range/Forest: LCC only[/]",
        title="Pipeline"
    ))

    # Step 1: Extract NRI transitions (5 categories)
    console.print("\n[bold]Step 1: Extract NRI Transitions (5 categories)[/]")
    console.print("-" * 50)

    transition_files = {
        'crop': processed_dir / 'start_crop.csv',
        'ps': processed_dir / 'start_ps.csv',
        'rg': processed_dir / 'start_rg.csv',
        'fr': processed_dir / 'start_fr.csv',
        'ur': processed_dir / 'start_ur.csv',
    }
    georef_file = processed_dir / 'georef.csv'

    all_exist = all(f.exists() for f in transition_files.values()) and georef_file.exists()

    if skip_extraction and all_exist:
        console.print("[yellow]Skipping extraction - files already exist[/]")
    else:
        from landuse.nri_extractor import extract_transitions_5cat
        extract_transitions_5cat(
            nri_file=nri_file,
            output_dir=processed_dir,
            start_year=start_year,
            end_year=end_year,
            sample_size=sample_size
        )

    # Step 2: Merge rent data
    console.print("\n[bold]Step 2: Merge Rent Data[/]")
    console.print("-" * 50)

    nr_file = processed_dir / 'net_returns_5use.csv'

    if skip_rent_merge and nr_file.exists():
        console.print("[yellow]Skipping rent merge - file already exists[/]")
    else:
        from landuse.rent_merger import create_net_returns_file
        # First create the standard 4-column file
        nr_temp_file = processed_dir / 'net_returns_4use.csv'
        create_net_returns_file(
            ag_rents_file=ag_rents_file,
            urban_rents_file=urban_rents_file,
            output_file=nr_temp_file,
            years=(rent_min_year, rent_max_year),
            lag_years=2,
            apply_lag_to_output=True
        )

        # Now create combined cropland returns
        nr_data = pd.read_csv(nr_temp_file)
        nr_data = create_combined_crop_returns(nr_data)
        nr_data.to_csv(nr_file, index=False)
        console.print(f"Created combined crop returns: {nr_file}")

    # Step 3: Run estimation
    console.print("\n[bold]Step 3: Estimate Multinomial Logit Models (4 starting uses)[/]")
    console.print("-" * 50)

    # Load data
    georef = pd.read_csv(georef_file)
    nr_data = pd.read_csv(nr_file)

    transitions = {}
    for name, filepath in transition_files.items():
        if filepath.exists():
            transitions[name] = pd.read_csv(filepath)
        else:
            transitions[name] = pd.DataFrame()

    # Determine years
    transition_years = list(range(rent_min_year + 2, rent_max_year + 3))
    console.print(f"Transition years: {transition_years}")

    # Parse subregions filter
    if region_filter:
        subregions = [r.strip() for r in region_filter.split(',')]
    else:
        subregions = sorted(georef['subregion'].unique())

    console.print(f"Subregions: {subregions}")

    # Run estimation for each region
    results_summary = []

    for subregion in subregions:
        subregion_name = RPA_SUBREGIONS.get(subregion, subregion)
        console.print(f"\n[bold cyan]{subregion} - {subregion_name}[/]")
        console.print("=" * 50)

        region_fips = georef[georef['subregion'] == subregion]['fips'].unique()
        console.print(f"Counties: {len(region_fips)}")

        region_specs = REGION_SPECS_5CAT.get(subregion, {})

        for start_name in ['crop', 'pasture', 'range', 'forest']:
            # Map display names to file keys
            trans_key = {'crop': 'crop', 'pasture': 'ps', 'range': 'rg', 'forest': 'fr'}.get(start_name, start_name)
            trans_df = transitions.get(trans_key, pd.DataFrame())
            if len(trans_df) == 0:
                continue

            # Filter to region
            est_data = trans_df[trans_df['fips'].isin(region_fips)].copy()
            if len(est_data) == 0:
                console.print(f"  {start_name}: No data")
                continue

            # Filter to years
            est_data = est_data[est_data['year'].isin(transition_years)].copy()

            # Merge net returns
            est_data = est_data.merge(nr_data, on=['fips', 'year'], how='inner')

            if len(est_data) < 100:
                console.print(f"  {start_name}: Only {len(est_data)} obs (min 100 required)")
                continue

            # Scale net returns
            for col in ['nr_cr', 'nr_ps', 'nr_ur']:
                if col in est_data.columns:
                    if col == 'nr_ur':
                        est_data[col] = est_data[col] / 1000
                    else:
                        est_data[col] = est_data[col] / 100

            # Check for urban transitions
            unique_ends = sorted(est_data['enduse'].unique())
            urban_code = 5  # Urban is code 5 in 5-category
            if urban_code not in unique_ends:
                console.print(f"  {start_name}: n={len(est_data):,}, No urban transitions")
                continue

            urban_count = (est_data['enduse'] == urban_code).sum()

            # Determine specification
            spec = region_specs.get(start_name, 'lcc')
            include_nr = 'nr_ur' in spec

            try:
                y = est_data['enduse']

                if include_nr and 'nr_ur' in est_data.columns:
                    X = sm.add_constant(est_data[['lcc', 'nr_ur']])
                    formula_desc = 'LCC + nr_ur'
                else:
                    X = sm.add_constant(est_data[['lcc']])
                    formula_desc = 'LCC'

                model = MNLogit(y, X)
                result = model.fit(method='bfgs', disp=False, maxiter=500)

                # Get coefficient for urban
                params = result.params
                sorted_ends = sorted(unique_ends)
                base_cat = sorted_ends[0]
                non_base = [e for e in sorted_ends if e != base_cat]
                col_to_enduse = {i: non_base[i] for i in range(len(non_base))}

                urban_col = None
                for col, enduse in col_to_enduse.items():
                    if enduse == urban_code:
                        urban_col = col
                        break

                if include_nr and urban_col is not None:
                    nr_ur_coef = params.loc['nr_ur', urban_col]
                    sign_ok = nr_ur_coef > 0
                    sign_str = '[green]✓[/]' if sign_ok else '[red]✗[/]'
                    console.print(f"  {start_name}: n={len(est_data):,}, urban={urban_count}, "
                                  f"LL={result.llf:.1f}, nr_ur→UR: {nr_ur_coef:+.3f} {sign_str}")

                    results_summary.append({
                        'region': subregion,
                        'start': start_name,
                        'n': len(est_data),
                        'urban_trans': urban_count,
                        'formula': formula_desc,
                        'nr_ur_coef': nr_ur_coef,
                        'valid': sign_ok,
                        'llf': result.llf
                    })
                else:
                    console.print(f"  {start_name}: n={len(est_data):,}, urban={urban_count}, "
                                  f"LL={result.llf:.1f} ({formula_desc})")
                    results_summary.append({
                        'region': subregion,
                        'start': start_name,
                        'n': len(est_data),
                        'urban_trans': urban_count,
                        'formula': formula_desc,
                        'nr_ur_coef': None,
                        'valid': True,
                        'llf': result.llf
                    })

                # Save model results
                if not check_signs_only:
                    model_file = results_dir / f'{start_name}_{subregion}_params.csv'
                    params.to_csv(model_file)

            except Exception as e:
                console.print(f"  {start_name}: [red]Error[/] - {str(e)[:60]}")

    # Summary
    console.print("\n")

    # Check for violations
    violations = [r for r in results_summary if r.get('nr_ur_coef') is not None and not r['valid']]

    if violations:
        console.print(Panel.fit(
            f"[yellow]Found {len(violations)} coefficient sign violation(s)[/]\n\n" +
            "\n".join([f"  {v['region']}/{v['start']}: nr_ur→UR = {v['nr_ur_coef']:+.3f}"
                       for v in violations]),
            title="Sign Violations"
        ))
    else:
        console.print(Panel.fit(
            f"[green]All {len([r for r in results_summary if r.get('nr_ur_coef')])} models with nr_ur have correct signs![/]",
            title="Sign Check Passed"
        ))

    # Save summary
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_file = results_dir / 'model_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        console.print(f"\nSummary saved to: {summary_file}")

    console.print(Panel.fit(
        f"[green]Pipeline Complete![/]\n\n"
        f"Processed data: {processed_dir}\n"
        f"Model results: {results_dir}",
        title="Summary"
    ))


def main():
    """Command-line interface."""
    project_root = Path(__file__).parent.parent
    default_nri = project_root / 'data' / 'nri17_csv.csv'
    default_ag_rents = project_root.parent / 'ag-rents' / 'data' / 'output' / 'county_cash_rents_panel.csv'
    default_urban_rents = project_root.parent / 'urban-rents' / 'data' / 'output' / 'urban_net_returns_1975_2023_with_ak_hi.csv'
    default_output = project_root / 'output' / '5cat_model'

    parser = argparse.ArgumentParser(
        description="Run 5-category land use transition model (combined cropland)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample of 100k NRI rows (for testing)
  uv run python scripts/run_5cat_model.py --sample 100000

  # Run full model
  uv run python scripts/run_5cat_model.py

  # Run for Southeast only
  uv run python scripts/run_5cat_model.py --region SE

  # Skip extraction if already done
  uv run python scripts/run_5cat_model.py --skip-extraction

Categories:
  CR  - Cropland (combined irrigated + non-irrigated, LCC + net returns)
  PS  - Pasture (LCC + net returns)
  RG  - Rangeland (LCC only)
  FR  - Forest (LCC only)
  UR  - Urban (LCC + net returns)
        """
    )

    parser.add_argument('--nri', type=Path, default=default_nri, help='Path to NRI data')
    parser.add_argument('--ag-rents', type=Path, default=default_ag_rents, help='Path to ag rents CSV')
    parser.add_argument('--urban-rents', type=Path, default=default_urban_rents, help='Path to urban rents CSV')
    parser.add_argument('--output', '-o', type=Path, default=default_output, help='Output directory')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for NRI extraction')
    parser.add_argument('--start-year', type=int, default=12, help='Start transition year (2-digit)')
    parser.add_argument('--end-year', type=int, default=17, help='End transition year (2-digit)')
    parser.add_argument('--rent-min-year', type=int, default=2010, help='Minimum rent data year')
    parser.add_argument('--rent-max-year', type=int, default=2015, help='Maximum rent data year')
    parser.add_argument('--lcc-only', action='store_true', help='Use LCC-only model for all uses')
    parser.add_argument('--region', type=str, default=None, help='Filter to specific subregion(s)')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip NRI extraction')
    parser.add_argument('--skip-rent-merge', action='store_true', help='Skip rent merge')
    parser.add_argument('--check-signs-only', action='store_true', help='Only check coefficient signs')

    args = parser.parse_args()

    # Validate paths
    if not args.nri.exists():
        console.print(f"[red]Error: NRI file not found: {args.nri}[/]")
        return 1

    if not args.ag_rents.exists():
        console.print(f"[red]Error: Ag rents file not found: {args.ag_rents}[/]")
        return 1

    if not args.urban_rents.exists():
        console.print(f"[red]Error: Urban rents file not found: {args.urban_rents}[/]")
        return 1

    run_pipeline(
        nri_file=args.nri,
        ag_rents_file=args.ag_rents,
        urban_rents_file=args.urban_rents,
        output_dir=args.output,
        sample_size=args.sample,
        start_year=args.start_year,
        end_year=args.end_year,
        rent_min_year=args.rent_min_year,
        rent_max_year=args.rent_max_year,
        use_net_returns=not args.lcc_only,
        skip_extraction=args.skip_extraction,
        skip_rent_merge=args.skip_rent_merge,
        region_filter=args.region,
        check_signs_only=args.check_signs_only
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
