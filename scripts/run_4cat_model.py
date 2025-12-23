#!/usr/bin/env python
"""
End-to-end script for 4-Category Land Use Transition Model

This script orchestrates the full pipeline:
1. Extract NRI transitions (crop/pasture/urban with irrigation split)
2. Merge agricultural and urban rent data
3. Run multinomial logit estimation
4. Generate results and summaries

Usage:
    uv run python scripts/run_4cat_model.py

Or with options:
    uv run python scripts/run_4cat_model.py --sample 100000 --output results/test
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


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
    region_filter: str = None
):
    """
    Run the complete 4-category model pipeline.

    Parameters:
    -----------
    nri_file : Path
        Path to nri17_csv.csv
    ag_rents_file : Path
        Path to ag rents CSV
    urban_rents_file : Path
        Path to urban rents CSV
    output_dir : Path
        Output directory for all results
    sample_size : int
        Optional sample size for NRI extraction
    start_year : int
        Start year for transitions (2-digit)
    end_year : int
        End year for transitions (2-digit)
    rent_min_year : int
        Minimum year for rent data
    rent_max_year : int
        Maximum year for rent data
    use_net_returns : bool
        Include net returns in model
    skip_extraction : bool
        Skip NRI extraction if files exist
    skip_rent_merge : bool
        Skip rent merge if file exists
    """
    output_dir = Path(output_dir)
    processed_dir = output_dir / 'processed'
    results_dir = output_dir / 'results'

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold blue]4-Category Land Use Transition Model[/]\n"
        "Irrigated Crop | Non-irrigated Crop | Pasture | Urban",
        title="Pipeline"
    ))

    # Step 1: Extract NRI transitions
    console.print("\n[bold]Step 1: Extract NRI Transitions[/]")
    console.print("-" * 50)

    transition_files = {
        'cr_irr': processed_dir / 'start_cr_irr.csv',
        'cr_dry': processed_dir / 'start_cr_dry.csv',
        'ps': processed_dir / 'start_ps.csv',
        'ur': processed_dir / 'start_ur.csv',
    }
    georef_file = processed_dir / 'georef.csv'

    all_exist = all(f.exists() for f in transition_files.values()) and georef_file.exists()

    if skip_extraction and all_exist:
        console.print("[yellow]Skipping extraction - files already exist[/]")
    else:
        from landuse.nri_extractor import extract_transitions
        extract_transitions(
            nri_file=nri_file,
            output_dir=processed_dir,
            start_year=start_year,
            end_year=end_year,
            sample_size=sample_size
        )

    # Step 2: Merge rent data
    console.print("\n[bold]Step 2: Merge Rent Data[/]")
    console.print("-" * 50)

    nr_file = processed_dir / 'net_returns_4use.csv'

    if skip_rent_merge and nr_file.exists():
        console.print("[yellow]Skipping rent merge - file already exists[/]")
    else:
        from landuse.rent_merger import create_net_returns_file
        create_net_returns_file(
            ag_rents_file=ag_rents_file,
            urban_rents_file=urban_rents_file,
            output_file=nr_file,
            years=(rent_min_year, rent_max_year),
            lag_years=2,
            apply_lag_to_output=True
        )

    # Step 3: Run logit estimation
    console.print("\n[bold]Step 3: Estimate Multinomial Logit Models[/]")
    console.print("-" * 50)

    from landuse.logit_estimation import main_4cat

    # Determine transition years based on rent data + lag
    transition_years = list(range(rent_min_year + 2, rent_max_year + 3))
    console.print(f"Transition years: {transition_years}")

    # Parse subregions filter
    subregions = [region_filter] if region_filter else None

    main_4cat(
        georef_file=str(georef_file),
        nr_data_file=str(nr_file),
        start_cr_irr_file=str(transition_files['cr_irr']),
        start_cr_dry_file=str(transition_files['cr_dry']),
        start_pasture_file=str(transition_files['ps']),
        start_urban_file=str(transition_files['ur']),
        output_dir=str(results_dir),
        use_net_returns=use_net_returns,
        years=transition_years,
        subregions=subregions
    )

    # Summary
    console.print("\n")
    console.print(Panel.fit(
        f"[green]Pipeline Complete![/]\n\n"
        f"Processed data: {processed_dir}\n"
        f"Model results: {results_dir}",
        title="Summary"
    ))


def main():
    """Command-line interface."""
    # Default paths
    project_root = Path(__file__).parent.parent
    default_nri = project_root / 'data' / 'nri17_csv.csv'
    default_ag_rents = project_root.parent / 'ag-rents' / 'data' / 'output' / 'county_cash_rents_panel.csv'
    default_urban_rents = project_root.parent / 'urban-rents' / 'data' / 'output' / 'urban_net_returns_1975_2023_with_ak_hi.csv'
    default_output = project_root / 'output' / '4cat_model'

    parser = argparse.ArgumentParser(
        description="Run 4-category land use transition model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample of 100k NRI rows (for testing)
  uv run python scripts/run_4cat_model.py --sample 100000

  # Run full model
  uv run python scripts/run_4cat_model.py

  # Skip extraction if already done
  uv run python scripts/run_4cat_model.py --skip-extraction
        """
    )

    parser.add_argument(
        '--nri',
        type=Path,
        default=default_nri,
        help=f'Path to NRI data (default: {default_nri})'
    )
    parser.add_argument(
        '--ag-rents',
        type=Path,
        default=default_ag_rents,
        help='Path to ag rents CSV'
    )
    parser.add_argument(
        '--urban-rents',
        type=Path,
        default=default_urban_rents,
        help='Path to urban rents CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=default_output,
        help='Output directory'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size for NRI extraction (default: all rows)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=12,
        help='Start transition year (2-digit, default: 12)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=17,
        help='End transition year (2-digit, default: 17)'
    )
    parser.add_argument(
        '--rent-min-year',
        type=int,
        default=2010,
        help='Minimum rent data year (default: 2010)'
    )
    parser.add_argument(
        '--rent-max-year',
        type=int,
        default=2015,
        help='Maximum rent data year (default: 2015)'
    )
    parser.add_argument(
        '--lcc-only',
        action='store_true',
        help='Use LCC-only model (no net returns)'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=None,
        help='Filter to specific subregion (e.g., SE, CB, DL). Default: all regions'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip NRI extraction if output files exist'
    )
    parser.add_argument(
        '--skip-rent-merge',
        action='store_true',
        help='Skip rent merge if output file exists'
    )

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

    # Run pipeline
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
        region_filter=args.region
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
