"""
Rent Data Merger for 4-Category Land Use Model

Merges agricultural rents (from ag-rents repository) with urban rents
(from urban-rents repository) to create a unified net returns file.

Output columns:
    - fips: County FIPS code
    - year: Year of observation
    - nr_cr_irr: Irrigated cropland net returns ($/acre)
    - nr_cr_dry: Non-irrigated cropland net returns ($/acre)
    - nr_ps: Pasture net returns ($/acre)
    - nr_ur: Urban net returns ($/acre)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console
from rich.table import Table

console = Console()

# Default file paths (relative to project root)
DEFAULT_AG_RENTS = Path('../ag-rents/data/output/county_cash_rents_panel.csv')
DEFAULT_URBAN_RENTS = Path('../urban-rents/data/output/urban_net_returns_1975_2023_with_ak_hi.csv')


def load_ag_rents(
    filepath: Path,
    years: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Load agricultural rents data.

    Args:
        filepath: Path to county_cash_rents_panel.csv
        years: Optional (min_year, max_year) tuple to filter

    Returns:
        DataFrame with columns: fips, year, nr_cr_irr, nr_cr_dry, nr_ps
    """
    console.print(f"[blue]Loading ag rents from: {filepath}[/]")

    df = pd.read_csv(filepath, dtype={'county_fips': str})

    # Convert FIPS to integer
    df['fips'] = df['county_fips'].str.zfill(5).astype(int)

    # Rename rent columns to standard names
    df = df.rename(columns={
        'cropland_rent_irrigated': 'nr_cr_irr',
        'cropland_rent_nonirrigated': 'nr_cr_dry',
        'pasture_rent': 'nr_ps'
    })

    # Select and filter
    cols = ['fips', 'year', 'nr_cr_irr', 'nr_cr_dry', 'nr_ps']
    df = df[cols].copy()

    if years:
        df = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]

    # Report stats
    console.print(f"  Counties: {df['fips'].nunique():,}")
    console.print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    console.print(f"  Rows: {len(df):,}")

    return df


def load_urban_rents(
    filepath: Path,
    years: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Load urban rents data.

    Args:
        filepath: Path to urban_net_returns_*.csv
        years: Optional (min_year, max_year) tuple to filter

    Returns:
        DataFrame with columns: fips, year, nr_ur
    """
    console.print(f"[blue]Loading urban rents from: {filepath}[/]")

    df = pd.read_csv(filepath, dtype={'county_fips': str})

    # Handle column name variations
    if 'county_fips' in df.columns:
        df['fips'] = df['county_fips'].astype(str).str.zfill(5).astype(int)
    elif 'county_geoid' in df.columns:
        df['fips'] = df['county_geoid'].astype(str).str.zfill(5).astype(int)

    # Find the urban net return column
    ur_col = None
    for col in df.columns:
        if 'urban_net_return' in col.lower():
            ur_col = col
            break

    if ur_col is None:
        raise ValueError("Could not find urban net return column")

    df['nr_ur'] = df[ur_col]

    # Handle year column
    if 'survey_year' in df.columns:
        df['year'] = df['survey_year']

    # Select and filter
    cols = ['fips', 'year', 'nr_ur']
    df = df[cols].copy()

    if years:
        df = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]

    # Report stats
    console.print(f"  Counties: {df['fips'].nunique():,}")
    console.print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    console.print(f"  Rows: {len(df):,}")

    return df


def merge_rents(
    ag_rents: pd.DataFrame,
    urban_rents: pd.DataFrame,
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Merge agricultural and urban rents.

    Args:
        ag_rents: DataFrame with ag rents (fips, year, nr_cr_irr, nr_cr_dry, nr_ps)
        urban_rents: DataFrame with urban rents (fips, year, nr_ur)
        how: Merge type ('inner', 'outer', 'left', 'right')

    Returns:
        Merged DataFrame with all net returns columns
    """
    console.print(f"\n[blue]Merging rents ({how} join on fips, year)...[/]")

    merged = pd.merge(
        ag_rents,
        urban_rents,
        on=['fips', 'year'],
        how=how
    )

    # Report merge stats
    console.print(f"  Merged rows: {len(merged):,}")
    console.print(f"  Counties: {merged['fips'].nunique():,}")
    console.print(f"  Years: {merged['year'].min()}-{merged['year'].max()}")

    # Check for missing values
    missing = merged.isnull().sum()
    if missing.any():
        console.print("\n[yellow]Missing values:[/]")
        for col in ['nr_cr_irr', 'nr_cr_dry', 'nr_ps', 'nr_ur']:
            if col in missing and missing[col] > 0:
                console.print(f"  {col}: {missing[col]:,} ({missing[col]/len(merged)*100:.1f}%)")

    return merged


def apply_lag(df: pd.DataFrame, lag_years: int = 2) -> pd.DataFrame:
    """
    Apply year lag to net returns (for matching with transitions).

    The model uses lagged net returns: transitions in year T are influenced
    by net returns from year T-lag_years.

    Args:
        df: Net returns DataFrame with 'year' column
        lag_years: Number of years to lag (default 2)

    Returns:
        DataFrame with adjusted year column
    """
    console.print(f"\n[blue]Applying {lag_years}-year lag to net returns...[/]")

    df = df.copy()
    df['year'] = df['year'] + lag_years

    console.print(f"  Original rent years map to transition years:")
    console.print(f"  {df['year'].min() - lag_years} -> {df['year'].min()}")
    console.print(f"  {df['year'].max() - lag_years} -> {df['year'].max()}")

    return df


def create_net_returns_file(
    ag_rents_file: Path,
    urban_rents_file: Path,
    output_file: Path,
    years: Optional[Tuple[int, int]] = None,
    lag_years: int = 2,
    apply_lag_to_output: bool = True
) -> pd.DataFrame:
    """
    Create unified net returns file from ag and urban rent sources.

    Args:
        ag_rents_file: Path to ag rents CSV
        urban_rents_file: Path to urban rents CSV
        output_file: Path to save merged output
        years: Optional (min, max) years to filter input data
        lag_years: Years to lag (default 2)
        apply_lag_to_output: Whether to apply lag to output years

    Returns:
        Merged net returns DataFrame
    """
    # Load data
    ag = load_ag_rents(ag_rents_file, years)
    urban = load_urban_rents(urban_rents_file, years)

    # Merge
    merged = merge_rents(ag, urban, how='inner')

    # Apply lag if requested
    if apply_lag_to_output:
        merged = apply_lag(merged, lag_years)

    # Sort and finalize column order
    merged = merged.sort_values(['fips', 'year'])
    merged = merged[['fips', 'year', 'nr_cr_irr', 'nr_cr_dry', 'nr_ps', 'nr_ur']]

    # Save
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)

    console.print(f"\n[green]Saved: {output_file}[/]")

    # Print summary statistics
    print_summary_stats(merged)

    return merged


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for net returns."""
    console.print("\n[bold]Net Returns Summary Statistics[/]")

    table = Table(title="$/acre by land use type")
    table.add_column("Statistic", style="bold")
    table.add_column("CR_IRR", justify="right")
    table.add_column("CR_DRY", justify="right")
    table.add_column("PS", justify="right")
    table.add_column("UR", justify="right")

    cols = ['nr_cr_irr', 'nr_cr_dry', 'nr_ps', 'nr_ur']

    for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        if stat in ['mean', 'std', 'min', 'max']:
            values = [f"{df[col].agg(stat):,.0f}" for col in cols]
        else:
            pct = float(stat.replace('%', '')) / 100
            values = [f"{df[col].quantile(pct):,.0f}" for col in cols]
        table.add_row(stat, *values)

    console.print(table)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge ag and urban rent data for land use modeling"
    )
    parser.add_argument(
        '--ag-rents',
        type=Path,
        default=DEFAULT_AG_RENTS,
        help='Path to ag rents CSV'
    )
    parser.add_argument(
        '--urban-rents',
        type=Path,
        default=DEFAULT_URBAN_RENTS,
        help='Path to urban rents CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/processed/net_returns_4use.csv'),
        help='Output file path'
    )
    parser.add_argument(
        '--min-year',
        type=int,
        default=2010,
        help='Minimum year to include (default: 2010)'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        default=2015,
        help='Maximum year to include (default: 2015)'
    )
    parser.add_argument(
        '--lag',
        type=int,
        default=2,
        help='Years to lag net returns (default: 2)'
    )
    parser.add_argument(
        '--no-lag',
        action='store_true',
        help='Do not apply lag to output years'
    )

    args = parser.parse_args()

    years = (args.min_year, args.max_year)

    create_net_returns_file(
        ag_rents_file=args.ag_rents,
        urban_rents_file=args.urban_rents,
        output_file=args.output,
        years=years,
        lag_years=args.lag,
        apply_lag_to_output=not args.no_lag
    )

    return 0


if __name__ == '__main__':
    exit(main())
