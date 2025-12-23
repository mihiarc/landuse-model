"""
NRI Data Extractor for 4-Category Land Use Model

Extracts land use transitions from the raw NRI 2017 dataset,
splitting cropland into irrigated and non-irrigated categories.

Land Use Categories:
    1 = Irrigated Cropland (CR_IRR)
    2 = Non-irrigated Cropland (CR_DRY)
    3 = Pasture (PS)
    4 = Urban (UR)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

# NRI BROAD codes we're interested in
NRI_BROAD_CROP = 1
NRI_BROAD_PASTURE = 3
NRI_BROAD_RANGE = 4
NRI_BROAD_FOREST = 5
NRI_BROAD_URBAN = 7

# Our 4-category codes (crop/pasture/urban with irrigation split)
LANDUSE_4CAT = {
    'CR_IRR': 1,  # Irrigated cropland
    'CR_DRY': 2,  # Non-irrigated cropland
    'PS': 3,      # Pasture
    'UR': 4,      # Urban
}

# Our 5-category codes (primary model - combined cropland)
LANDUSE_5CAT = {
    'CR': 1,      # Cropland (irrigated + non-irrigated combined)
    'PS': 2,      # Pasture
    'RG': 3,      # Rangeland
    'FR': 4,      # Forest
    'UR': 5,      # Urban
}

# Reverse mapping for display
LANDUSE_NAMES_4CAT = {v: k for k, v in LANDUSE_4CAT.items()}
LANDUSE_NAMES_5CAT = {v: k for k, v in LANDUSE_5CAT.items()}
LANDUSE_NAMES = LANDUSE_NAMES_5CAT  # Default to 5-cat (primary model)

# =============================================================================
# RPA Subregions - More granular regional breakdown
# Based on agricultural production zones and ecological characteristics
# Reference: USDA ERS Farm Resource Regions, RPA Assessment subregions
# =============================================================================

# RPA Subregion codes
RPA_SUBREGIONS = {
    'NE': 'Northeast',           # New England + Mid-Atlantic
    'LS': 'Lake States',         # Great Lakes region
    'CB': 'Corn Belt',           # Core Corn Belt
    'NP': 'Northern Plains',     # Northern Great Plains
    'AP': 'Appalachian',         # Appalachian region
    'SE': 'Southeast',           # Coastal Southeast
    'DL': 'Delta',               # Mississippi Delta
    'SP': 'Southern Plains',     # Southern Great Plains (TX, OK)
    'MT': 'Mountain',            # Rocky Mountain states
    'PC': 'Pacific Coast',       # West Coast
}

# State FIPS to Subregion mapping
STATE_TO_SUBREGION = {
    # ----- NORTHEAST (NE) -----
    '09': 'NE',  # Connecticut
    '23': 'NE',  # Maine
    '25': 'NE',  # Massachusetts
    '33': 'NE',  # New Hampshire
    '44': 'NE',  # Rhode Island
    '50': 'NE',  # Vermont
    '34': 'NE',  # New Jersey
    '36': 'NE',  # New York
    '42': 'NE',  # Pennsylvania
    '10': 'NE',  # Delaware
    '24': 'NE',  # Maryland
    '11': 'NE',  # DC

    # ----- LAKE STATES (LS) -----
    '26': 'LS',  # Michigan
    '55': 'LS',  # Wisconsin
    '27': 'LS',  # Minnesota

    # ----- CORN BELT (CB) -----
    '17': 'CB',  # Illinois
    '18': 'CB',  # Indiana
    '19': 'CB',  # Iowa
    '29': 'CB',  # Missouri
    '39': 'CB',  # Ohio

    # ----- NORTHERN PLAINS (NP) -----
    '20': 'NP',  # Kansas
    '31': 'NP',  # Nebraska
    '38': 'NP',  # North Dakota
    '46': 'NP',  # South Dakota

    # ----- APPALACHIAN (AP) -----
    '21': 'AP',  # Kentucky
    '37': 'AP',  # North Carolina
    '47': 'AP',  # Tennessee
    '51': 'AP',  # Virginia
    '54': 'AP',  # West Virginia

    # ----- SOUTHEAST (SE) -----
    '01': 'SE',  # Alabama
    '12': 'SE',  # Florida
    '13': 'SE',  # Georgia
    '45': 'SE',  # South Carolina

    # ----- DELTA (DL) -----
    '05': 'DL',  # Arkansas
    '22': 'DL',  # Louisiana
    '28': 'DL',  # Mississippi

    # ----- SOUTHERN PLAINS (SP) -----
    '40': 'SP',  # Oklahoma
    '48': 'SP',  # Texas

    # ----- MOUNTAIN (MT) -----
    '04': 'MT',  # Arizona
    '08': 'MT',  # Colorado
    '16': 'MT',  # Idaho
    '30': 'MT',  # Montana
    '32': 'MT',  # Nevada
    '35': 'MT',  # New Mexico
    '49': 'MT',  # Utah
    '56': 'MT',  # Wyoming

    # ----- PACIFIC COAST (PC) -----
    '02': 'PC',  # Alaska
    '06': 'PC',  # California
    '15': 'PC',  # Hawaii
    '41': 'PC',  # Oregon
    '53': 'PC',  # Washington
}

# Subregion to RPA Region mapping (for aggregation if needed)
SUBREGION_TO_REGION = {
    'NE': 'NO',  # Northeast -> North
    'LS': 'NO',  # Lake States -> North
    'CB': 'NO',  # Corn Belt -> North
    'NP': 'NO',  # Northern Plains -> North (could be RM)
    'AP': 'SO',  # Appalachian -> South
    'SE': 'SO',  # Southeast -> South
    'DL': 'SO',  # Delta -> South
    'SP': 'SO',  # Southern Plains -> South
    'MT': 'RM',  # Mountain -> Rocky Mountain
    'PC': 'PC',  # Pacific Coast -> Pacific Coast
}

# Alias for backward compatibility
STATE_REGIONS = STATE_TO_SUBREGION


def nri_to_4cat(broad_code: int, irrtyp: int) -> Optional[int]:
    """
    Convert NRI BROAD code + irrigation type to 4-category code.

    Args:
        broad_code: NRI BROAD land use code
        irrtyp: Irrigation type (0 = non-irrigated, >0 = irrigated)

    Returns:
        4-category code (1-4) or None if not in our categories
    """
    if broad_code == NRI_BROAD_CROP:
        # Split cropland by irrigation status
        return LANDUSE_4CAT['CR_IRR'] if irrtyp > 0 else LANDUSE_4CAT['CR_DRY']
    elif broad_code == NRI_BROAD_PASTURE:
        return LANDUSE_4CAT['PS']
    elif broad_code == NRI_BROAD_URBAN:
        return LANDUSE_4CAT['UR']
    else:
        return None


def nri_to_5cat(broad_code: int, irrtyp: int = None) -> Optional[int]:
    """
    Convert NRI BROAD code to 5-category code (combined cropland).

    Args:
        broad_code: NRI BROAD land use code
        irrtyp: Irrigation type (ignored - cropland is combined)

    Returns:
        5-category code (1-5) or None if not in our categories
    """
    if broad_code == NRI_BROAD_CROP:
        return LANDUSE_5CAT['CR']  # Combined cropland
    elif broad_code == NRI_BROAD_PASTURE:
        return LANDUSE_5CAT['PS']
    elif broad_code == NRI_BROAD_RANGE:
        return LANDUSE_5CAT['RG']
    elif broad_code == NRI_BROAD_FOREST:
        return LANDUSE_5CAT['FR']
    elif broad_code == NRI_BROAD_URBAN:
        return LANDUSE_5CAT['UR']
    else:
        return None


def get_state_region(fips: str) -> str:
    """Get region code from FIPS."""
    state_fips = str(fips).zfill(5)[:2]
    return STATE_REGIONS.get(state_fips, 'WE')  # Default to West if unknown


def extract_transitions(
    nri_file: Path,
    output_dir: Path,
    start_year: int = 12,
    end_year: int = 17,
    chunk_size: int = 50000,
    sample_size: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Extract land use transitions from raw NRI data.

    Args:
        nri_file: Path to nri17_csv.csv
        output_dir: Directory to save output files
        start_year: Starting year (2-digit, e.g., 12 for 2012)
        end_year: Ending year (2-digit, e.g., 17 for 2017)
        chunk_size: Rows to process at a time
        sample_size: Optional limit on total rows to process

    Returns:
        Dictionary of DataFrames by starting land use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Columns we need from NRI
    base_cols = ['state', 'county', 'fips', 'xfact']
    year_suffixes = [f'{y:02d}' for y in range(start_year, end_year + 1)]

    broad_cols = [f'broad{y}' for y in year_suffixes]
    irrtyp_cols = [f'irrtyp{y}' for y in year_suffixes]
    lcc_cols = [f'lcc{y}' for y in year_suffixes]

    usecols = base_cols + broad_cols + irrtyp_cols + lcc_cols

    console.print(f"[bold blue]Extracting NRI transitions from years 20{start_year}-20{end_year}[/]")
    console.print(f"Reading columns: {len(usecols)} total")

    # Collect all transitions
    all_transitions = []
    rows_processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing NRI chunks...", total=None)

        for chunk in pd.read_csv(nri_file, usecols=usecols, chunksize=chunk_size,
                                  dtype={'fips': str, 'state': str, 'county': str}):

            if sample_size and rows_processed >= sample_size:
                break

            # Process each transition period (year t to year t+1)
            for i, y in enumerate(year_suffixes[:-1]):
                next_y = year_suffixes[i + 1]
                year_int = 2000 + int(y)

                broad_start = f'broad{y}'
                broad_end = f'broad{next_y}'
                irrtyp_start = f'irrtyp{y}'
                irrtyp_end = f'irrtyp{next_y}'
                lcc_col = f'lcc{y}'

                # Filter to rows with valid land use codes
                mask = (
                    chunk[broad_start].isin([NRI_BROAD_CROP, NRI_BROAD_PASTURE, NRI_BROAD_URBAN]) &
                    chunk[broad_end].isin([NRI_BROAD_CROP, NRI_BROAD_PASTURE, NRI_BROAD_URBAN])
                )

                subset = chunk[mask].copy()
                if len(subset) == 0:
                    continue

                # Fill missing irrtyp with 0 (non-irrigated)
                subset[irrtyp_start] = pd.to_numeric(subset[irrtyp_start], errors='coerce').fillna(0).astype(int)
                subset[irrtyp_end] = pd.to_numeric(subset[irrtyp_end], errors='coerce').fillna(0).astype(int)

                # Convert to 4-category codes
                subset['startuse'] = subset.apply(
                    lambda r: nri_to_4cat(r[broad_start], r[irrtyp_start]), axis=1
                )
                subset['enduse'] = subset.apply(
                    lambda r: nri_to_4cat(r[broad_end], r[irrtyp_end]), axis=1
                )

                # Clean LCC (extract first digit, default to 4)
                subset['lcc'] = (
                    subset[lcc_col]
                    .astype(str)
                    .str.extract(r'(\d)')[0]
                    .fillna('4')
                    .astype(int)
                    .clip(1, 8)
                )

                # Clean xfact
                subset['xfact'] = pd.to_numeric(subset['xfact'], errors='coerce').fillna(1.0)

                # Build output dataframe
                transitions = pd.DataFrame({
                    'fips': subset['fips'].astype(str).str.zfill(5).astype(int),
                    'year': year_int,
                    'riad_id': subset.index.astype(str) + '_' + y,
                    'startuse': subset['startuse'],
                    'enduse': subset['enduse'],
                    'lcc': subset['lcc'],
                    'xfact': subset['xfact']
                })

                # Drop rows with invalid codes
                transitions = transitions.dropna(subset=['startuse', 'enduse'])
                transitions['startuse'] = transitions['startuse'].astype(int)
                transitions['enduse'] = transitions['enduse'].astype(int)

                all_transitions.append(transitions)

            rows_processed += len(chunk)
            progress.update(task, description=f"Processed {rows_processed:,} rows...")

            if sample_size and rows_processed >= sample_size:
                break

    # Combine all transitions
    console.print(f"\n[green]Processed {rows_processed:,} total rows[/]")

    if not all_transitions:
        console.print("[red]No valid transitions found![/]")
        return {}

    df = pd.concat(all_transitions, ignore_index=True)
    console.print(f"[green]Total transitions: {len(df):,}[/]")

    # Split by starting land use
    results = {}
    for code, name in LANDUSE_NAMES.items():
        subset = df[df['startuse'] == code].copy()
        if len(subset) > 0:
            results[name] = subset
            console.print(f"  {name}: {len(subset):,} observations")

    # Create georef file
    georef = df[['fips']].drop_duplicates().copy()
    georef['county_fips'] = georef['fips']
    georef['state_fips'] = georef['fips'].astype(str).str.zfill(5).str[:2]
    georef['region'] = georef['state_fips'].map(lambda x: STATE_REGIONS.get(x, 'WE'))
    georef['subregion'] = georef['region']  # Simplified - same as region
    georef = georef[['fips', 'county_fips', 'subregion', 'region']]

    # Save files
    console.print("\n[bold]Saving output files...[/]")

    file_mapping = {
        'CR_IRR': 'start_cr_irr.csv',
        'CR_DRY': 'start_cr_dry.csv',
        'PS': 'start_ps.csv',
        'UR': 'start_ur.csv'
    }

    for name, filename in file_mapping.items():
        if name in results:
            filepath = output_dir / filename
            results[name].to_csv(filepath, index=False)
            console.print(f"  Saved: {filepath}")

    georef_path = output_dir / 'georef.csv'
    georef.to_csv(georef_path, index=False)
    console.print(f"  Saved: {georef_path}")

    # Print transition matrix
    print_transition_matrix(df)

    return results


def print_transition_matrix(df: pd.DataFrame, landuse_names: Dict = None) -> None:
    """Print a transition matrix summary."""
    if landuse_names is None:
        landuse_names = LANDUSE_NAMES

    console.print("\n[bold]Transition Matrix (row % of starts)[/]")

    # Create cross-tabulation
    cross = pd.crosstab(
        df['startuse'].map(landuse_names),
        df['enduse'].map(landuse_names),
        normalize='index'
    ) * 100

    table = Table(title="Land Use Transitions (%)")
    table.add_column("From \\ To", style="bold")

    for col in cross.columns:
        table.add_column(col, justify="right")

    for idx in cross.index:
        row_values = [f"{cross.loc[idx, col]:.1f}" for col in cross.columns]
        table.add_row(idx, *row_values)

    console.print(table)


def extract_transitions_5cat(
    nri_file: Path,
    output_dir: Path,
    start_year: int = 12,
    end_year: int = 17,
    chunk_size: int = 50000,
    sample_size: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Extract 5-category land use transitions from raw NRI data.

    This version combines irrigated and non-irrigated cropland into
    a single "Cropland" category for more statistical power.

    Categories:
        1 = Cropland (CR) - combined irrigated + non-irrigated
        2 = Pasture (PS)
        3 = Rangeland (RG)
        4 = Forest (FR)
        5 = Urban (UR)

    Args:
        nri_file: Path to nri17_csv.csv
        output_dir: Directory to save output files
        start_year: Starting year (2-digit, e.g., 12 for 2012)
        end_year: Ending year (2-digit, e.g., 17 for 2017)
        chunk_size: Rows to process at a time
        sample_size: Optional limit on total rows to process

    Returns:
        Dictionary of DataFrames by starting land use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Valid BROAD codes for 5-category model
    valid_broad_codes = [NRI_BROAD_CROP, NRI_BROAD_PASTURE, NRI_BROAD_RANGE,
                         NRI_BROAD_FOREST, NRI_BROAD_URBAN]

    # Columns we need from NRI
    base_cols = ['state', 'county', 'fips', 'xfact']
    year_suffixes = [f'{y:02d}' for y in range(start_year, end_year + 1)]

    broad_cols = [f'broad{y}' for y in year_suffixes]
    lcc_cols = [f'lcc{y}' for y in year_suffixes]

    # Note: We don't need irrtyp for 5-category model since cropland is combined
    usecols = base_cols + broad_cols + lcc_cols

    console.print(f"[bold blue]Extracting 5-category NRI transitions from years 20{start_year}-20{end_year}[/]")
    console.print(f"Categories: CR (combined), PS, RG, FR, UR")
    console.print(f"Reading columns: {len(usecols)} total")

    # Collect all transitions
    all_transitions = []
    rows_processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing NRI chunks...", total=None)

        for chunk in pd.read_csv(nri_file, usecols=usecols, chunksize=chunk_size,
                                  dtype={'fips': str, 'state': str, 'county': str}):

            if sample_size and rows_processed >= sample_size:
                break

            # Process each transition period (year t to year t+1)
            for i, y in enumerate(year_suffixes[:-1]):
                next_y = year_suffixes[i + 1]
                year_int = 2000 + int(y)

                broad_start = f'broad{y}'
                broad_end = f'broad{next_y}'
                lcc_col = f'lcc{y}'

                # Filter to rows with valid land use codes (all 5 categories)
                mask = (
                    chunk[broad_start].isin(valid_broad_codes) &
                    chunk[broad_end].isin(valid_broad_codes)
                )

                subset = chunk[mask].copy()
                if len(subset) == 0:
                    continue

                # Convert to 5-category codes (no irrigation split)
                subset['startuse'] = subset[broad_start].apply(nri_to_5cat)
                subset['enduse'] = subset[broad_end].apply(nri_to_5cat)

                # Clean LCC (extract first digit, default to 4)
                subset['lcc'] = (
                    subset[lcc_col]
                    .astype(str)
                    .str.extract(r'(\d)')[0]
                    .fillna('4')
                    .astype(int)
                    .clip(1, 8)
                )

                # Clean xfact
                subset['xfact'] = pd.to_numeric(subset['xfact'], errors='coerce').fillna(1.0)

                # Build output dataframe
                transitions = pd.DataFrame({
                    'fips': subset['fips'].astype(str).str.zfill(5).astype(int),
                    'year': year_int,
                    'riad_id': subset.index.astype(str) + '_' + y,
                    'startuse': subset['startuse'],
                    'enduse': subset['enduse'],
                    'lcc': subset['lcc'],
                    'xfact': subset['xfact']
                })

                # Drop rows with invalid codes
                transitions = transitions.dropna(subset=['startuse', 'enduse'])
                transitions['startuse'] = transitions['startuse'].astype(int)
                transitions['enduse'] = transitions['enduse'].astype(int)

                all_transitions.append(transitions)

            rows_processed += len(chunk)
            progress.update(task, description=f"Processed {rows_processed:,} rows...")

            if sample_size and rows_processed >= sample_size:
                break

    # Combine all transitions
    console.print(f"\n[green]Processed {rows_processed:,} total rows[/]")

    if not all_transitions:
        console.print("[red]No valid transitions found![/]")
        return {}

    df = pd.concat(all_transitions, ignore_index=True)
    console.print(f"[green]Total transitions: {len(df):,}[/]")

    # Split by starting land use
    results = {}
    for code, name in LANDUSE_NAMES_5CAT.items():
        subset = df[df['startuse'] == code].copy()
        if len(subset) > 0:
            results[name] = subset
            console.print(f"  {name}: {len(subset):,} observations")

    # Create georef file
    georef = df[['fips']].drop_duplicates().copy()
    georef['county_fips'] = georef['fips']
    georef['state_fips'] = georef['fips'].astype(str).str.zfill(5).str[:2]
    georef['region'] = georef['state_fips'].map(lambda x: STATE_REGIONS.get(x, 'WE'))
    georef['subregion'] = georef['region']  # Simplified - same as region
    georef = georef[['fips', 'county_fips', 'subregion', 'region']]

    # Save files
    console.print("\n[bold]Saving output files...[/]")

    file_mapping = {
        'CR': 'start_crop.csv',
        'PS': 'start_ps.csv',
        'RG': 'start_rg.csv',
        'FR': 'start_fr.csv',
        'UR': 'start_ur.csv'
    }

    for name, filename in file_mapping.items():
        if name in results:
            filepath = output_dir / filename
            results[name].to_csv(filepath, index=False)
            console.print(f"  Saved: {filepath}")

    georef_path = output_dir / 'georef.csv'
    georef.to_csv(georef_path, index=False)
    console.print(f"  Saved: {georef_path}")

    # Print transition matrix
    print_transition_matrix(df, LANDUSE_NAMES_5CAT)

    return results


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract land use transitions from NRI data"
    )
    parser.add_argument(
        'nri_file',
        type=Path,
        help='Path to nri17_csv.csv'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/processed'),
        help='Output directory (default: data/processed)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=12,
        help='Start year (2-digit, default: 12 for 2012)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=17,
        help='End year (2-digit, default: 17 for 2017)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50000,
        help='Chunk size for reading CSV (default: 50000)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size limit (default: process all)'
    )

    args = parser.parse_args()

    if not args.nri_file.exists():
        console.print(f"[red]Error: NRI file not found: {args.nri_file}[/]")
        return 1

    extract_transitions(
        nri_file=args.nri_file,
        output_dir=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        chunk_size=args.chunk_size,
        sample_size=args.sample
    )

    return 0


if __name__ == '__main__':
    exit(main())
