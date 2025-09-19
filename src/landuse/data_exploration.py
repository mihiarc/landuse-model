"""
Data exploration and visualization utilities for land use analysis.
Ported from nri_data_explore.R and metro_population_changes.R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats


def explore_nri_data(nri_file: str,
                     output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Explore NRI (Natural Resources Inventory) data patterns.

    Parameters:
    -----------
    nri_file : str
        Path to NRI data file
    output_dir : str, optional
        Directory to save exploration results

    Returns:
    --------
    pd.DataFrame
        Summary statistics of NRI data
    """
    # Load NRI data
    if nri_file.endswith('.csv'):
        nri = pd.read_csv(nri_file)
    else:
        nri = pd.read_pickle(nri_file)

    print("NRI Data Overview:")
    print(f"Shape: {nri.shape}")
    print(f"Columns: {list(nri.columns)}")
    print("\nData types:")
    print(nri.dtypes)

    # Basic statistics
    summary = nri.describe()
    print("\nSummary Statistics:")
    print(summary)

    # Check for missing values
    missing = nri.isnull().sum()
    if missing.any():
        print("\nMissing Values:")
        print(missing[missing > 0])

    # Analyze land use categories if present
    if 'landuse' in nri.columns or 'land_use' in nri.columns:
        lu_col = 'landuse' if 'landuse' in nri.columns else 'land_use'
        print(f"\nLand Use Distribution:")
        print(nri[lu_col].value_counts())

    # Analyze temporal patterns if year column exists
    if 'year' in nri.columns:
        print("\nYear Range:")
        print(f"From {nri['year'].min()} to {nri['year'].max()}")

        # Trends over time
        yearly_stats = nri.groupby('year').agg({
            col: ['mean', 'std'] for col in nri.select_dtypes(include=[np.number]).columns
        })

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path / 'nri_summary_stats.csv')

    return summary


def analyze_metro_population_changes(population_file: str,
                                    metro_definition_file: Optional[str] = None,
                                    years: List[int] = None) -> pd.DataFrame:
    """
    Analyze population changes in metropolitan areas.

    Parameters:
    -----------
    population_file : str
        Path to population data file
    metro_definition_file : str, optional
        Path to metro area definitions
    years : List[int], optional
        Years to analyze

    Returns:
    --------
    pd.DataFrame
        Metro area population changes
    """
    # Load population data
    if population_file.endswith('.csv'):
        pop_data = pd.read_csv(population_file)
    else:
        pop_data = pd.read_pickle(population_file)

    # Load metro definitions if provided
    if metro_definition_file:
        if metro_definition_file.endswith('.csv'):
            metro_def = pd.read_csv(metro_definition_file)
        else:
            metro_def = pd.read_pickle(metro_definition_file)

        # Merge with population data
        pop_data = pop_data.merge(metro_def, on='fips', how='left')

    # Filter years if specified
    if years and 'year' in pop_data.columns:
        pop_data = pop_data[pop_data['year'].isin(years)]

    # Calculate population changes
    if 'year' in pop_data.columns:
        # Sort by geographic unit and year
        pop_data = pop_data.sort_values(['fips', 'year'])

        # Calculate year-over-year changes
        pop_data['pop_change'] = pop_data.groupby('fips')['population'].diff()
        pop_data['pop_pct_change'] = pop_data.groupby('fips')['population'].pct_change() * 100

        # Calculate growth rates
        def calculate_growth_rate(group):
            if len(group) > 1:
                years_diff = group['year'].max() - group['year'].min()
                if years_diff > 0:
                    initial_pop = group['population'].iloc[0]
                    final_pop = group['population'].iloc[-1]
                    if initial_pop > 0:
                        growth_rate = ((final_pop / initial_pop) ** (1 / years_diff) - 1) * 100
                        return growth_rate
            return np.nan

        growth_rates = pop_data.groupby('fips').apply(calculate_growth_rate)
        growth_rates = growth_rates.reset_index()
        growth_rates.columns = ['fips', 'annual_growth_rate']

        # Merge growth rates back
        pop_data = pop_data.merge(growth_rates, on='fips', how='left')

    # Aggregate to metro level if metro areas are defined
    if 'metro_area' in pop_data.columns or 'cbsa' in pop_data.columns:
        metro_col = 'metro_area' if 'metro_area' in pop_data.columns else 'cbsa'

        metro_summary = pop_data.groupby([metro_col, 'year']).agg({
            'population': 'sum',
            'pop_change': 'sum',
            'fips': 'count'
        }).reset_index()
        metro_summary.rename(columns={'fips': 'num_counties'}, inplace=True)

        # Calculate metro-level growth rates
        metro_summary = metro_summary.sort_values([metro_col, 'year'])
        metro_summary['metro_pop_pct_change'] = metro_summary.groupby(metro_col)['population'].pct_change() * 100

        return metro_summary

    return pop_data


def create_transition_matrix(data: pd.DataFrame,
                            start_col: str = 'startuse',
                            end_col: str = 'enduse',
                            weight_col: Optional[str] = 'xfact') -> pd.DataFrame:
    """
    Create land use transition matrix.

    Parameters:
    -----------
    data : pd.DataFrame
        Data with land use transitions
    start_col : str
        Column name for starting land use
    end_col : str
        Column name for ending land use
    weight_col : str, optional
        Column for weighting observations

    Returns:
    --------
    pd.DataFrame
        Transition probability matrix
    """
    if weight_col and weight_col in data.columns:
        # Weighted transition matrix
        transition = data.groupby([start_col, end_col])[weight_col].sum().unstack(fill_value=0)
    else:
        # Unweighted transition matrix
        transition = pd.crosstab(data[start_col], data[end_col])

    # Convert to probabilities
    transition_prob = transition.div(transition.sum(axis=1), axis=0)

    return transition_prob


def plot_transition_heatmap(transition_matrix: pd.DataFrame,
                           title: str = "Land Use Transition Probabilities",
                           output_file: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap of transition probabilities.

    Parameters:
    -----------
    transition_matrix : pd.DataFrame
        Transition probability matrix
    title : str
        Plot title
    output_file : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1, cbar_kws={'label': 'Transition Probability'},
                ax=ax)

    ax.set_title(title)
    ax.set_xlabel('End Use')
    ax.set_ylabel('Start Use')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def analyze_spatial_patterns(data: pd.DataFrame,
                            shapefile: str,
                            variable: str,
                            fips_col: str = 'fips') -> gpd.GeoDataFrame:
    """
    Analyze spatial patterns in the data.

    Parameters:
    -----------
    data : pd.DataFrame
        Data to analyze
    shapefile : str
        Path to shapefile with geometries
    variable : str
        Variable to map
    fips_col : str
        Column with FIPS codes

    Returns:
    --------
    gpd.GeoDataFrame
        Merged spatial data
    """
    # Load shapefile
    gdf = gpd.read_file(shapefile)

    # Ensure FIPS codes are strings with proper formatting
    data[fips_col] = data[fips_col].astype(str).str.zfill(5)
    if 'GEOID' in gdf.columns:
        gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(5)
    elif 'FIPS' in gdf.columns:
        gdf['FIPS'] = gdf['FIPS'].astype(str).str.zfill(5)

    # Merge data with geometries
    merge_col = 'GEOID' if 'GEOID' in gdf.columns else 'FIPS'
    spatial_data = gdf.merge(data, left_on=merge_col, right_on=fips_col, how='left')

    # Calculate spatial statistics
    if variable in spatial_data.columns:
        # Moran's I for spatial autocorrelation
        from libpysal.weights import Queen
        w = Queen.from_dataframe(spatial_data)

        from esda.moran import Moran
        moran = Moran(spatial_data[variable].dropna(), w)

        print(f"Spatial Autocorrelation Analysis for {variable}:")
        print(f"Moran's I: {moran.I:.4f}")
        print(f"P-value: {moran.p_norm:.4f}")

    return spatial_data


def create_temporal_plots(data: pd.DataFrame,
                         variables: List[str],
                         time_col: str = 'year',
                         group_col: Optional[str] = None,
                         output_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Create temporal trend plots.

    Parameters:
    -----------
    data : pd.DataFrame
        Data with temporal information
    variables : List[str]
        Variables to plot
    time_col : str
        Column with time information
    group_col : str, optional
        Column to group by
    output_dir : str, optional
        Directory to save plots

    Returns:
    --------
    Dict[str, plt.Figure]
        Dictionary of figures
    """
    figures = {}

    for var in variables:
        if var not in data.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        if group_col and group_col in data.columns:
            # Plot by groups
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group]
                temporal_mean = group_data.groupby(time_col)[var].mean()
                ax.plot(temporal_mean.index, temporal_mean.values, marker='o', label=str(group))
            ax.legend()
        else:
            # Single line plot
            temporal_mean = data.groupby(time_col)[var].mean()
            temporal_std = data.groupby(time_col)[var].std()

            ax.plot(temporal_mean.index, temporal_mean.values, marker='o', color='blue', label='Mean')
            ax.fill_between(temporal_mean.index,
                           temporal_mean.values - temporal_std.values,
                           temporal_mean.values + temporal_std.values,
                           alpha=0.3, color='blue', label='±1 Std Dev')
            ax.legend()

        ax.set_xlabel(time_col.capitalize())
        ax.set_ylabel(var)
        ax.set_title(f'Temporal Trends: {var}')
        ax.grid(True, alpha=0.3)

        figures[var] = fig

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path / f'{var}_temporal_trend.pdf', dpi=300, bbox_inches='tight')

    return figures


def generate_summary_report(data_dict: Dict[str, pd.DataFrame],
                           output_file: str):
    """
    Generate comprehensive summary report of all datasets.

    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of datasets to summarize
    output_file : str
        Path to save report
    """
    with open(output_file, 'w') as f:
        f.write("Land Use Data Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")

        for name, data in data_dict.items():
            f.write(f"\nDataset: {name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Shape: {data.shape}\n")
            f.write(f"Memory usage: {data.memory_usage().sum() / 1024**2:.2f} MB\n")

            # Data types
            f.write("\nColumn Types:\n")
            for dtype, count in data.dtypes.value_counts().items():
                f.write(f"  {dtype}: {count} columns\n")

            # Missing values
            missing = data.isnull().sum()
            if missing.any():
                f.write("\nColumns with missing values:\n")
                for col, count in missing[missing > 0].items():
                    pct = count / len(data) * 100
                    f.write(f"  {col}: {count} ({pct:.1f}%)\n")

            # Numeric summary
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                f.write("\nNumeric columns summary:\n")
                summary = data[numeric_cols].describe()
                f.write(summary.to_string())
                f.write("\n")

            f.write("\n")

    print(f"Summary report saved to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_exploration.py <data_file> [output_dir]")
        sys.exit(1)

    data_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "exploration_results"

    # Run exploration based on file type
    if "nri" in data_file.lower():
        explore_nri_data(data_file, output_dir)
    elif "pop" in data_file.lower() or "metro" in data_file.lower():
        analyze_metro_population_changes(data_file)
    else:
        print(f"Exploring general data file: {data_file}")
        if data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
        else:
            data = pd.read_pickle(data_file)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary = data.describe()
        summary.to_csv(Path(output_dir) / 'summary_stats.csv')
        print(f"Summary statistics saved to {output_dir}/summary_stats.csv")