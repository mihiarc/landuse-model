"""
Climate change impact on land use conversion probabilities.
Ported from cc_impact_landuse.R
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calc_utility(model_coeffs: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate utility values for each land use type.

    Parameters:
    -----------
    model_coeffs : np.ndarray
        Model coefficients array
    df : pd.DataFrame
        Input dataframe with required columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with added utility columns
    """
    df = df.copy()

    df['crop_util'] = model_coeffs[6] * df['nr.cr']

    df['pasture_util'] = (model_coeffs[2] +
                          model_coeffs[3] * df['lcc'] +
                          model_coeffs[8] * df['nr.ps'])

    df['forest_util'] = (model_coeffs[0] +
                        model_coeffs[1] * df['lcc'] +
                        model_coeffs[7] * df['nr.fr'])

    df['urban_util'] = (model_coeffs[4] +
                       model_coeffs[5] * df['lcc'] +
                       model_coeffs[9] * df['nr.ur'])

    return df


def calc_prob_chg(model_coeffs: np.ndarray,
                  df1: pd.DataFrame,
                  df2: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate probability changes between two scenarios.

    Parameters:
    -----------
    model_coeffs : np.ndarray
        Model coefficients
    df1 : pd.DataFrame
        Baseline scenario dataframe
    df2 : pd.DataFrame
        Alternative scenario dataframe

    Returns:
    --------
    pd.DataFrame
        DataFrame with probability changes
    """
    df1 = df1.copy()
    df1['nr.cr'] = df1['crnr_obs']
    df1['nr.fr'] = df1['frnr_obs']
    df1['nr.ur'] = df1['urnr_obs']

    df1 = calc_utility(model_coeffs, df1)
    df2 = calc_utility(model_coeffs, df2)

    # Calculate probabilities for baseline
    total_exp1 = (np.exp(df1['crop_util']) + np.exp(df1['pasture_util']) +
                 np.exp(df1['forest_util']) + np.exp(df1['urban_util']))

    df1['pr_crop0'] = np.exp(df1['crop_util']) / total_exp1
    df1['pr_pasture0'] = np.exp(df1['pasture_util']) / total_exp1
    df1['pr_forest0'] = np.exp(df1['forest_util']) / total_exp1
    df1['pr_urban0'] = np.exp(df1['urban_util']) / total_exp1

    df1 = df1[['fips', 'riad_id', 'xfact'] + [col for col in df1.columns if col.startswith('pr')]]

    # Calculate probabilities for alternative scenario
    total_exp2 = (np.exp(df2['crop_util']) + np.exp(df2['pasture_util']) +
                 np.exp(df2['forest_util']) + np.exp(df2['urban_util']))

    df2['pr_crop1'] = np.exp(df2['crop_util']) / total_exp2
    df2['pr_pasture1'] = np.exp(df2['pasture_util']) / total_exp2
    df2['pr_forest1'] = np.exp(df2['forest_util']) / total_exp2
    df2['pr_urban1'] = np.exp(df2['urban_util']) / total_exp2

    df2 = df2[[col for col in df2.columns if col.startswith('pr')]]

    # Combine and calculate changes
    df = pd.concat([df1, df2], axis=1)
    df['probchg_crop'] = df['pr_crop1'] - df['pr_crop0']
    df['probchg_pasture'] = df['pr_pasture1'] - df['pr_pasture0']
    df['probchg_forest'] = df['pr_forest1'] - df['pr_forest0']
    df['probchg_urban'] = df['pr_urban1'] - df['pr_urban0']

    return df


def aggr_to_county(mfx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate marginal effects to county level.

    Parameters:
    -----------
    mfx_df : pd.DataFrame
        DataFrame with marginal effects

    Returns:
    --------
    pd.DataFrame
        County-level aggregated data
    """
    def weighted_mean(x, w):
        return np.average(x, weights=w)

    result = mfx_df.groupby('fips').apply(
        lambda x: pd.Series({
            'mfx_crop': weighted_mean(x['mfx_crop'], x['xfact']),
            'mfx_pasture': weighted_mean(x['mfx_pasture'], x['xfact']),
            'mfx_forest': weighted_mean(x['mfx_forest'], x['xfact']),
            'mfx_urban': weighted_mean(x['mfx_urban'], x['xfact'])
        })
    ).reset_index()

    return result


def load_climate_impacts(cc_impacts_dir: str) -> pd.DataFrame:
    """
    Load and format climate change impact data.

    Parameters:
    -----------
    cc_impacts_dir : str
        Directory containing climate impact data files

    Returns:
    --------
    pd.DataFrame
        Combined climate impact data
    """
    cc_dir = Path(cc_impacts_dir)

    # Load crop impacts
    cc_crop = pd.read_pickle(cc_dir / 'crop_climate_change_impact.rds')
    cc_crop['fips'] = cc_crop['GEOID'].astype(int)
    cc_crop = cc_crop[['fips', 'nr', 'impact']].rename(columns={'nr': 'crnr_obs', 'impact': 'crnr_impact'})

    # Load forest impacts
    cc_forest = pd.read_pickle(cc_dir / 'forest_climate_change_impact.rds')
    cc_forest['fips'] = cc_forest['GEOID'].astype(int)
    cc_forest = cc_forest[['fips', 'nracre', 'impact']].rename(columns={'nracre': 'frnr_obs', 'impact': 'frnr_impact'})

    # Load urban impacts
    cc_urban = pd.read_pickle(cc_dir / 'urban_climate_change_impact.rds')
    cc_urban = cc_urban[['fips', 'urnr', 'impact']].rename(columns={'urnr': 'urnr_obs', 'impact': 'urnr_impact'})

    # Merge all impacts
    cc_impact = cc_urban.merge(cc_forest, on='fips', how='outer').merge(cc_crop, on='fips', how='outer')

    return cc_impact


def calculate_scenario_impacts(df_start: pd.DataFrame,
                               cc_impact: pd.DataFrame,
                               model_coeffs: np.ndarray,
                               start_type: str,
                               region_filter: str = 'SO',
                               year_filter: int = 2012) -> Dict[str, pd.DataFrame]:
    """
    Calculate climate impacts for different scenarios.

    Parameters:
    -----------
    df_start : pd.DataFrame
        Starting land use data
    cc_impact : pd.DataFrame
        Climate impact data
    model_coeffs : np.ndarray
        Model coefficients
    start_type : str
        Starting land use type ('crop' or 'forest')
    region_filter : str
        Region to filter
    year_filter : int
        Year to filter

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of impact scenarios
    """
    # Merge and filter data
    df = df_start.merge(cc_impact, on='fips', how='inner')
    df = df[(df['region'] == region_filter) & (df['year'] == year_filter)]
    df = df.drop(columns=[col for col in df.columns if col.startswith('nrmean') or col.startswith('nrchange')])
    df = df.drop(columns=['subregion', 'region', 'weight', 'year'], errors='ignore')

    impacts = {}

    # Full impact
    df_full = df.copy()
    df_full['nr.cr'] = df_full['crnr_obs'] + df_full['crnr_impact']
    df_full['nr.fr'] = df_full['frnr_obs'] + df_full['frnr_impact']
    df_full['nr.ur'] = df_full['urnr_obs'] + df_full['urnr_impact']

    full_impact = calc_prob_chg(model_coeffs, df, df_full)
    full_impact = full_impact[['fips', 'riad_id', 'xfact'] + [col for col in full_impact.columns if col.startswith('probchg')]]
    full_impact = pd.melt(full_impact, id_vars=['fips', 'riad_id', 'xfact'],
                          value_vars=[col for col in full_impact.columns if col.startswith('probchg')],
                          var_name='landuse', value_name='probchg')
    full_impact['impact_type'] = f'full_impact_start{start_type}'
    impacts['full'] = full_impact

    # Crop impact only
    df_crop = df.copy()
    df_crop['nr.cr'] = df_crop['crnr_obs'] + df_crop['crnr_impact']

    crop_impact = calc_prob_chg(model_coeffs, df, df_crop)
    crop_impact = crop_impact[['fips', 'riad_id', 'xfact'] + [col for col in crop_impact.columns if col.startswith('probchg')]]
    crop_impact = pd.melt(crop_impact, id_vars=['fips', 'riad_id', 'xfact'],
                          value_vars=[col for col in crop_impact.columns if col.startswith('probchg')],
                          var_name='landuse', value_name='probchg')
    crop_impact['impact_type'] = f'crop_impact_start{start_type}'
    impacts['crop'] = crop_impact

    # Forest impact only
    df_forest = df.copy()
    df_forest['nr.fr'] = df_forest['frnr_obs'] + df_forest['frnr_impact']

    forest_impact = calc_prob_chg(model_coeffs, df, df_forest)
    forest_impact = forest_impact[['fips', 'riad_id', 'xfact'] + [col for col in forest_impact.columns if col.startswith('probchg')]]
    forest_impact = pd.melt(forest_impact, id_vars=['fips', 'riad_id', 'xfact'],
                           value_vars=[col for col in forest_impact.columns if col.startswith('probchg')],
                           var_name='landuse', value_name='probchg')
    forest_impact['impact_type'] = f'forest_impact_start{start_type}'
    impacts['forest'] = forest_impact

    # Urban impact only
    df_urban = df.copy()
    df_urban['nr.ur'] = df_urban['urnr_obs'] + df_urban['urnr_impact']

    urban_impact = calc_prob_chg(model_coeffs, df, df_urban)
    urban_impact = urban_impact[['fips', 'riad_id', 'xfact'] + [col for col in urban_impact.columns if col.startswith('probchg')]]
    urban_impact = pd.melt(urban_impact, id_vars=['fips', 'riad_id', 'xfact'],
                          value_vars=[col for col in urban_impact.columns if col.startswith('probchg')],
                          var_name='landuse', value_name='probchg')
    urban_impact['impact_type'] = f'urban_impact_start{start_type}'
    impacts['urban'] = urban_impact

    return impacts


def create_impact_boxplots(impacts_df: pd.DataFrame,
                          start_type: str,
                          output_file: Optional[str] = None) -> plt.Figure:
    """
    Create boxplots showing climate change impacts on conversion probabilities.

    Parameters:
    -----------
    impacts_df : pd.DataFrame
        Combined impact data
    start_type : str
        Starting land use type
    output_file : str, optional
        Path to save the figure

    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    landuses = ['crop', 'forest', 'pasture', 'urban']
    impact_types = ['full', 'crop', 'forest', 'urban']

    for i, landuse in enumerate(landuses):
        ax = axes[i]

        plot_data = []
        labels = []

        for impact in impact_types:
            mask = (impacts_df['type_landuse'].str.contains(f'{impact}_impact_start{start_type}') &
                   impacts_df['landuse'].str.contains(landuse))
            data = impacts_df[mask]['probchg'].dropna()
            if len(data) > 0:
                plot_data.append(data)
                labels.append(impact.capitalize())

        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_ylabel('Change in Probability')
            ax.set_title(f'Climate Impact on {landuse.capitalize()} Conversion')
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels(labels, rotation=45)

    plt.suptitle(f'Climate Change Impacts - Land Starting in {start_type.capitalize()}', fontsize=14)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def create_impact_histograms(impacts_df: pd.DataFrame,
                            start_type: str,
                            output_file: Optional[str] = None) -> plt.Figure:
    """
    Create histograms showing distribution of climate impacts.

    Parameters:
    -----------
    impacts_df : pd.DataFrame
        Impact data
    start_type : str
        Starting land use type
    output_file : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    landuses = ['crop', 'forest', 'urban', 'pasture']

    for i, landuse in enumerate(landuses):
        ax = axes[i]

        mask = (impacts_df['impact_type'] == f'full_impact_start{start_type}') & \
               (impacts_df['landuse'] == f'probchg_{landuse}')

        data = impacts_df[mask]['probchg'].dropna()

        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel(f'Change in Probability\nof {"Remaining in" if landuse == start_type else "Converting to"} {landuse.capitalize()}')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'Distribution of Climate Impacts - Land Starting in {start_type.capitalize()}', fontsize=14)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def create_impact_maps(impacts_df: pd.DataFrame,
                      counties_gdf: gpd.GeoDataFrame,
                      start_type: str,
                      output_file: Optional[str] = None) -> plt.Figure:
    """
    Create maps showing spatial distribution of climate impacts.

    Parameters:
    -----------
    impacts_df : pd.DataFrame
        Impact data aggregated to county level
    counties_gdf : gpd.GeoDataFrame
        County geometries
    start_type : str
        Starting land use type
    output_file : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Aggregate to county level
    county_impacts = impacts_df.groupby(['fips', 'landuse', 'impact_type']).apply(
        lambda x: pd.Series({
            'probchg': np.average(x['probchg'], weights=x['xfact'])
        })
    ).reset_index()

    county_impacts['fips'] = county_impacts['fips'].astype(str).str.zfill(5)

    # Merge with geometries
    map_data = counties_gdf.merge(county_impacts, left_on='GEOID', right_on='fips', how='right')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    landuses = ['crop', 'forest', 'urban', 'pasture']
    titles = {
        'crop': 'Remaining in' if start_type == 'crop' else 'Converting to',
        'forest': 'Remaining in' if start_type == 'forest' else 'Converting to',
        'urban': 'Converting to',
        'pasture': 'Converting to'
    }

    for i, landuse in enumerate(landuses):
        ax = axes[i]

        plot_data = map_data[
            (map_data['impact_type'] == f'full_impact_start{start_type}') &
            (map_data['landuse'] == f'probchg_{landuse}')
        ]

        if not plot_data.empty:
            plot_data.plot(column='probchg', ax=ax, legend=True,
                          cmap='RdYlGn', edgecolor='black', linewidth=0.5)
            ax.set_title(f'Change in Probability of {titles[landuse]} {landuse.capitalize()} Use')
            ax.axis('off')

    plt.suptitle(f'Spatial Distribution of Climate Impacts - Land Starting in {start_type.capitalize()}', fontsize=14)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def main(model_data_path: str,
         estimation_data_path: str,
         cc_impacts_dir: str,
         shapefile_path: str,
         output_dir: str):
    """
    Main function to run climate impact analysis.

    Parameters:
    -----------
    model_data_path : str
        Path to model coefficients data
    estimation_data_path : str
        Path to estimation data
    cc_impacts_dir : str
        Directory with climate impact data
    shapefile_path : str
        Path to county shapefile
    output_dir : str
        Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading model data...")
    landuse_models = pd.read_pickle(model_data_path)
    model_crstart = landuse_models[2]  # Adjust index as needed
    model_frstart = landuse_models[10]  # Adjust index as needed

    print("Loading estimation data...")
    est_dat = pd.read_pickle(estimation_data_path)
    dat_crstart = est_dat['cropstart_east']
    dat_frstart = est_dat['foreststart_east']

    print("Loading climate impact data...")
    cc_impact = load_climate_impacts(cc_impacts_dir)

    print("Loading county shapefile...")
    counties = gpd.read_file(shapefile_path)

    # Calculate impacts for crop start
    print("Calculating impacts for land starting in crop...")
    crop_impacts = calculate_scenario_impacts(
        dat_crstart, cc_impact, model_crstart.params.values, 'cr'
    )

    # Calculate impacts for forest start
    print("Calculating impacts for land starting in forest...")
    forest_impacts = calculate_scenario_impacts(
        dat_frstart, cc_impact, model_frstart.params.values, 'fr'
    )

    # Combine all impacts
    all_crop_impacts = pd.concat(crop_impacts.values())
    all_crop_impacts['type_landuse'] = all_crop_impacts['impact_type'] + '_' + all_crop_impacts['landuse']

    all_forest_impacts = pd.concat(forest_impacts.values())
    all_forest_impacts['type_landuse'] = all_forest_impacts['impact_type'] + '_' + all_forest_impacts['landuse']

    # Create visualizations
    print("Creating visualizations...")

    # Boxplots
    create_impact_boxplots(all_crop_impacts, 'cr',
                          output_path / 'cc_impact_crstart_boxplot.pdf')
    create_impact_boxplots(all_forest_impacts, 'fr',
                          output_path / 'cc_impact_frstart_boxplot.pdf')

    # Histograms
    create_impact_histograms(all_crop_impacts, 'cr',
                           output_path / 'cc_impact_crstart_histogram.pdf')
    create_impact_histograms(all_forest_impacts, 'fr',
                           output_path / 'cc_impact_frstart_histogram.pdf')

    # Maps
    create_impact_maps(all_crop_impacts, counties, 'cr',
                      output_path / 'cc_impact_map_crstart.pdf')
    create_impact_maps(all_forest_impacts, counties, 'fr',
                      output_path / 'cc_impact_map_frstart.pdf')

    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print("Usage: python climate_impact.py <model_data> <estimation_data> <cc_impacts_dir> <shapefile> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])