"""
Discrete choice logit model estimation for land use transitions.
Ported from estimate_logit_crop_forest.R
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from pathlib import Path


def prepare_estimation_data(start_data: pd.DataFrame,
                           nr_data: pd.DataFrame,
                           georef: pd.DataFrame,
                           years: List[int],
                           regions: Optional[List[str]] = None,
                           use_net_returns: bool = False) -> pd.DataFrame:
    """
    Prepare data for logit estimation.

    Parameters:
    -----------
    start_data : pd.DataFrame
        Starting land use data
    nr_data : pd.DataFrame
        Net returns data (optional, only used if use_net_returns=True)
    georef : pd.DataFrame
        Geographic reference data with regions
    years : List[int]
        Years to include in estimation
    regions : List[str], optional
        Regions to filter
    use_net_returns : bool, optional
        Whether to include net returns in the model (default: False)

    Returns:
    --------
    pd.DataFrame
        Prepared estimation data
    """
    data = start_data.copy()

    # Only merge net returns if explicitly requested
    if use_net_returns and nr_data is not None:
        # Apply lag to net returns
        nr_data_lagged = nr_data.copy()
        nr_data_lagged['year'] = nr_data_lagged['year'] + 2
        # Merge data
        data = data.merge(nr_data_lagged, on=['fips', 'year'], how='left')

    # Handle georef columns - it may have both 'fips' and 'county_fips'
    # Drop county_fips if both exist to avoid confusion
    if 'fips' in georef.columns and 'county_fips' in georef.columns:
        georef = georef.drop(columns=['county_fips'])
    elif 'county_fips' in georef.columns and 'fips' not in georef.columns:
        georef = georef.rename(columns={'county_fips': 'fips'})

    # Select only needed columns to avoid any duplicates
    georef_cols = ['fips', 'subregion', 'region']
    available_cols = [col for col in georef_cols if col in georef.columns]
    georef_subset = georef[available_cols].drop_duplicates(subset=['fips'])
    data = data.merge(georef_subset, on='fips')

    # Clean up columns
    cols_to_drop = [col for col in data.columns if col.endswith('ag')]
    data = data.drop(columns=cols_to_drop, errors='ignore')

    # Convert land capability class to integer
    data['lcc'] = pd.to_numeric(data['lcc'], errors='coerce')

    # Create categorical variables for LCC (for better model fit)
    # LCC 1-2 are best for agriculture, 3-4 moderate, 5-8 poor
    data['lcc_cat'] = pd.cut(data['lcc'],
                             bins=[0, 2, 4, 8],
                             labels=['good', 'moderate', 'poor'])

    # Filter by years if specified
    if years:
        data = data[data['year'].isin(years)]

    # Filter by regions if specified
    if regions:
        data = data[data['region'].isin(regions)]

    # Calculate weights
    if 'xfact' in data.columns:
        data['weight'] = (data['xfact'] / data['xfact'].sum()) * len(data)

    return data


def estimate_mnlogit(data: pd.DataFrame,
                    formula: str,
                    ref_category: Optional[int] = None) -> Any:
    """
    Estimate multinomial logit model.

    Parameters:
    -----------
    data : pd.DataFrame
        Estimation data
    formula : str
        Model formula
    ref_category : int, optional
        Reference category for MNLogit

    Returns:
    --------
    MultinomialResultsWrapper
        Fitted model results
    """
    # Create design matrix
    from patsy import dmatrices
    y, X = dmatrices(formula, data, return_type='dataframe')

    # Fit model
    model = MNLogit(y, X)
    result = model.fit(method='bfgs', maxiter=1000, disp=False)

    return result


def estimate_land_use_transitions(crop_start: pd.DataFrame,
                                 pasture_start: pd.DataFrame,
                                 forest_start: pd.DataFrame,
                                 nr_data: pd.DataFrame,
                                 georef: pd.DataFrame,
                                 years: List[int] = None,
                                 use_net_returns: bool = False) -> Dict:
    """
    Estimate land use transition models for different starting conditions.

    Parameters:
    -----------
    crop_start : pd.DataFrame
        Data for land starting in crop use
    pasture_start : pd.DataFrame
        Data for land starting in pasture use
    forest_start : pd.DataFrame
        Data for land starting in forest use
    nr_data : pd.DataFrame
        Net returns data (optional, only used if use_net_returns=True)
    georef : pd.DataFrame
        Geographic reference data
    years : List[int]
        Years to include
    use_net_returns : bool, optional
        Whether to include net returns in the model (default: False, uses only LCC)

    Returns:
    --------
    Dict
        Dictionary of estimated models
    """
    models = {}

    # Prepare data for each starting condition and region
    regions = {
        'all': None,
        'east': ['NO', 'SO'],
        'south': ['SO'],
        'north': ['NO']
    }

    starting_conditions = {
        'crop': crop_start,
        'pasture': pasture_start,
        'forest': forest_start
    }

    for start_name, start_data in starting_conditions.items():
        for region_name, region_filter in regions.items():
            print(f"Estimating LCC-only model for {start_name} start in {region_name} region...")

            # Prepare data
            est_data = prepare_estimation_data(
                start_data, nr_data, georef, years, region_filter, use_net_returns
            )

            # Store prepared data
            models[f'{start_name}start_{region_name}_data'] = est_data

            # Define formula based on whether to use net returns
            if use_net_returns:
                # Check if net returns columns are available
                nr_cols = ['nr_cr', 'nr_ps', 'nr_fr', 'nr_ur']
                available_nr_cols = [col for col in nr_cols if col in est_data.columns]
                if available_nr_cols:
                    formula = f'enduse ~ lcc + {" + ".join(available_nr_cols)}'
                else:
                    print(f"  Warning: Net returns requested but not available, using LCC only")
                    formula = 'enduse ~ lcc'
            else:
                # Simplified model using only LCC
                formula = 'enduse ~ lcc'

            print(f"  Using formula: {formula}")

            try:
                # Estimate model
                model = estimate_mnlogit(est_data, formula)
                models[f'{start_name}start_{region_name}'] = model
                print(f"  Model estimated successfully. Log-likelihood: {model.llf:.2f}")
            except Exception as e:
                print(f"  Failed to estimate model: {str(e)}")
                models[f'{start_name}start_{region_name}'] = None

    return models


def calculate_marginal_effects(model: Any,
                              data: pd.DataFrame,
                              variable: str) -> pd.DataFrame:
    """
    Calculate marginal effects for a variable.

    Parameters:
    -----------
    model : MultinomialResultsWrapper
        Fitted MNLogit model
    data : pd.DataFrame
        Data used for prediction
    variable : str
        Variable to calculate marginal effects for

    Returns:
    --------
    pd.DataFrame
        Marginal effects
    """
    # Get marginal effects at means
    margeff = model.get_margeff(at='mean')

    # Extract marginal effects for the specified variable
    me_summary = margeff.summary_frame()
    me_variable = me_summary[me_summary.index.str.contains(variable)]

    return me_variable


def save_estimation_results(models: Dict,
                           output_dir: str,
                           save_pickle: bool = True,
                           save_csv: bool = True):
    """
    Save estimation results to files.

    Parameters:
    -----------
    models : Dict
        Dictionary of estimated models
    output_dir : str
        Output directory path
    save_pickle : bool
        Save as pickle files
    save_csv : bool
        Save summary statistics as CSV
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_pickle:
        # Save models dictionary
        import pickle
        with open(output_path / 'landuse_models.pkl', 'wb') as f:
            pickle.dump(models, f)

        # Save estimation data
        estimation_data = {k: v for k, v in models.items() if k.endswith('_data')}
        with open(output_path / 'estimation_data.pkl', 'wb') as f:
            pickle.dump(estimation_data, f)

    if save_csv:
        # Save model summaries
        summaries = []
        for name, model in models.items():
            if not name.endswith('_data') and model is not None:
                try:
                    summary_df = pd.DataFrame({
                        'model': name,
                        'coefficient': model.params.index,
                        'estimate': model.params.values,
                        'std_error': model.bse.values if hasattr(model, 'bse') and model.bse is not None else [np.nan] * len(model.params),
                        't_value': model.tvalues.values if hasattr(model, 'tvalues') and model.tvalues is not None else [np.nan] * len(model.params),
                        'p_value': model.pvalues.values if hasattr(model, 'pvalues') and model.pvalues is not None else [np.nan] * len(model.params)
                    })
                    summaries.append(summary_df)
                except Exception as e:
                    print(f"Warning: Could not save summary for {name}: {str(e)}")

        if summaries:
            all_summaries = pd.concat(summaries, ignore_index=True)
            all_summaries.to_csv(output_path / 'model_summaries.csv', index=False)


def load_data(georef_file: str,
             nr_data_file: str,
             start_crop_file: str,
             start_pasture_file: str,
             start_forest_file: str) -> Tuple[pd.DataFrame, ...]:
    """
    Load data files for estimation.

    Parameters:
    -----------
    georef_file : str
        Path to geographic reference file
    nr_data_file : str
        Path to net returns data file
    start_crop_file : str
        Path to crop start data file
    start_pasture_file : str
        Path to pasture start data file
    start_forest_file : str
        Path to forest start data file

    Returns:
    --------
    Tuple[pd.DataFrame, ...]
        Loaded dataframes
    """
    # Load geographic reference
    georef = pd.read_csv(georef_file)
    # Only rename county_fips to fips if fips doesn't already exist
    if 'fips' not in georef.columns and 'county_fips' in georef.columns:
        georef = georef.rename(columns={'county_fips': 'fips'})
    if 'fips' in georef.columns:
        georef['fips'] = georef['fips'].astype(int)

    # Load net returns data
    if nr_data_file.endswith('.rds'):
        nr_data = pd.read_pickle(nr_data_file)
    else:
        nr_data = pd.read_csv(nr_data_file)

    # Load starting condition data
    if start_crop_file.endswith('.rds'):
        start_crop = pd.read_pickle(start_crop_file)
    else:
        start_crop = pd.read_csv(start_crop_file)

    if start_pasture_file.endswith('.rds'):
        start_pasture = pd.read_pickle(start_pasture_file)
    else:
        start_pasture = pd.read_csv(start_pasture_file)

    if start_forest_file.endswith('.rds'):
        start_forest = pd.read_pickle(start_forest_file)
    else:
        start_forest = pd.read_csv(start_forest_file)

    return georef, nr_data, start_crop, start_pasture, start_forest


def main(georef_file: str,
         nr_data_file: str,
         start_crop_file: str,
         start_pasture_file: str,
         start_forest_file: str,
         output_dir: str,
         use_net_returns: bool = False):
    """
    Main function to run logit estimation.

    Parameters:
    -----------
    georef_file : str
        Path to geographic reference file
    nr_data_file : str
        Path to net returns data (optional, only used if use_net_returns=True)
    start_crop_file : str
        Path to crop start data
    start_pasture_file : str
        Path to pasture start data
    start_forest_file : str
        Path to forest start data
    output_dir : str
        Output directory
    use_net_returns : bool, optional
        Whether to include net returns in the model (default: False, uses only LCC)
    """
    print("Loading data files...")
    georef, nr_data, start_crop, start_pasture, start_forest = load_data(
        georef_file, nr_data_file, start_crop_file, start_pasture_file, start_forest_file
    )

    model_type = "LCC-only" if not use_net_returns else "LCC with net returns"
    print(f"Estimating land use transition models ({model_type})...")
    models = estimate_land_use_transitions(
        start_crop, start_pasture, start_forest, nr_data, georef,
        use_net_returns=use_net_returns
    )

    print("Saving results...")
    save_estimation_results(models, output_dir)

    print(f"Estimation complete. Results saved to {output_dir}")

    # Print summary of models
    print(f"\nModel Summary ({model_type}):")
    for name, model in models.items():
        if not name.endswith('_data') and model is not None:
            print(f"\n{name}:")
            print(f"  Log-likelihood: {model.llf:.2f}")
            print(f"  AIC: {model.aic:.2f}")
            print(f"  Number of observations: {model.nobs:.0f}")


# =============================================================================
# 4-Category Model (Crop-Pasture-Urban with Irrigation Split)
# =============================================================================

def prepare_estimation_data_4cat(
    start_data: pd.DataFrame,
    nr_data: pd.DataFrame,
    georef: pd.DataFrame,
    years: List[int],
    regions: Optional[List[str]] = None,
    use_net_returns: bool = True,
    scale_net_returns: bool = True
) -> pd.DataFrame:
    """
    Prepare data for 4-category logit estimation.

    This version uses the 4-category net returns columns:
    nr_cr_irr, nr_cr_dry, nr_ps, nr_ur

    Parameters:
    -----------
    start_data : pd.DataFrame
        Starting land use data
    nr_data : pd.DataFrame
        Net returns data with 4-category columns (already lagged)
    georef : pd.DataFrame
        Geographic reference data with regions
    years : List[int]
        Years to include in estimation
    regions : List[str], optional
        Regions to filter
    use_net_returns : bool, optional
        Whether to include net returns in the model (default: True)
    scale_net_returns : bool, optional
        Whether to standardize net returns for numerical stability (default: True)

    Returns:
    --------
    pd.DataFrame
        Prepared estimation data
    """
    data = start_data.copy()

    # Merge net returns if requested (net returns should already be lagged)
    if use_net_returns and nr_data is not None:
        data = data.merge(nr_data, on=['fips', 'year'], how='left')

        # Scale net returns for numerical stability
        # Divide by scaling factor to bring values to more manageable range
        # This doesn't affect interpretation - just helps convergence
        # Urban returns are much larger, so scale more aggressively
        if scale_net_returns:
            ag_nr_cols = ['nr_cr_irr', 'nr_cr_dry', 'nr_ps']
            for col in ag_nr_cols:
                if col in data.columns:
                    data[col] = data[col] / 100.0  # Scale ag to hundreds of dollars
            if 'nr_ur' in data.columns:
                data['nr_ur'] = data['nr_ur'] / 1000.0  # Scale urban to thousands of dollars

    # Handle georef columns
    if 'fips' in georef.columns and 'county_fips' in georef.columns:
        georef = georef.drop(columns=['county_fips'])
    elif 'county_fips' in georef.columns and 'fips' not in georef.columns:
        georef = georef.rename(columns={'county_fips': 'fips'})

    # Select only needed columns to avoid duplicates
    georef_cols = ['fips', 'subregion', 'region']
    available_cols = [col for col in georef_cols if col in georef.columns]
    georef_subset = georef[available_cols].drop_duplicates(subset=['fips'])
    data = data.merge(georef_subset, on='fips')

    # Convert land capability class to integer
    data['lcc'] = pd.to_numeric(data['lcc'], errors='coerce')

    # Create categorical LCC
    data['lcc_cat'] = pd.cut(
        data['lcc'],
        bins=[0, 2, 4, 8],
        labels=['good', 'moderate', 'poor']
    )

    # Filter by years
    if years:
        data = data[data['year'].isin(years)]

    # Filter by regions
    if regions:
        data = data[data['region'].isin(regions)]

    # Calculate weights
    if 'xfact' in data.columns:
        data['weight'] = (data['xfact'] / data['xfact'].sum()) * len(data)

    return data


# =============================================================================
# RPA Subregions for Regional Models
# =============================================================================

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


def estimate_land_use_transitions_4cat(
    cr_irr_start: pd.DataFrame,
    cr_dry_start: pd.DataFrame,
    pasture_start: pd.DataFrame,
    urban_start: pd.DataFrame,
    nr_data: pd.DataFrame,
    georef: pd.DataFrame,
    years: List[int] = None,
    use_net_returns: bool = True,
    min_observations: int = 100,
    subregions: Optional[List[str]] = None
) -> Dict:
    """
    Estimate 4-category land use transition models by RPA subregion.

    Models transitions between:
    - Irrigated cropland (CR_IRR, code 1)
    - Non-irrigated cropland (CR_DRY, code 2)
    - Pasture (PS, code 3)
    - Urban (UR, code 4)

    Estimates separate models for each RPA subregion:
    NE (Northeast), LS (Lake States), CB (Corn Belt), NP (Northern Plains),
    AP (Appalachian), SE (Southeast), DL (Delta), SP (Southern Plains),
    MT (Mountain), PC (Pacific Coast)

    Parameters:
    -----------
    cr_irr_start : pd.DataFrame
        Data for land starting in irrigated cropland
    cr_dry_start : pd.DataFrame
        Data for land starting in non-irrigated cropland
    pasture_start : pd.DataFrame
        Data for land starting in pasture
    urban_start : pd.DataFrame
        Data for land starting in urban
    nr_data : pd.DataFrame
        Net returns data with columns: nr_cr_irr, nr_cr_dry, nr_ps, nr_ur
    georef : pd.DataFrame
        Geographic reference data with 'region' column containing subregion codes
    years : List[int]
        Years to include
    use_net_returns : bool
        Whether to include net returns in model (default: True)
    min_observations : int
        Minimum observations required to estimate a model (default: 100)
    subregions : List[str], optional
        Specific subregions to estimate (e.g., ['SE', 'DL']). If None, estimates all.

    Returns:
    --------
    Dict
        Dictionary of estimated models and prepared data, keyed by
        '{startuse}_{subregion}' (e.g., 'cr_irr_CB' for irrigated crop in Corn Belt)
    """
    models = {}

    # Starting conditions
    starting_conditions = {
        'cr_irr': cr_irr_start,
        'cr_dry': cr_dry_start,
        'pasture': pasture_start,
        'urban': urban_start
    }

    # 4-category net returns columns
    nr_cols_4cat = ['nr_cr_irr', 'nr_cr_dry', 'nr_ps', 'nr_ur']

    # Get available subregions from the data
    all_data = pd.concat([df for df in starting_conditions.values() if df is not None and len(df) > 0])
    if 'region' not in georef.columns:
        print("Error: georef must have 'region' column with subregion codes")
        return models

    # Merge to get regions
    all_data_with_region = all_data.merge(
        georef[['fips', 'region']].drop_duplicates(),
        on='fips',
        how='left'
    )
    available_subregions = all_data_with_region['region'].dropna().unique()

    # Filter to specific subregions if requested
    if subregions:
        available_subregions = [s for s in available_subregions if s in subregions]
        if not available_subregions:
            print(f"Warning: No data found for requested subregions: {subregions}")
            return models

    print(f"\n{'='*60}")
    print("Estimating Regional Models by RPA Subregion")
    print(f"{'='*60}")
    print(f"Available subregions: {sorted(available_subregions)}")
    print(f"Starting conditions: {list(starting_conditions.keys())}")
    print(f"{'='*60}\n")

    for subregion in sorted(available_subregions):
        subregion_name = RPA_SUBREGIONS.get(subregion, subregion)
        print(f"\n[{subregion}] {subregion_name}")
        print("-" * 40)

        for start_name, start_data in starting_conditions.items():
            if start_data is None or len(start_data) == 0:
                continue

            model_key = f'{start_name}_{subregion}'

            # Prepare data for this subregion
            est_data = prepare_estimation_data_4cat(
                start_data, nr_data, georef, years,
                regions=[subregion],
                use_net_returns=use_net_returns
            )

            if len(est_data) < min_observations:
                print(f"  {start_name}: skipped (n={len(est_data)} < {min_observations})")
                continue

            # Store prepared data
            models[f'{model_key}_data'] = est_data

            # Build formula
            if use_net_returns:
                available_nr_cols = [col for col in nr_cols_4cat if col in est_data.columns]
                if available_nr_cols:
                    formula = f'enduse ~ lcc + {" + ".join(available_nr_cols)}'
                else:
                    formula = 'enduse ~ lcc'
            else:
                formula = 'enduse ~ lcc'

            try:
                model = estimate_mnlogit(est_data, formula)
                models[model_key] = model
                print(f"  {start_name}: n={len(est_data):,}, LL={model.llf:.1f}")
            except Exception as e:
                print(f"  {start_name}: FAILED - {str(e)[:50]}")
                models[model_key] = None

    # Summary
    successful = [k for k in models.keys() if not k.endswith('_data') and models[k] is not None]
    print(f"\n{'='*60}")
    print(f"Successfully estimated: {len(successful)} models")
    print(f"{'='*60}")

    return models


def load_data_4cat(
    georef_file: str,
    nr_data_file: str,
    start_cr_irr_file: str,
    start_cr_dry_file: str,
    start_pasture_file: str,
    start_urban_file: str
) -> Tuple[pd.DataFrame, ...]:
    """
    Load data files for 4-category estimation.

    Parameters:
    -----------
    georef_file : str
        Path to geographic reference file
    nr_data_file : str
        Path to net returns data file (4-category format)
    start_cr_irr_file : str
        Path to irrigated crop start data
    start_cr_dry_file : str
        Path to non-irrigated crop start data
    start_pasture_file : str
        Path to pasture start data
    start_urban_file : str
        Path to urban start data

    Returns:
    --------
    Tuple[pd.DataFrame, ...]
        Loaded dataframes: georef, nr_data, cr_irr, cr_dry, pasture, urban
    """
    # Load geographic reference
    georef = pd.read_csv(georef_file)
    if 'fips' not in georef.columns and 'county_fips' in georef.columns:
        georef = georef.rename(columns={'county_fips': 'fips'})
    if 'fips' in georef.columns:
        georef['fips'] = georef['fips'].astype(int)

    # Load net returns
    nr_data = pd.read_csv(nr_data_file)

    # Load starting condition data
    def load_if_exists(filepath):
        path = Path(filepath)
        if path.exists():
            return pd.read_csv(filepath)
        else:
            print(f"Warning: {filepath} not found")
            return pd.DataFrame()

    cr_irr = load_if_exists(start_cr_irr_file)
    cr_dry = load_if_exists(start_cr_dry_file)
    pasture = load_if_exists(start_pasture_file)
    urban = load_if_exists(start_urban_file)

    return georef, nr_data, cr_irr, cr_dry, pasture, urban


def main_4cat(
    georef_file: str,
    nr_data_file: str,
    start_cr_irr_file: str,
    start_cr_dry_file: str,
    start_pasture_file: str,
    start_urban_file: str,
    output_dir: str,
    use_net_returns: bool = True,
    years: List[int] = None,
    subregions: List[str] = None
):
    """
    Main function to run 4-category logit estimation.

    Parameters:
    -----------
    georef_file : str
        Path to geographic reference file
    nr_data_file : str
        Path to net returns data (4-category format)
    start_cr_irr_file : str
        Path to irrigated crop start data
    start_cr_dry_file : str
        Path to non-irrigated crop start data
    start_pasture_file : str
        Path to pasture start data
    start_urban_file : str
        Path to urban start data
    output_dir : str
        Output directory
    use_net_returns : bool
        Whether to include net returns
    years : List[int]
        Years to include
    subregions : List[str], optional
        Specific subregions to estimate (e.g., ['SE', 'DL']). If None, estimates all.
    """
    print("Loading 4-category data files...")
    georef, nr_data, cr_irr, cr_dry, pasture, urban = load_data_4cat(
        georef_file, nr_data_file,
        start_cr_irr_file, start_cr_dry_file,
        start_pasture_file, start_urban_file
    )

    # Report data summary
    print("\nData Summary:")
    print(f"  Geographic reference: {len(georef):,} counties")
    print(f"  Net returns: {len(nr_data):,} county-years")
    print(f"  CR_IRR starts: {len(cr_irr):,}")
    print(f"  CR_DRY starts: {len(cr_dry):,}")
    print(f"  Pasture starts: {len(pasture):,}")
    print(f"  Urban starts: {len(urban):,}")

    model_type = "with net returns" if use_net_returns else "LCC-only"
    print(f"\nEstimating 4-category models ({model_type})...")

    models = estimate_land_use_transitions_4cat(
        cr_irr, cr_dry, pasture, urban,
        nr_data, georef,
        years=years,
        use_net_returns=use_net_returns,
        subregions=subregions
    )

    print("\nSaving results...")
    save_estimation_results(models, output_dir)

    print(f"\nEstimation complete. Results saved to {output_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print("Model Summary:")
    print('='*60)
    successful_models = [k for k in models.keys() if not k.endswith('_data') and models[k] is not None]
    print(f"Successfully estimated: {len(successful_models)} models")

    for name in sorted(successful_models):
        model = models[name]
        print(f"\n{name}:")
        print(f"  Log-likelihood: {model.llf:.2f}")
        print(f"  AIC: {model.aic:.2f}")
        print(f"  Observations: {model.nobs:.0f}")


# =============================================================================
# 6-Category Model Functions (with Range and Forest)
# =============================================================================

# 6-category land use names
LANDUSE_NAMES_6CAT = {
    1: 'CR_IRR',   # Irrigated cropland
    2: 'CR_DRY',   # Non-irrigated cropland
    3: 'PS',       # Pasture
    4: 'RG',       # Rangeland
    5: 'FR',       # Forest
    6: 'UR',       # Urban
}


def prepare_estimation_data_6cat(
    start_data: pd.DataFrame,
    nr_data: pd.DataFrame,
    georef: pd.DataFrame,
    years: List[int],
    regions: Optional[List[str]] = None,
    use_net_returns: bool = True,
    scale_net_returns: bool = True
) -> pd.DataFrame:
    """
    Prepare data for 6-category logit estimation.

    In the 6-category model:
    - Ag uses (CR_IRR, CR_DRY, PS) and Urban have net returns
    - Range (RG) and Forest (FR) are driven by LCC only

    The model includes net returns for ag/urban, which affects the relative
    probability of choosing each end use. Range and forest don't have their
    own returns but are influenced by the opportunity cost (returns of other uses).
    """
    data = start_data.copy()

    # Only merge net returns if explicitly requested
    if use_net_returns and nr_data is not None:
        # Apply lag to net returns (already lagged in data, but double-check)
        nr_data_copy = nr_data.copy()

        # Merge data
        data = data.merge(nr_data_copy, on=['fips', 'year'], how='left')

        # Scale net returns for numerical stability
        # Urban returns are much larger, so scale more aggressively
        if scale_net_returns:
            ag_nr_cols = ['nr_cr_irr', 'nr_cr_dry', 'nr_ps']
            for col in ag_nr_cols:
                if col in data.columns:
                    data[col] = data[col] / 100.0  # Scale ag to hundreds of dollars
            if 'nr_ur' in data.columns:
                data['nr_ur'] = data['nr_ur'] / 1000.0  # Scale urban to thousands of dollars

    # Handle georef columns
    if 'fips' in georef.columns and 'county_fips' in georef.columns:
        georef = georef.drop(columns=['county_fips'])
    elif 'county_fips' in georef.columns and 'fips' not in georef.columns:
        georef = georef.rename(columns={'county_fips': 'fips'})

    # Select only needed columns
    georef_cols = ['fips', 'subregion', 'region']
    available_cols = [col for col in georef_cols if col in georef.columns]
    georef_subset = georef[available_cols].drop_duplicates(subset=['fips'])
    data = data.merge(georef_subset, on='fips')

    # Convert land capability class to integer
    data['lcc'] = pd.to_numeric(data['lcc'], errors='coerce')

    # Filter by years if specified
    if years:
        data = data[data['year'].isin(years)]

    # Filter by regions if specified
    if regions:
        data = data[data['region'].isin(regions)]

    # Drop rows with missing values in key columns
    required_cols = ['enduse', 'lcc']
    if use_net_returns:
        nr_cols = ['nr_cr_irr', 'nr_cr_dry', 'nr_ps', 'nr_ur']
        required_cols.extend([c for c in nr_cols if c in data.columns])

    data = data.dropna(subset=required_cols)

    # Calculate weights
    if 'xfact' in data.columns:
        data['weight'] = (data['xfact'] / data['xfact'].sum()) * len(data)

    return data


def get_county_feasible_uses(all_transitions: pd.DataFrame) -> Dict[int, set]:
    """
    Identify which land uses are feasible (observed) in each county.

    A land use is feasible in a county if it appears as either a start or end use
    for any observation in that county. This restricts the choice set to only
    alternatives that actually exist in each county.

    Parameters:
    -----------
    all_transitions : pd.DataFrame
        Combined transition data with 'fips', 'startuse', 'enduse' columns

    Returns:
    --------
    Dict[int, set]
        Dictionary mapping county FIPS to set of feasible land use codes
    """
    county_uses = {}

    for fips in all_transitions['fips'].unique():
        county_data = all_transitions[all_transitions['fips'] == fips]
        # Land use is feasible if observed as either start or end
        start_uses = set(county_data['startuse'].unique())
        end_uses = set(county_data['enduse'].unique())
        county_uses[fips] = start_uses | end_uses

    return county_uses


def summarize_nr_coefficients(model, model_name: str) -> str:
    """
    Return summary of net returns coefficients with sign validation.

    Economic theory: own-use net returns should have POSITIVE coefficients
    (higher returns → higher probability of choosing that use)

    MNLogit params structure: DataFrame with
    - Index = variable names (Intercept, lcc, nr_cr_irr, etc.)
    - Columns = alternative codes (0, 1, 2, 3... relative to base)

    Returns:
        String with coefficient summary showing sign pattern, e.g., "[nr_cr_irr:---, nr_ps:+-+]"
    """
    if model is None or not hasattr(model, 'params'):
        return ""

    try:
        params = model.params
        nr_vars = ['nr_cr_irr', 'nr_cr_dry', 'nr_ps', 'nr_ur']

        # Check if params is a DataFrame (MNLogit) or Series
        if isinstance(params, pd.DataFrame):
            nr_summary = []
            for nr_var in nr_vars:
                if nr_var in params.index:
                    # Get coefficient row for this variable across all alternatives
                    coefs = params.loc[nr_var]
                    # Count positive/negative coefficients
                    pos_count = (coefs > 0).sum()
                    neg_count = (coefs < 0).sum()
                    total = len(coefs)

                    if neg_count == total:
                        nr_summary.append(f"{nr_var}:all-")
                    elif pos_count == total:
                        nr_summary.append(f"{nr_var}:all+")
                    else:
                        # Show pattern: + or - for each alternative
                        signs = ''.join(['+' if c > 0 else '-' for c in coefs])
                        nr_summary.append(f"{nr_var}:{signs}")

            if nr_summary:
                return f" [{', '.join(nr_summary)}]"
        else:
            # Handle Series case (shouldn't happen but just in case)
            nr_summary = []
            for idx in params.index:
                idx_str = str(idx).lower()
                for nr_var in nr_vars:
                    if nr_var in idx_str:
                        sign = '+' if params[idx] > 0 else '-'
                        nr_summary.append(f"{nr_var}:{sign}")
                        break
            if nr_summary:
                return f" [{', '.join(set(nr_summary))}]"

        return ""
    except Exception:
        return ""


def filter_to_feasible_choices(
    data: pd.DataFrame,
    county_feasible_uses: Dict[int, set]
) -> pd.DataFrame:
    """
    Filter data to only include observations where the end use is feasible.

    Parameters:
    -----------
    data : pd.DataFrame
        Estimation data with 'fips' and 'enduse' columns
    county_feasible_uses : Dict[int, set]
        Dictionary mapping county FIPS to set of feasible land use codes

    Returns:
    --------
    pd.DataFrame
        Filtered data where end use is feasible in each county
    """
    def is_feasible(row):
        fips = row['fips']
        enduse = row['enduse']
        if fips in county_feasible_uses:
            return enduse in county_feasible_uses[fips]
        return True  # If county not found, include all

    mask = data.apply(is_feasible, axis=1)
    return data[mask].copy()


def estimate_land_use_transitions_6cat(
    cr_irr_start: pd.DataFrame,
    cr_dry_start: pd.DataFrame,
    pasture_start: pd.DataFrame,
    range_start: pd.DataFrame,
    forest_start: pd.DataFrame,
    urban_start: pd.DataFrame,
    nr_data: pd.DataFrame,
    georef: pd.DataFrame,
    years: List[int] = None,
    use_net_returns: bool = True,
    min_observations: int = 100,
    subregions: Optional[List[str]] = None,
    restrict_choice_set: bool = True,
    region_specs: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict:
    """
    Estimate 6-category land use transition models by RPA subregion.

    Models transitions FROM 5 starting uses (urban is excluded as irreversible):
    - Irrigated cropland (CR_IRR, code 1)
    - Non-irrigated cropland (CR_DRY, code 2)
    - Pasture (PS, code 3)
    - Rangeland (RG, code 4)
    - Forest (FR, code 5)

    Note: No model is estimated for land starting in urban use. We assume urban
    development is irreversible - once land is converted to urban use, it never
    reverts to agricultural or natural land uses.

    Specifications are region-specific based on data variation and economic validity.
    Default specification:
    - Ag starts (cr_irr, cr_dry, pasture): LCC + nr_ur (urban destination returns)
    - Range/Forest: LCC only

    Parameters:
    -----------
    cr_irr_start, cr_dry_start, pasture_start, range_start, forest_start, urban_start
        DataFrames for each starting land use (urban_start is ignored)
    nr_data : pd.DataFrame
        Net returns data with columns: nr_cr_irr, nr_cr_dry, nr_ps, nr_ur
    georef : pd.DataFrame
        Geographic reference data with 'region' column
    years : List[int]
        Years to include
    use_net_returns : bool
        Include net returns for ag uses (default: True)
    min_observations : int
        Minimum observations required (default: 100)
    subregions : List[str], optional
        Specific subregions to estimate
    restrict_choice_set : bool
        If True, only include end uses that exist in each county (default: True)
    region_specs : Dict[str, Dict[str, str]], optional
        Region-specific model specifications. Format:
        {'SE': {'cr_irr': 'lcc + nr_ur', 'pasture': 'lcc', ...}, ...}
        If not provided, uses default specification.

    Returns:
    --------
    Dict
        Dictionary of estimated models keyed by '{startuse}_{subregion}'
    """
    models = {}

    # Starting conditions - 5 categories (urban excluded as irreversible)
    # Urban start is not modeled because urban development is assumed irreversible
    starting_conditions = {
        'cr_irr': cr_irr_start,
        'cr_dry': cr_dry_start,
        'pasture': pasture_start,
        'range': range_start,
        'forest': forest_start,
        # 'urban' excluded - urban development is irreversible
    }

    # Uses that include urban net returns in their specification
    # Range and forest use LCC only (no rent data available)
    ag_starts_with_nr = ['cr_irr', 'cr_dry', 'pasture']

    # Get available subregions
    all_data = pd.concat([df for df in starting_conditions.values() if df is not None and len(df) > 0])
    if 'region' not in georef.columns:
        print("Error: georef must have 'region' column with subregion codes")
        return models

    all_data_with_region = all_data.merge(
        georef[['fips', 'region']].drop_duplicates(),
        on='fips',
        how='left'
    )
    available_subregions = all_data_with_region['region'].dropna().unique()

    # Filter to specific subregions if requested
    if subregions:
        available_subregions = [s for s in available_subregions if s in subregions]
        if not available_subregions:
            print(f"Warning: No data found for requested subregions: {subregions}")
            return models

    # Compute county-level feasible land uses for choice set restriction
    county_feasible_uses = None
    if restrict_choice_set:
        print("Computing county-level feasible choice sets...")
        county_feasible_uses = get_county_feasible_uses(all_data)
        # Count counties with each number of feasible uses
        use_counts = {}
        for fips, uses in county_feasible_uses.items():
            n = len(uses)
            use_counts[n] = use_counts.get(n, 0) + 1
        print(f"  Counties by # feasible uses: {dict(sorted(use_counts.items()))}")

    print(f"\n{'='*60}")
    print("Estimating 5-Category Regional Models by RPA Subregion")
    print(f"{'='*60}")
    print(f"Starting uses: CR_IRR, CR_DRY, PS, RG, FR")
    print(f"Note: Urban start excluded (irreversible)")
    print(f"Ag starts (CR_IRR, CR_DRY, PS): LCC + nr_ur")
    print(f"Range/Forest: LCC only")
    print(f"Choice set restriction: {'ON' if restrict_choice_set else 'OFF'}")
    print(f"Available subregions: {sorted(available_subregions)}")
    print(f"{'='*60}\n")

    for subregion in sorted(available_subregions):
        subregion_name = RPA_SUBREGIONS.get(subregion, subregion)
        print(f"\n[{subregion}] {subregion_name}")
        print("-" * 40)

        for start_name, start_data in starting_conditions.items():
            if start_data is None or len(start_data) == 0:
                continue

            model_key = f'{start_name}_{subregion}'

            # Determine specification based on starting use:
            # - Ag starts (cr_irr, cr_dry, pasture): LCC + nr_ur (urban destination returns)
            # - Range/Forest starts: LCC only (no rent data available)
            include_nr = use_net_returns and start_name in ag_starts_with_nr

            # Prepare data for this subregion
            est_data = prepare_estimation_data_6cat(
                start_data, nr_data, georef, years,
                regions=[subregion],
                use_net_returns=include_nr
            )

            # Apply choice set restriction if enabled
            if restrict_choice_set and county_feasible_uses is not None:
                n_before = len(est_data)
                est_data = filter_to_feasible_choices(est_data, county_feasible_uses)
                n_removed = n_before - len(est_data)
                if n_removed > 0:
                    print(f"  {start_name}: removed {n_removed:,} obs with infeasible end uses")

            if len(est_data) < min_observations:
                print(f"  {start_name}: skipped (n={len(est_data)} < {min_observations})")
                continue

            # Store prepared data
            models[f'{model_key}_data'] = est_data

            # Build formula based on starting use and region
            # Check for region-specific specification first
            if region_specs and subregion in region_specs and start_name in region_specs[subregion]:
                # Use region-specific formula
                spec_vars = region_specs[subregion][start_name]
                formula = f'enduse ~ {spec_vars}'
                spec_label = spec_vars.replace(' ', '')
            elif include_nr and 'nr_ur' in est_data.columns:
                # Default for ag starts: LCC + nr_ur
                formula = 'enduse ~ lcc + nr_ur'
                spec_label = 'LCC+nr_ur'
            else:
                # Default for range/forest: LCC only
                formula = 'enduse ~ lcc'
                spec_label = 'LCC'

            try:
                model = estimate_mnlogit(est_data, formula)
                models[model_key] = model

                # Get coefficient sign summary for net returns variables
                nr_sign_info = ""
                if 'nr_' in formula and not np.isnan(model.llf):
                    nr_sign_info = summarize_nr_coefficients(model, model_key)

                print(f"  {start_name}: n={len(est_data):,}, LL={model.llf:.1f} [{spec_label}]{nr_sign_info}")

            except Exception as e:
                print(f"  {start_name}: FAILED - {str(e)[:50]}")
                models[model_key] = None

    # Summary
    successful = [k for k in models.keys() if not k.endswith('_data') and models[k] is not None]
    print(f"\n{'='*60}")
    print(f"Successfully estimated: {len(successful)} models")
    print(f"{'='*60}")

    return models


def load_data_6cat(
    georef_file: str,
    nr_data_file: str,
    start_cr_irr_file: str,
    start_cr_dry_file: str,
    start_pasture_file: str,
    start_range_file: str,
    start_forest_file: str,
    start_urban_file: str
) -> Tuple[pd.DataFrame, ...]:
    """
    Load data files for 6-category estimation.

    Returns:
        Tuple: georef, nr_data, cr_irr, cr_dry, pasture, range, forest, urban
    """
    # Load geographic reference
    georef = pd.read_csv(georef_file)
    if 'fips' not in georef.columns and 'county_fips' in georef.columns:
        georef = georef.rename(columns={'county_fips': 'fips'})
    if 'fips' in georef.columns:
        georef['fips'] = georef['fips'].astype(int)

    # Load net returns
    nr_data = pd.read_csv(nr_data_file)

    # Load starting condition data
    def load_if_exists(filepath):
        path = Path(filepath)
        if path.exists():
            return pd.read_csv(filepath)
        else:
            print(f"Warning: {filepath} not found")
            return pd.DataFrame()

    cr_irr = load_if_exists(start_cr_irr_file)
    cr_dry = load_if_exists(start_cr_dry_file)
    pasture = load_if_exists(start_pasture_file)
    range_data = load_if_exists(start_range_file)
    forest = load_if_exists(start_forest_file)
    urban = load_if_exists(start_urban_file)

    return georef, nr_data, cr_irr, cr_dry, pasture, range_data, forest, urban


def main_6cat(
    georef_file: str,
    nr_data_file: str,
    start_cr_irr_file: str,
    start_cr_dry_file: str,
    start_pasture_file: str,
    start_range_file: str,
    start_forest_file: str,
    start_urban_file: str,
    output_dir: str,
    use_net_returns: bool = True,
    years: List[int] = None,
    subregions: List[str] = None,
    region_specs: Dict[str, Dict[str, str]] = None
):
    """
    Main function to run 6-category logit estimation.

    Parameters:
    -----------
    georef_file : str
        Path to geographic reference file
    nr_data_file : str
        Path to net returns data (4-column format: ag + urban)
    start_*_file : str
        Paths to starting land use data files
    output_dir : str
        Output directory
    use_net_returns : bool
        Whether to include net returns for ag/urban (default: True)
    years : List[int]
        Years to include
    subregions : List[str], optional
        Specific subregions to estimate
    region_specs : Dict[str, Dict[str, str]], optional
        Region-specific model specifications
    """
    print("Loading 6-category data files...")
    georef, nr_data, cr_irr, cr_dry, pasture, range_data, forest, urban = load_data_6cat(
        georef_file, nr_data_file,
        start_cr_irr_file, start_cr_dry_file,
        start_pasture_file, start_range_file,
        start_forest_file, start_urban_file
    )

    # Report data summary
    print("\nData Summary:")
    print(f"  Geographic reference: {len(georef):,} counties")
    print(f"  Net returns: {len(nr_data):,} county-years")
    print(f"  CR_IRR starts: {len(cr_irr):,}")
    print(f"  CR_DRY starts: {len(cr_dry):,}")
    print(f"  Pasture starts: {len(pasture):,}")
    print(f"  Range starts: {len(range_data):,}")
    print(f"  Forest starts: {len(forest):,}")
    print(f"  Urban starts: {len(urban):,}")

    model_type = "mixed (ag/urban: LCC+NR, range/forest: LCC)" if use_net_returns else "LCC-only"
    print(f"\nEstimating 6-category models ({model_type})...")

    models = estimate_land_use_transitions_6cat(
        cr_irr, cr_dry, pasture, range_data, forest, urban,
        nr_data, georef,
        years=years,
        use_net_returns=use_net_returns,
        subregions=subregions,
        region_specs=region_specs
    )

    print("\nSaving results...")
    save_estimation_results(models, output_dir)

    print(f"\nEstimation complete. Results saved to {output_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print("Model Summary:")
    print('='*60)
    successful_models = [k for k in models.keys() if not k.endswith('_data') and models[k] is not None]
    print(f"Successfully estimated: {len(successful_models)} models")

    for name in sorted(successful_models):
        model = models[name]
        print(f"\n{name}:")
        print(f"  Log-likelihood: {model.llf:.2f}")
        print(f"  AIC: {model.aic:.2f}")
        print(f"  Observations: {model.nobs:.0f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 7:
        print("Usage: python logit_estimation.py <georef> <nr_data> <crop_start> <pasture_start> <forest_start> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])