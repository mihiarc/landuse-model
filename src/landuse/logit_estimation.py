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
                           regions: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepare data for logit estimation.

    Parameters:
    -----------
    start_data : pd.DataFrame
        Starting land use data
    nr_data : pd.DataFrame
        Net returns data
    georef : pd.DataFrame
        Geographic reference data with regions
    years : List[int]
        Years to include in estimation
    regions : List[str], optional
        Regions to filter

    Returns:
    --------
    pd.DataFrame
        Prepared estimation data
    """
    # Apply lag to net returns
    nr_data = nr_data.copy()
    nr_data['year'] = nr_data['year'] + 2

    # Merge data
    data = start_data.merge(nr_data, on=['fips', 'year'])

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

    # Filter by years if specified
    if years:
        data = data[data['year'].isin(years)]

    # Filter by regions if specified
    if regions:
        data = data[data['region'].isin(regions)]

    # Calculate weights
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
                                 years: List[int] = None) -> Dict:
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
        Net returns data
    georef : pd.DataFrame
        Geographic reference data
    years : List[int]
        Years to include

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
            print(f"Estimating model for {start_name} start in {region_name} region...")

            # Prepare data
            est_data = prepare_estimation_data(
                start_data, nr_data, georef, years, region_filter
            )

            # Store prepared data
            models[f'{start_name}start_{region_name}_data'] = est_data

            # Define formula based on available variables
            formula = 'enduse ~ lcc + nr_cr + nr_ps + nr_fr + nr_ur'

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
         output_dir: str):
    """
    Main function to run logit estimation.

    Parameters:
    -----------
    georef_file : str
        Path to geographic reference file
    nr_data_file : str
        Path to net returns data
    start_crop_file : str
        Path to crop start data
    start_pasture_file : str
        Path to pasture start data
    start_forest_file : str
        Path to forest start data
    output_dir : str
        Output directory
    """
    print("Loading data files...")
    georef, nr_data, start_crop, start_pasture, start_forest = load_data(
        georef_file, nr_data_file, start_crop_file, start_pasture_file, start_forest_file
    )

    print("Estimating land use transition models...")
    models = estimate_land_use_transitions(
        start_crop, start_pasture, start_forest, nr_data, georef
    )

    print("Saving results...")
    save_estimation_results(models, output_dir)

    print(f"Estimation complete. Results saved to {output_dir}")

    # Print summary of models
    print("\nModel Summary:")
    for name, model in models.items():
        if not name.endswith('_data') and model is not None:
            print(f"\n{name}:")
            print(f"  Log-likelihood: {model.llf:.2f}")
            print(f"  AIC: {model.aic:.2f}")
            print(f"  Number of observations: {model.nobs:.0f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 7:
        print("Usage: python logit_estimation.py <georef> <nr_data> <crop_start> <pasture_start> <forest_start> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])