"""
Conservation Reserve Program (CRP) Enrollment Modeling Module

This module models CRP enrollment probability as a function of Land Capability Class (LCC).
CRP typically targets environmentally sensitive agricultural land that would benefit from
conservation practices.

Key relationships:
- Mid-range LCC land (3-6) is most likely to enroll in CRP
- Very high quality land (LCC 7-8) stays in production (too valuable for crops)
- Very low quality land (LCC 1-2) may already be in forest/non-agricultural use
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from pathlib import Path


def get_crp_enrollment_probability(lcc: int,
                                  current_use: int = None,
                                  crp_payment_rate: float = None) -> float:
    """
    Calculate base CRP enrollment probability based on LCC.

    Parameters:
    -----------
    lcc : int
        Land Capability Class (1-8, higher = better quality)
    current_use : int, optional
        Current land use (NRI BROAD code)
    crp_payment_rate : float, optional
        CRP payment rate ($/acre) if available

    Returns:
    --------
    float
        Probability of CRP enrollment (0-1)
    """
    # Base probability by LCC
    # CRP targets marginal cropland - not the best, not the worst
    lcc_probabilities = {
        1: 0.02,  # Very poor land - likely already not in crops
        2: 0.05,  # Poor land - some potential
        3: 0.15,  # Marginal land - good CRP candidate
        4: 0.25,  # Marginal land - prime CRP target
        5: 0.30,  # Marginal land - prime CRP target
        6: 0.20,  # Moderate quality - some CRP potential
        7: 0.08,  # Good land - usually stays in production
        8: 0.03   # Best land - rarely enters CRP
    }

    base_prob = lcc_probabilities.get(lcc, 0.10)

    # Adjust for current use (CRP typically comes from cropland)
    if current_use is not None:
        if current_use == 1:  # Cropland - most likely to enter CRP
            base_prob *= 1.5
        elif current_use == 3:  # Pasture - moderate CRP potential
            base_prob *= 0.8
        elif current_use == 5:  # Forest - unlikely to enter CRP
            base_prob *= 0.1
        else:  # Other uses - very unlikely
            base_prob *= 0.05

    # Adjust for payment rate if provided
    if crp_payment_rate is not None:
        # Higher payments increase enrollment probability
        # Assumes base rate of $50/acre
        payment_factor = crp_payment_rate / 50.0
        base_prob *= min(2.0, max(0.5, payment_factor))

    return min(1.0, max(0.0, base_prob))


def prepare_crp_data(land_data: pd.DataFrame,
                     crp_history: pd.DataFrame = None,
                     georef: pd.DataFrame = None) -> pd.DataFrame:
    """
    Prepare data for CRP enrollment modeling.

    Parameters:
    -----------
    land_data : pd.DataFrame
        Land use data with LCC and current use
    crp_history : pd.DataFrame, optional
        Historical CRP enrollment data
    georef : pd.DataFrame, optional
        Geographic reference data

    Returns:
    --------
    pd.DataFrame
        Prepared data for CRP modeling
    """
    data = land_data.copy()

    # Create binary CRP enrollment variable if not present
    if 'in_crp' not in data.columns:
        # Check if enduse == 12 (CRP code)
        if 'enduse' in data.columns:
            data['in_crp'] = (data['enduse'] == 12).astype(int)
        else:
            data['in_crp'] = 0

    # Create LCC categorical variables for better model fit
    data['lcc_cat'] = pd.cut(data['lcc'],
                             bins=[0, 2, 4, 6, 8],
                             labels=['poor', 'marginal', 'moderate', 'good'])

    # Create dummy for marginal land (prime CRP target)
    data['is_marginal'] = ((data['lcc'] >= 3) & (data['lcc'] <= 5)).astype(int)

    # Create dummy for current cropland
    if 'startuse' in data.columns:
        data['is_cropland'] = (data['startuse'] == 1).astype(int)

    # Merge historical CRP data if provided
    if crp_history is not None:
        data = data.merge(crp_history, on=['fips', 'year'], how='left', suffixes=('', '_hist'))

    # Merge geographic data if provided
    if georef is not None:
        if 'fips' in data.columns and 'fips' in georef.columns:
            georef_subset = georef[['fips', 'region', 'subregion']].drop_duplicates()
            data = data.merge(georef_subset, on='fips', how='left')

    return data


def estimate_crp_enrollment_model(data: pd.DataFrame,
                                 formula: str = None,
                                 use_weights: bool = True) -> Any:
    """
    Estimate CRP enrollment probability model.

    Parameters:
    -----------
    data : pd.DataFrame
        Prepared data with CRP enrollment and predictors
    formula : str, optional
        Model formula (default: 'in_crp ~ lcc + is_cropland')
    use_weights : bool
        Whether to use survey weights if available

    Returns:
    --------
    LogitResults
        Fitted logit model results
    """
    if formula is None:
        # Default formula - CRP as function of LCC and current use
        base_vars = ['lcc']

        # Add available variables
        if 'is_cropland' in data.columns:
            base_vars.append('is_cropland')
        if 'is_marginal' in data.columns:
            base_vars.append('is_marginal')

        formula = f"in_crp ~ {' + '.join(base_vars)}"

    print(f"Estimating CRP enrollment model with formula: {formula}")

    # Create design matrix
    from patsy import dmatrices
    y, X = dmatrices(formula, data, return_type='dataframe')

    # Handle weights
    if use_weights and 'weight' in data.columns:
        weights = data['weight'].values
    elif use_weights and 'xfact' in data.columns:
        weights = data['xfact'].values
        weights = weights / weights.sum() * len(weights)
    else:
        weights = None

    # Fit logit model
    if weights is not None:
        model = Logit(y, X)
        result = model.fit_regularized(method='l1', alpha=0.0, disp=False)
    else:
        model = Logit(y, X)
        result = model.fit(method='bfgs', maxiter=1000, disp=False)

    return result


def predict_crp_transitions(model: Any,
                           land_data: pd.DataFrame,
                           scenario: str = 'baseline') -> pd.DataFrame:
    """
    Predict CRP enrollment transitions under different scenarios.

    Parameters:
    -----------
    model : LogitResults
        Fitted CRP enrollment model
    land_data : pd.DataFrame
        Current land use data
    scenario : str
        Scenario name ('baseline', 'high_payment', 'conservation_priority')

    Returns:
    --------
    pd.DataFrame
        Predicted CRP enrollment probabilities and transitions
    """
    data = land_data.copy()

    # Get baseline predictions
    # Build X to match model's expected columns
    X_dict = {}

    # Add intercept if in model
    if 'Intercept' in model.params.index:
        X_dict['Intercept'] = 1
    elif 'const' in model.params.index:
        X_dict['const'] = 1

    # Add other variables
    if 'lcc' in model.params.index:
        X_dict['lcc'] = data['lcc']
    if 'is_cropland' in model.params.index:
        X_dict['is_cropland'] = data.get('is_cropland', 0)
    if 'is_marginal' in model.params.index:
        X_dict['is_marginal'] = data.get('is_marginal', 0)

    X = pd.DataFrame(X_dict, index=data.index)

    # Predict probabilities
    data['crp_prob_baseline'] = model.predict(X)

    # Apply scenario adjustments
    if scenario == 'high_payment':
        # Higher payments increase enrollment, especially for marginal land
        data['crp_prob'] = data['crp_prob_baseline'] * 1.5
        if 'lcc' in data.columns:
            # Extra boost for marginal land
            marginal_mask = (data['lcc'] >= 3) & (data['lcc'] <= 5)
            data.loc[marginal_mask, 'crp_prob'] *= 1.2

    elif scenario == 'conservation_priority':
        # Focus on environmentally sensitive land
        data['crp_prob'] = data['crp_prob_baseline']
        if 'lcc' in data.columns:
            # Prioritize LCC 3-5
            priority_mask = (data['lcc'] >= 3) & (data['lcc'] <= 5)
            data.loc[priority_mask, 'crp_prob'] *= 1.8
            # Reduce enrollment on best land
            good_land_mask = data['lcc'] >= 7
            data.loc[good_land_mask, 'crp_prob'] *= 0.5

    else:  # baseline
        data['crp_prob'] = data['crp_prob_baseline']

    # Cap probabilities at 0-1
    data['crp_prob'] = data['crp_prob'].clip(0, 1)

    # Simulate transitions based on probability
    np.random.seed(42)
    data['enters_crp'] = np.random.random(len(data)) < data['crp_prob']

    # Calculate transition summary
    if 'startuse' in data.columns:
        data['new_use'] = data['startuse'].copy()
        data.loc[data['enters_crp'], 'new_use'] = 12  # CRP code

    return data


def calculate_crp_impacts(transitions: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate environmental and economic impacts of CRP enrollment.

    Parameters:
    -----------
    transitions : pd.DataFrame
        Predicted CRP transitions with land characteristics

    Returns:
    --------
    Dict[str, float]
        Impact metrics
    """
    impacts = {}

    # Total acres entering CRP
    if 'enters_crp' in transitions.columns and 'xfact' in transitions.columns:
        crp_acres = transitions[transitions['enters_crp']]['xfact'].sum()
        impacts['total_crp_acres'] = crp_acres

    # Distribution by LCC
    if 'enters_crp' in transitions.columns and 'lcc' in transitions.columns:
        crp_by_lcc = transitions[transitions['enters_crp']].groupby('lcc').size()
        impacts['crp_by_lcc'] = crp_by_lcc.to_dict()

    # Source land use
    if 'enters_crp' in transitions.columns and 'startuse' in transitions.columns:
        crp_sources = transitions[transitions['enters_crp']]['startuse'].value_counts()
        impacts['crp_from_cropland'] = crp_sources.get(1, 0)
        impacts['crp_from_pasture'] = crp_sources.get(3, 0)

    # Environmental benefits (simplified estimates)
    if 'total_crp_acres' in impacts:
        # Soil erosion reduction (tons/year)
        # Assume 5 tons/acre/year reduction on average
        impacts['erosion_reduction_tons'] = impacts['total_crp_acres'] * 5

        # Carbon sequestration (tons CO2/year)
        # Assume 1.5 tons CO2/acre/year
        impacts['carbon_sequestration_tons'] = impacts['total_crp_acres'] * 1.5

        # Water quality improvement (relative index)
        impacts['water_quality_index'] = min(100, impacts['total_crp_acres'] / 1000)

    return impacts


def create_crp_report(models: Dict[str, Any],
                      impacts: Dict[str, float],
                      output_file: str = None) -> str:
    """
    Create a summary report of CRP enrollment modeling results.

    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of fitted models by region/scenario
    impacts : Dict[str, float]
        Environmental and economic impacts
    output_file : str, optional
        Path to save report

    Returns:
    --------
    str
        Formatted report text
    """
    report = []
    report.append("=" * 70)
    report.append("CRP ENROLLMENT MODELING REPORT")
    report.append("=" * 70)
    report.append("")

    # Model results
    report.append("MODEL ESTIMATION RESULTS")
    report.append("-" * 40)

    for name, model in models.items():
        if model is not None and hasattr(model, 'params'):
            report.append(f"\nModel: {name}")
            report.append(f"  Observations: {model.nobs:.0f}")
            report.append(f"  Log-likelihood: {model.llf:.2f}")
            report.append(f"  Pseudo R-squared: {model.prsquared:.4f}")

            # Key coefficients
            report.append("  Key Coefficients:")
            if 'lcc' in model.params.index:
                lcc_coef = model.params['lcc']
                lcc_pval = model.pvalues['lcc']
                report.append(f"    LCC: {lcc_coef:.4f} (p={lcc_pval:.4f})")

                # Interpretation
                if lcc_coef < 0:
                    report.append("    → Negative: Lower quality land more likely to enter CRP")
                else:
                    report.append("    → Positive: Higher quality land more likely to enter CRP")

            if 'is_cropland' in model.params.index:
                crop_coef = model.params['is_cropland']
                report.append(f"    Cropland: {crop_coef:.4f}")
                if crop_coef > 0:
                    report.append("    → Cropland more likely to enter CRP than other uses")

    # Impact summary
    report.append("\n" + "=" * 70)
    report.append("ENVIRONMENTAL IMPACTS")
    report.append("-" * 40)

    if 'total_crp_acres' in impacts:
        report.append(f"Total CRP Enrollment: {impacts['total_crp_acres']:,.0f} acres")

    if 'crp_by_lcc' in impacts:
        report.append("\nEnrollment by Land Capability Class:")
        for lcc, count in sorted(impacts['crp_by_lcc'].items()):
            report.append(f"  LCC {lcc}: {count} parcels")

    if 'erosion_reduction_tons' in impacts:
        report.append(f"\nSoil Erosion Reduction: {impacts['erosion_reduction_tons']:,.0f} tons/year")

    if 'carbon_sequestration_tons' in impacts:
        report.append(f"Carbon Sequestration: {impacts['carbon_sequestration_tons']:,.0f} tons CO2/year")

    # Join report lines
    report_text = "\n".join(report)

    # Save if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)

    return report_text


# Example usage
if __name__ == "__main__":
    # Example: Calculate CRP probability for different LCC values
    print("CRP Enrollment Probability by LCC:")
    print("-" * 40)
    for lcc in range(1, 9):
        prob_from_crop = get_crp_enrollment_probability(lcc, current_use=1)
        prob_from_pasture = get_crp_enrollment_probability(lcc, current_use=3)
        print(f"LCC {lcc}: From Crop={prob_from_crop:.2%}, From Pasture={prob_from_pasture:.2%}")