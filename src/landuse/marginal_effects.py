"""
Calculate and analyze marginal effects from logit models.
Ported from logit_ame_crop_forest.R and mfx_urban.R
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_average_marginal_effects(model: Any,
                                      data: pd.DataFrame,
                                      variables: List[str]) -> pd.DataFrame:
    """
    Calculate average marginal effects for specified variables.

    Parameters:
    -----------
    model : MultinomialResultsWrapper
        Fitted MNLogit model
    data : pd.DataFrame
        Data for marginal effect calculation
    variables : List[str]
        Variables to calculate effects for

    Returns:
    --------
    pd.DataFrame
        Marginal effects results
    """
    results = []

    for var in variables:
        try:
            # Calculate marginal effects
            margeff = model.get_margeff(at='mean', method='dydx', dummy=False)

            # Extract results for this variable
            me_frame = margeff.summary_frame()
            var_effects = me_frame[me_frame.index.str.contains(var)]

            if not var_effects.empty:
                for outcome in var_effects.index:
                    results.append({
                        'variable': var,
                        'outcome': outcome,
                        'marginal_effect': var_effects.loc[outcome, 'dy/dx'],
                        'std_error': var_effects.loc[outcome, 'Std. Err.'],
                        'z_value': var_effects.loc[outcome, 'z'],
                        'p_value': var_effects.loc[outcome, 'P>|z|'],
                        'ci_lower': var_effects.loc[outcome, '[0.025'],
                        'ci_upper': var_effects.loc[outcome, '0.975]']
                    })
        except Exception as e:
            print(f"Could not calculate marginal effects for {var}: {str(e)}")

    return pd.DataFrame(results)


def calculate_elasticities(model: Any,
                          data: pd.DataFrame,
                          price_vars: List[str]) -> pd.DataFrame:
    """
    Calculate elasticities for price variables.

    Parameters:
    -----------
    model : MultinomialResultsWrapper
        Fitted model
    data : pd.DataFrame
        Data for calculation
    price_vars : List[str]
        Price/net return variables

    Returns:
    --------
    pd.DataFrame
        Elasticity estimates
    """
    # Predict probabilities
    probs = model.predict(data)

    elasticities = []

    for price_var in price_vars:
        # Calculate numerical elasticity using finite differences
        delta = 0.01  # 1% change
        data_plus = data.copy()
        data_plus[price_var] = data_plus[price_var] * (1 + delta)

        probs_plus = model.predict(data_plus)

        # Calculate elasticity for each outcome
        for col in probs.columns:
            elasticity = ((probs_plus[col] - probs[col]) / probs[col]) / delta
            mean_elasticity = elasticity.mean()

            elasticities.append({
                'variable': price_var,
                'outcome': col,
                'elasticity': mean_elasticity,
                'std_dev': elasticity.std(),
                'min': elasticity.min(),
                'max': elasticity.max()
            })

    return pd.DataFrame(elasticities)


def weighted_average_marginal_effects(model: Any,
                                     data: pd.DataFrame,
                                     variables: List[str],
                                     weight_col: str = 'xfact') -> pd.DataFrame:
    """
    Calculate weighted average marginal effects.

    Parameters:
    -----------
    model : MultinomialResultsWrapper
        Fitted model
    data : pd.DataFrame
        Data with weights
    variables : List[str]
        Variables of interest
    weight_col : str
        Weight column name

    Returns:
    --------
    pd.DataFrame
        Weighted marginal effects
    """
    # Get predictions for each observation
    X = data[model.exog_names]
    predictions = model.predict(X)

    results = []

    for var in variables:
        if var not in X.columns:
            continue

        # Calculate numerical marginal effects
        delta = X[var].std() * 0.001  # Small change
        X_plus = X.copy()
        X_plus[var] = X_plus[var] + delta

        pred_plus = model.predict(X_plus)

        # Calculate marginal effects
        marginal_effects = (pred_plus - predictions) / delta

        # Calculate weighted averages
        weights = data[weight_col] / data[weight_col].sum()

        for outcome in marginal_effects.columns:
            me = marginal_effects[outcome]
            weighted_me = np.average(me, weights=weights)

            results.append({
                'variable': var,
                'outcome': outcome,
                'weighted_marginal_effect': weighted_me,
                'unweighted_marginal_effect': me.mean(),
                'std_dev': np.sqrt(np.average((me - weighted_me)**2, weights=weights))
            })

    return pd.DataFrame(results)


def plot_marginal_effects(marginal_effects: pd.DataFrame,
                         title: str = "Average Marginal Effects",
                         output_file: Optional[str] = None) -> plt.Figure:
    """
    Create plot of marginal effects with confidence intervals.

    Parameters:
    -----------
    marginal_effects : pd.DataFrame
        Marginal effects data
    title : str
        Plot title
    output_file : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for plotting
    me_pivot = marginal_effects.pivot(index='variable', columns='outcome', values='marginal_effect')

    # Create grouped bar plot
    me_pivot.plot(kind='bar', ax=ax)

    ax.set_xlabel('Variable')
    ax.set_ylabel('Marginal Effect')
    ax.set_title(title)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def plot_elasticities(elasticities: pd.DataFrame,
                     title: str = "Price Elasticities",
                     output_file: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap of elasticities.

    Parameters:
    -----------
    elasticities : pd.DataFrame
        Elasticity estimates
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

    # Pivot for heatmap
    elast_pivot = elasticities.pivot(index='variable', columns='outcome', values='elasticity')

    # Create heatmap
    sns.heatmap(elast_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Elasticity'}, ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Land Use Outcome')
    ax.set_ylabel('Price Variable')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def analyze_temporal_effects(models: Dict[str, Any],
                            data: Dict[str, pd.DataFrame],
                            variables: List[str]) -> pd.DataFrame:
    """
    Analyze how marginal effects change over time.

    Parameters:
    -----------
    models : Dict
        Models by time period
    data : Dict
        Data by time period
    variables : List[str]
        Variables to analyze

    Returns:
    --------
    pd.DataFrame
        Temporal analysis results
    """
    temporal_results = []

    for period, model in models.items():
        if model is None:
            continue

        period_data = data[period]
        me = calculate_average_marginal_effects(model, period_data, variables)
        me['period'] = period
        temporal_results.append(me)

    return pd.concat(temporal_results, ignore_index=True)


def decompose_probability_changes(model: Any,
                                 base_data: pd.DataFrame,
                                 scenario_data: pd.DataFrame,
                                 variables: List[str]) -> pd.DataFrame:
    """
    Decompose probability changes into contributions from each variable.

    Parameters:
    -----------
    model : MultinomialResultsWrapper
        Fitted model
    base_data : pd.DataFrame
        Baseline scenario data
    scenario_data : pd.DataFrame
        Alternative scenario data
    variables : List[str]
        Variables to decompose

    Returns:
    --------
    pd.DataFrame
        Decomposition results
    """
    # Calculate baseline probabilities
    base_probs = model.predict(base_data)
    scenario_probs = model.predict(scenario_data)
    total_change = scenario_probs - base_probs

    decomposition = []

    for var in variables:
        # Create intermediate scenario changing only this variable
        intermediate = base_data.copy()
        intermediate[var] = scenario_data[var]

        intermediate_probs = model.predict(intermediate)
        var_contribution = intermediate_probs - base_probs

        for outcome in var_contribution.columns:
            decomposition.append({
                'variable': var,
                'outcome': outcome,
                'contribution': var_contribution[outcome].mean(),
                'total_change': total_change[outcome].mean(),
                'percent_contribution': (var_contribution[outcome].mean() /
                                       total_change[outcome].mean() * 100
                                       if total_change[outcome].mean() != 0 else 0)
            })

    return pd.DataFrame(decomposition)


def create_marginal_effects_report(models: Dict,
                                  data: Dict,
                                  output_dir: str):
    """
    Create comprehensive marginal effects analysis report.

    Parameters:
    -----------
    models : Dict
        Dictionary of fitted models
    data : Dict
        Dictionary of data
    output_dir : str
        Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Variables to analyze
    net_return_vars = ['nr_cr', 'nr_ps', 'nr_fr', 'nr_ur']
    other_vars = ['lcc']
    all_vars = net_return_vars + other_vars

    all_results = []

    for model_name, model in models.items():
        if model is None or model_name.endswith('_data'):
            continue

        print(f"Analyzing {model_name}...")

        # Get corresponding data
        data_name = model_name + '_data'
        if data_name not in data:
            continue

        model_data = data[data_name]

        # Calculate marginal effects
        me = calculate_average_marginal_effects(model, model_data, all_vars)
        me['model'] = model_name
        all_results.append(me)

        # Calculate elasticities
        elast = calculate_elasticities(model, model_data, net_return_vars)
        elast.to_csv(output_path / f'{model_name}_elasticities.csv', index=False)

        # Create visualizations
        plot_marginal_effects(me, f"Marginal Effects - {model_name}",
                            output_path / f'{model_name}_marginal_effects.pdf')

        plot_elasticities(elast, f"Elasticities - {model_name}",
                        output_path / f'{model_name}_elasticities.pdf')

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(output_path / 'all_marginal_effects.csv', index=False)

        # Create summary statistics
        summary = combined_results.groupby(['model', 'variable', 'outcome']).agg({
            'marginal_effect': ['mean', 'std', 'min', 'max'],
            'p_value': 'mean'
        }).round(4)
        summary.to_csv(output_path / 'marginal_effects_summary.csv')

    print(f"Marginal effects analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import sys
    import pickle

    if len(sys.argv) != 3:
        print("Usage: python marginal_effects.py <models_file> <output_dir>")
        sys.exit(1)

    # Load models
    with open(sys.argv[1], 'rb') as f:
        models_data = pickle.load(f)

    # Separate models and data
    models = {k: v for k, v in models_data.items() if not k.endswith('_data')}
    data = {k: v for k, v in models_data.items() if k.endswith('_data')}

    create_marginal_effects_report(models, data, sys.argv[2])