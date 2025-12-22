#!/usr/bin/env python3
"""
Run CRP enrollment model with real NRI data.
This script loads actual land use transition data containing LCC information
and applies the CRP enrollment model to predict and analyze CRP transitions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from landuse.crp_enrollment import (
    get_crp_enrollment_probability,
    prepare_crp_data,
    estimate_crp_enrollment_model,
    predict_crp_transitions,
    calculate_crp_impacts,
    create_crp_report
)

console = Console()


def load_real_nri_data():
    """Load and combine real NRI transition data."""
    rprint("\n[bold cyan]Loading Real NRI Data[/bold cyan]")

    data_files = {
        'crop': 'test_data/start_crop.csv',
        'pasture': 'test_data/start_pasture.csv',
        'forest': 'test_data/start_forest.csv'
    }

    all_data = []

    for land_type, file_path in data_files.items():
        rprint(f"  Loading {land_type} data from {file_path}...")
        df = pd.read_csv(file_path)
        df['source_type'] = land_type
        all_data.append(df)
        rprint(f"    ✓ Loaded {len(df):,} records")

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    rprint(f"\n[green]Total records loaded: {len(combined_data):,}[/green]")

    # Add CRP indicator (enduse == 12 indicates CRP)
    combined_data['in_crp'] = (combined_data['enduse'] == 12).astype(int)

    return combined_data


def analyze_lcc_distribution(data):
    """Analyze the distribution of Land Capability Classes in the data."""
    rprint("\n[bold cyan]Land Capability Class Distribution[/bold cyan]")

    # Create table
    table = Table(title="LCC Distribution in Real Data")
    table.add_column("LCC", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")
    table.add_column("CRP Rate", style="yellow")
    table.add_column("Description", style="white")

    lcc_descriptions = {
        1: "Very poor - severe limitations",
        2: "Poor - major limitations",
        3: "Marginal - moderate limitations",
        4: "Marginal - moderate limitations",
        5: "Fair - some limitations",
        6: "Good - few limitations",
        7: "Very good - minimal limitations",
        8: "Best - no limitations"
    }

    for lcc in sorted(data['lcc'].unique()):
        lcc_data = data[data['lcc'] == lcc]
        count = len(lcc_data)
        pct = count / len(data) * 100
        crp_rate = lcc_data['in_crp'].mean() * 100
        desc = lcc_descriptions.get(lcc, "Unknown")

        table.add_row(
            str(lcc),
            f"{count:,}",
            f"{pct:.1f}%",
            f"{crp_rate:.2f}%",
            desc
        )

    console.print(table)

    # Summary statistics
    crp_data = data[data['in_crp'] == 1]
    rprint(f"\n[bold]Overall CRP Statistics:[/bold]")
    rprint(f"  Total CRP transitions: {len(crp_data):,}")
    rprint(f"  Overall CRP rate: {data['in_crp'].mean()*100:.2f}%")
    rprint(f"  Most common LCC in CRP: {crp_data['lcc'].mode().values[0] if len(crp_data) > 0 else 'N/A'}")


def estimate_model_with_real_data(data):
    """Estimate CRP enrollment model using real NRI data."""
    rprint("\n[bold cyan]Estimating CRP Model with Real Data[/bold cyan]")

    # Prepare data
    model_data = prepare_crp_data(data)

    # Split by year for train/test
    train_years = [2010, 2012]
    test_years = [2014, 2015, 2017]

    train_data = model_data[model_data['year'].isin(train_years)]
    test_data = model_data[model_data['year'].isin(test_years)]

    rprint(f"  Training set: {len(train_data):,} observations (years: {train_years})")
    rprint(f"  Test set: {len(test_data):,} observations (years: {test_years})")

    # Estimate different model specifications
    models = {}

    try:
        # Model 1: LCC only
        rprint("\n[yellow]Model 1: CRP ~ LCC[/yellow]")
        model1 = estimate_crp_enrollment_model(
            train_data,
            formula="in_crp ~ lcc",
            use_weights=True
        )
        models['lcc_only'] = model1
        rprint(f"  Log-likelihood: {model1.llf:.2f}")
        rprint(f"  Pseudo R²: {model1.prsquared:.4f}")

        # Model 2: LCC + Cropland
        if 'is_cropland' in train_data.columns:
            rprint("\n[yellow]Model 2: CRP ~ LCC + is_cropland[/yellow]")
            model2 = estimate_crp_enrollment_model(
                train_data,
                formula="in_crp ~ lcc + is_cropland",
                use_weights=True
            )
            models['lcc_cropland'] = model2
            rprint(f"  Log-likelihood: {model2.llf:.2f}")
            rprint(f"  Pseudo R²: {model2.prsquared:.4f}")

        # Model 3: LCC + Marginal land
        if 'is_marginal' in train_data.columns:
            rprint("\n[yellow]Model 3: CRP ~ LCC + is_marginal[/yellow]")
            model3 = estimate_crp_enrollment_model(
                train_data,
                formula="in_crp ~ lcc + is_marginal",
                use_weights=True
            )
            models['lcc_marginal'] = model3
            rprint(f"  Log-likelihood: {model3.llf:.2f}")
            rprint(f"  Pseudo R²: {model3.prsquared:.4f}")

        # Select best model
        best_model_name = min(models.items(), key=lambda x: x[1].aic)[0]
        best_model = models[best_model_name]

        rprint(f"\n[green]Best model: {best_model_name} (AIC: {best_model.aic:.2f})[/green]")

        # Display coefficients
        rprint("\n[bold]Model Coefficients:[/bold]")
        coef_table = Table()
        coef_table.add_column("Variable", style="cyan")
        coef_table.add_column("Coefficient", style="magenta")
        coef_table.add_column("Std Error", style="yellow")
        coef_table.add_column("P-value", style="green")

        for var in best_model.params.index:
            coef_table.add_row(
                var,
                f"{best_model.params[var]:.4f}",
                f"{best_model.bse[var]:.4f}",
                f"{best_model.pvalues[var]:.4f}"
            )

        console.print(coef_table)

        return best_model, test_data

    except Exception as e:
        rprint(f"[red]Error estimating model: {e}[/red]")
        return None, test_data


def run_scenarios(model, test_data):
    """Run different CRP enrollment scenarios."""
    rprint("\n[bold cyan]Running CRP Enrollment Scenarios[/bold cyan]")

    scenarios = {
        'baseline': 'Current policy conditions',
        'high_payment': 'Increased CRP payment rates',
        'conservation_priority': 'Focus on environmentally sensitive land'
    }

    results = {}

    for scenario_name, description in scenarios.items():
        rprint(f"\n[yellow]Scenario: {scenario_name}[/yellow]")
        rprint(f"  Description: {description}")

        # Make predictions
        predictions = predict_crp_transitions(model, test_data, scenario_name)

        # Calculate metrics
        enrollment_rate = predictions['crp_prob'].mean() * 100
        transitions = predictions['enters_crp'].sum()

        # Weight by xfact for acres
        if 'xfact' in predictions.columns:
            crp_acres = predictions[predictions['enters_crp']]['xfact'].sum()
        else:
            crp_acres = transitions * 100  # Assume 100 acres per observation

        rprint(f"  Average enrollment probability: {enrollment_rate:.2f}%")
        rprint(f"  Predicted transitions to CRP: {transitions:,}")
        rprint(f"  Estimated CRP acres: {crp_acres:,.0f}")

        # Store results
        results[scenario_name] = {
            'predictions': predictions,
            'enrollment_rate': enrollment_rate,
            'transitions': transitions,
            'acres': crp_acres
        }

    return results


def visualize_results(data, model, scenario_results):
    """Create visualizations of CRP model results."""
    rprint("\n[bold cyan]Creating Visualizations[/bold cyan]")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Actual CRP enrollment by LCC
    ax1 = axes[0, 0]
    lcc_crp = data.groupby('lcc')['in_crp'].mean() * 100
    lcc_crp.plot(kind='bar', ax=ax1, color='forestgreen')
    ax1.set_xlabel('Land Capability Class')
    ax1.set_ylabel('CRP Enrollment Rate (%)')
    ax1.set_title('Actual CRP Enrollment by LCC')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    # 2. Transition sources
    ax2 = axes[0, 1]
    crp_data = data[data['in_crp'] == 1]
    if len(crp_data) > 0:
        source_counts = crp_data['startuse'].value_counts()
        use_names = {1: 'Cropland', 3: 'Pasture', 5: 'Forest', 2: 'Other'}
        labels = [use_names.get(code, f"Code {code}") for code in source_counts.index]
        ax2.pie(source_counts.values, labels=labels, autopct='%1.1f%%',
                colors=['gold', 'lightgreen', 'forestgreen', 'gray'])
        ax2.set_title('Sources of CRP Land')

    # 3. Model predictions vs actual
    ax3 = axes[0, 2]
    if model is not None:
        lcc_range = sorted(data['lcc'].unique())
        actual_rates = [data[data['lcc'] == lcc]['in_crp'].mean() * 100 for lcc in lcc_range]

        # Get model predictions for each LCC
        predicted_rates = []
        for lcc in lcc_range:
            prob = get_crp_enrollment_probability(lcc, current_use=1) * 100
            predicted_rates.append(prob)

        x = np.arange(len(lcc_range))
        width = 0.35

        ax3.bar(x - width/2, actual_rates, width, label='Actual', color='blue', alpha=0.7)
        ax3.bar(x + width/2, predicted_rates, width, label='Model', color='red', alpha=0.7)
        ax3.set_xlabel('Land Capability Class')
        ax3.set_ylabel('CRP Enrollment Rate (%)')
        ax3.set_title('Model vs Actual CRP Rates')
        ax3.set_xticks(x)
        ax3.set_xticklabels(lcc_range)
        ax3.legend()

    # 4. Scenario comparison
    ax4 = axes[1, 0]
    scenario_names = list(scenario_results.keys())
    enrollment_rates = [scenario_results[s]['enrollment_rate'] for s in scenario_names]

    ax4.bar(scenario_names, enrollment_rates, color=['gray', 'green', 'blue'])
    ax4.set_ylabel('Enrollment Rate (%)')
    ax4.set_title('CRP Enrollment by Scenario')
    ax4.set_xticklabels(scenario_names, rotation=45, ha='right')

    # 5. Predicted acres by scenario
    ax5 = axes[1, 1]
    acres = [scenario_results[s]['acres'] for s in scenario_names]
    ax5.bar(scenario_names, acres, color=['gray', 'green', 'blue'])
    ax5.set_ylabel('CRP Acres')
    ax5.set_title('Predicted CRP Acres by Scenario')
    ax5.set_xticklabels(scenario_names, rotation=45, ha='right')

    # 6. Environmental impacts
    ax6 = axes[1, 2]
    baseline_pred = scenario_results['baseline']['predictions']
    impacts = calculate_crp_impacts(baseline_pred)

    if impacts:
        impact_names = ['Erosion\nReduction\n(tons)', 'Carbon\nSequestration\n(tons CO2)']
        impact_values = [
            impacts.get('erosion_reduction_tons', 0),
            impacts.get('carbon_sequestration_tons', 0)
        ]

        ax6.bar(impact_names, impact_values, color=['brown', 'green'])
        ax6.set_ylabel('Annual Impact')
        ax6.set_title('Environmental Benefits (Baseline)')

    plt.suptitle('CRP Enrollment Analysis with Real NRI Data', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = 'crp_real_data_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    rprint(f"  ✓ Saved visualization to {output_file}")
    plt.close()


def main():
    """Main execution function."""
    console.print(Panel.fit(
        "[bold green]CRP Enrollment Model with Real NRI Data[/bold green]\n"
        "Analyzing Conservation Reserve Program enrollment patterns using actual land transition data",
        title="Analysis Start"
    ))

    # Load data
    data = load_real_nri_data()

    # Analyze LCC distribution
    analyze_lcc_distribution(data)

    # Estimate model
    model, test_data = estimate_model_with_real_data(data)

    if model is not None:
        # Run scenarios
        scenario_results = run_scenarios(model, test_data)

        # Create visualizations
        visualize_results(data, model, scenario_results)

        # Generate report
        rprint("\n[bold cyan]Generating Report[/bold cyan]")
        impacts = calculate_crp_impacts(scenario_results['baseline']['predictions'])
        report = create_crp_report(
            models={'real_data_model': model},
            impacts=impacts,
            output_file='crp_real_data_report.txt'
        )
        rprint("  ✓ Report saved to crp_real_data_report.txt")

        # Summary
        console.print(Panel.fit(
            "[bold green]Analysis Complete![/bold green]\n\n"
            "Key Findings:\n"
            f"• Overall CRP enrollment rate: {data['in_crp'].mean()*100:.2f}%\n"
            f"• Model Pseudo R²: {model.prsquared:.4f}\n"
            f"• Baseline scenario predicts {scenario_results['baseline']['acres']:,.0f} CRP acres\n"
            f"• High payment scenario increases enrollment by "
            f"{(scenario_results['high_payment']['enrollment_rate'] - scenario_results['baseline']['enrollment_rate']):.1f}%\n\n"
            "Files generated:\n"
            "• crp_real_data_analysis.png - Visualization of results\n"
            "• crp_real_data_report.txt - Detailed analysis report",
            title="Summary"
        ))
    else:
        rprint("[red]Model estimation failed. Please check the data and try again.[/red]")

    return data, model, scenario_results if model else None


if __name__ == "__main__":
    data, model, results = main()