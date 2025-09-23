#!/usr/bin/env python3
"""
Simulate CRP enrollment on real NRI data using LCC-based probabilities.
Since the real data doesn't have actual CRP transitions, we simulate them
based on Land Capability Class and the CRP enrollment probability model.
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
from rich.progress import track

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from landuse.crp_enrollment import (
    get_crp_enrollment_probability,
    prepare_crp_data,
    calculate_crp_impacts
)
from landuse.nri_codes import NRI_BROAD_CODES, MODELING_CODES

console = Console()


def load_and_prepare_data():
    """Load real NRI data and prepare for CRP simulation."""
    rprint("\n[bold cyan]Loading Real NRI Data[/bold cyan]")

    # Load all land use transition data
    crop_data = pd.read_csv('test_data/start_crop.csv')
    pasture_data = pd.read_csv('test_data/start_pasture.csv')
    forest_data = pd.read_csv('test_data/start_forest.csv')

    rprint(f"  Crop data: {len(crop_data):,} records")
    rprint(f"  Pasture data: {len(pasture_data):,} records")
    rprint(f"  Forest data: {len(forest_data):,} records")

    # Combine all data
    all_data = pd.concat([crop_data, pasture_data, forest_data], ignore_index=True)
    rprint(f"[green]Total records: {len(all_data):,}[/green]")

    return all_data


def simulate_crp_enrollment(data, scenario='baseline', crp_target_rate=0.10):
    """
    Simulate CRP enrollment based on LCC and land use.

    Parameters:
    -----------
    data : pd.DataFrame
        Land use data with LCC information
    scenario : str
        Scenario name ('baseline', 'high_payment', 'conservation_focus')
    crp_target_rate : float
        Target overall CRP enrollment rate (default 10%)
    """
    rprint(f"\n[bold cyan]Simulating CRP Enrollment - {scenario}[/bold cyan]")

    # Calculate base CRP probability for each parcel
    data['crp_prob_base'] = data.apply(
        lambda row: get_crp_enrollment_probability(
            lcc=row['lcc'],
            current_use=row['startuse']
        ),
        axis=1
    )

    # Adjust probabilities by scenario
    if scenario == 'high_payment':
        # Higher payments increase enrollment
        data['crp_prob'] = data['crp_prob_base'] * 1.5
        # Extra boost for marginal land (LCC 3-5)
        marginal_mask = (data['lcc'] >= 3) & (data['lcc'] <= 5)
        data.loc[marginal_mask, 'crp_prob'] *= 1.2

    elif scenario == 'conservation_focus':
        # Focus on environmentally sensitive land
        data['crp_prob'] = data['crp_prob_base']
        # Prioritize marginal cropland (LCC 3-5)
        priority_mask = ((data['lcc'] >= 3) & (data['lcc'] <= 5) &
                        (data['startuse'] == 1))
        data.loc[priority_mask, 'crp_prob'] *= 2.0
        # Reduce enrollment on best land
        good_land_mask = data['lcc'] >= 7
        data.loc[good_land_mask, 'crp_prob'] *= 0.3

    else:  # baseline
        data['crp_prob'] = data['crp_prob_base']

    # Cap probabilities at 0-1
    data['crp_prob'] = data['crp_prob'].clip(0, 1)

    # Simulate enrollment decisions
    np.random.seed(42)
    data['enters_crp'] = np.random.random(len(data)) < data['crp_prob']

    # Create new land use variable
    data['new_enduse'] = data['enduse'].copy()
    data.loc[data['enters_crp'], 'new_enduse'] = 12  # CRP code

    # Calculate metrics
    enrollment_rate = data['enters_crp'].mean() * 100
    crp_acres = data[data['enters_crp']]['xfact'].sum()

    rprint(f"  Simulated enrollment rate: {enrollment_rate:.2f}%")
    rprint(f"  Total CRP acres: {crp_acres:,.0f}")

    return data


def analyze_crp_by_lcc(data):
    """Analyze CRP enrollment patterns by Land Capability Class."""
    rprint("\n[bold cyan]CRP Enrollment Analysis by LCC[/bold cyan]")

    # Create analysis table
    table = Table(title="Simulated CRP Enrollment by Land Capability Class")
    table.add_column("LCC", style="cyan")
    table.add_column("Total\nParcels", style="magenta")
    table.add_column("CRP\nParcels", style="yellow")
    table.add_column("CRP Rate", style="green")
    table.add_column("Total\nAcres", style="blue")
    table.add_column("CRP\nAcres", style="red")
    table.add_column("Primary\nSource", style="white")

    for lcc in sorted(data['lcc'].unique()):
        lcc_data = data[data['lcc'] == lcc]
        crp_data = lcc_data[lcc_data['enters_crp']]

        total_parcels = len(lcc_data)
        crp_parcels = len(crp_data)
        crp_rate = crp_parcels / total_parcels * 100 if total_parcels > 0 else 0

        total_acres = lcc_data['xfact'].sum()
        crp_acres = crp_data['xfact'].sum()

        # Find primary source
        if len(crp_data) > 0:
            primary_source = crp_data['startuse'].mode().values[0]
            source_name = MODELING_CODES.get(primary_source, f"Code {primary_source}")
        else:
            source_name = "N/A"

        table.add_row(
            str(lcc),
            f"{total_parcels:,}",
            f"{crp_parcels:,}",
            f"{crp_rate:.1f}%",
            f"{total_acres:,.0f}",
            f"{crp_acres:,.0f}",
            source_name
        )

    console.print(table)

    # Summary by source land use
    rprint("\n[bold]CRP Sources (from which land use):[/bold]")
    crp_data = data[data['enters_crp']]
    source_summary = crp_data.groupby('startuse').agg({
        'xfact': 'sum',
        'enters_crp': 'count'
    }).rename(columns={'enters_crp': 'count'})

    for startuse, row in source_summary.iterrows():
        use_name = MODELING_CODES.get(startuse, f"Code {startuse}")
        pct = row['count'] / len(crp_data) * 100
        rprint(f"  From {use_name}: {row['count']:,} parcels ({pct:.1f}%), {row['xfact']:,.0f} acres")


def create_visualizations(results_dict):
    """Create comprehensive visualizations of CRP simulation results."""
    rprint("\n[bold cyan]Creating Visualizations[/bold cyan]")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Get baseline data for reference
    baseline_data = results_dict['baseline']

    # 1. CRP enrollment rate by LCC (all scenarios)
    ax1 = axes[0, 0]
    for scenario_name, data in results_dict.items():
        lcc_rates = data.groupby('lcc')['enters_crp'].mean() * 100
        ax1.plot(lcc_rates.index, lcc_rates.values, marker='o', label=scenario_name)
    ax1.set_xlabel('Land Capability Class')
    ax1.set_ylabel('CRP Enrollment Rate (%)')
    ax1.set_title('CRP Enrollment by LCC - All Scenarios')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Total CRP acres by scenario
    ax2 = axes[0, 1]
    scenarios = list(results_dict.keys())
    total_acres = [results_dict[s][results_dict[s]['enters_crp']]['xfact'].sum()
                   for s in scenarios]
    colors = ['gray', 'green', 'blue'][:len(scenarios)]
    ax2.bar(scenarios, total_acres, color=colors)
    ax2.set_ylabel('Total CRP Acres')
    ax2.set_title('Total CRP Enrollment by Scenario')
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')

    # 3. Source land use pie chart (baseline)
    ax3 = axes[0, 2]
    crp_data = baseline_data[baseline_data['enters_crp']]
    source_counts = crp_data['startuse'].value_counts()
    labels = [MODELING_CODES.get(code, f"Code {code}") for code in source_counts.index]
    ax3.pie(source_counts.values, labels=labels, autopct='%1.1f%%',
            colors=['gold', 'lightgreen', 'forestgreen'])
    ax3.set_title('CRP Sources - Baseline Scenario')

    # 4. LCC distribution of CRP land
    ax4 = axes[1, 0]
    crp_lcc = baseline_data[baseline_data['enters_crp']]['lcc'].value_counts().sort_index()
    ax4.bar(crp_lcc.index, crp_lcc.values, color='forestgreen')
    ax4.set_xlabel('Land Capability Class')
    ax4.set_ylabel('Number of CRP Parcels')
    ax4.set_title('LCC Distribution of CRP Land')

    # 5. Probability vs actual enrollment
    ax5 = axes[1, 1]
    prob_bins = np.linspace(0, 1, 11)
    baseline_data['prob_bin'] = pd.cut(baseline_data['crp_prob'], prob_bins)
    enrollment_by_prob = baseline_data.groupby('prob_bin')['enters_crp'].mean() * 100
    ax5.bar(range(len(enrollment_by_prob)), enrollment_by_prob.values, color='blue', alpha=0.7)
    ax5.set_xlabel('Probability Bin')
    ax5.set_ylabel('Actual Enrollment Rate (%)')
    ax5.set_title('Enrollment Rate by Probability')
    ax5.set_xticks(range(len(enrollment_by_prob)))
    ax5.set_xticklabels([f"{b.left:.1f}-{b.right:.1f}" for b in enrollment_by_prob.index],
                        rotation=45, ha='right')

    # 6. Environmental impacts comparison
    ax6 = axes[1, 2]
    impact_data = []
    for scenario_name, data in results_dict.items():
        impacts = calculate_crp_impacts(data)
        impact_data.append({
            'scenario': scenario_name,
            'erosion': impacts.get('erosion_reduction_tons', 0),
            'carbon': impacts.get('carbon_sequestration_tons', 0)
        })

    impact_df = pd.DataFrame(impact_data)
    x = np.arange(len(impact_df))
    width = 0.35
    ax6.bar(x - width/2, impact_df['erosion'], width, label='Erosion Reduction', color='brown')
    ax6.bar(x + width/2, impact_df['carbon'], width, label='Carbon Sequestration', color='green')
    ax6.set_xlabel('Scenario')
    ax6.set_ylabel('Annual Impact (tons)')
    ax6.set_title('Environmental Benefits by Scenario')
    ax6.set_xticks(x)
    ax6.set_xticklabels(impact_df['scenario'])
    ax6.legend()

    # 7. CRP by year (if multiple years)
    ax7 = axes[2, 0]
    if 'year' in baseline_data.columns:
        yearly_crp = baseline_data.groupby('year')['enters_crp'].agg(['sum', 'mean'])
        yearly_crp['mean'] *= 100
        ax7.bar(yearly_crp.index, yearly_crp['sum'], color='darkblue', alpha=0.7)
        ax7.set_xlabel('Year')
        ax7.set_ylabel('CRP Transitions')
        ax7.set_title('CRP Enrollment Over Time')
        ax7_twin = ax7.twinx()
        ax7_twin.plot(yearly_crp.index, yearly_crp['mean'], color='red', marker='o', label='Rate %')
        ax7_twin.set_ylabel('Enrollment Rate (%)', color='red')
        ax7_twin.tick_params(axis='y', labelcolor='red')

    # 8. Regional patterns (if FIPS available)
    ax8 = axes[2, 1]
    if 'fips' in baseline_data.columns:
        # Get top 10 counties by CRP enrollment
        county_crp = baseline_data[baseline_data['enters_crp']].groupby('fips')['xfact'].sum()
        top_counties = county_crp.nlargest(10)
        ax8.barh(range(len(top_counties)), top_counties.values, color='teal')
        ax8.set_yticks(range(len(top_counties)))
        ax8.set_yticklabels([f"County {fips}" for fips in top_counties.index])
        ax8.set_xlabel('CRP Acres')
        ax8.set_title('Top 10 Counties by CRP Enrollment')

    # 9. Scenario comparison summary
    ax9 = axes[2, 2]
    summary_data = []
    for scenario_name, data in results_dict.items():
        summary_data.append({
            'Scenario': scenario_name,
            'Enrollment Rate': f"{data['enters_crp'].mean() * 100:.1f}%",
            'Total Acres': f"{data[data['enters_crp']]['xfact'].sum():,.0f}",
            'Avg LCC': f"{data[data['enters_crp']]['lcc'].mean():.1f}"
        })

    # Create text summary
    ax9.axis('tight')
    ax9.axis('off')
    summary_df = pd.DataFrame(summary_data)
    table_data = ax9.table(cellText=summary_df.values,
                          colLabels=summary_df.columns,
                          cellLoc='center',
                          loc='center')
    table_data.auto_set_font_size(False)
    table_data.set_fontsize(9)
    table_data.scale(1, 1.5)
    ax9.set_title('Scenario Comparison Summary', y=0.95)

    plt.suptitle('CRP Enrollment Simulation Results - Real NRI Data with LCC',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = 'crp_simulation_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    rprint(f"  ✓ Saved visualization to {output_file}")
    plt.close()


def generate_report(results_dict):
    """Generate a comprehensive report of CRP simulation results."""
    rprint("\n[bold cyan]Generating Report[/bold cyan]")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CRP ENROLLMENT SIMULATION REPORT")
    report_lines.append("Using Real NRI Data with Land Capability Classes")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall summary
    baseline_data = results_dict['baseline']
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total land parcels analyzed: {len(baseline_data):,}")
    report_lines.append(f"Total acres represented: {baseline_data['xfact'].sum():,.0f}")
    report_lines.append(f"LCC range: {baseline_data['lcc'].min()} - {baseline_data['lcc'].max()}")
    report_lines.append("")

    # Scenario results
    report_lines.append("SCENARIO RESULTS")
    report_lines.append("-" * 40)

    for scenario_name, data in results_dict.items():
        crp_data = data[data['enters_crp']]
        report_lines.append(f"\n{scenario_name.upper()} SCENARIO:")
        report_lines.append(f"  Enrollment rate: {data['enters_crp'].mean() * 100:.2f}%")
        report_lines.append(f"  CRP parcels: {len(crp_data):,}")
        report_lines.append(f"  CRP acres: {crp_data['xfact'].sum():,.0f}")

        if len(crp_data) > 0:
            report_lines.append(f"  Average LCC of CRP land: {crp_data['lcc'].mean():.2f}")

            # Source breakdown
            source_pct = crp_data.groupby('startuse').size() / len(crp_data) * 100
            report_lines.append("  Sources:")
            for use_code, pct in source_pct.items():
                use_name = MODELING_CODES.get(use_code, f"Code {use_code}")
                report_lines.append(f"    From {use_name}: {pct:.1f}%")

    # Environmental impacts
    report_lines.append("\n" + "=" * 80)
    report_lines.append("ENVIRONMENTAL IMPACTS")
    report_lines.append("-" * 40)

    for scenario_name, data in results_dict.items():
        impacts = calculate_crp_impacts(data)
        report_lines.append(f"\n{scenario_name.upper()}:")
        if 'erosion_reduction_tons' in impacts:
            report_lines.append(f"  Erosion reduction: {impacts['erosion_reduction_tons']:,.0f} tons/year")
        if 'carbon_sequestration_tons' in impacts:
            report_lines.append(f"  Carbon sequestration: {impacts['carbon_sequestration_tons']:,.0f} tons CO2/year")

    # LCC analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("LAND CAPABILITY CLASS ANALYSIS")
    report_lines.append("-" * 40)

    baseline_crp = baseline_data[baseline_data['enters_crp']]
    lcc_summary = baseline_crp.groupby('lcc').agg({
        'enters_crp': 'count',
        'xfact': 'sum'
    }).rename(columns={'enters_crp': 'count'})

    report_lines.append("\nCRP Enrollment by LCC (Baseline):")
    for lcc, row in lcc_summary.iterrows():
        total_lcc = len(baseline_data[baseline_data['lcc'] == lcc])
        rate = row['count'] / total_lcc * 100 if total_lcc > 0 else 0
        report_lines.append(f"  LCC {lcc}: {row['count']:,} parcels ({rate:.1f}% of LCC {lcc}), {row['xfact']:,.0f} acres")

    # Key findings
    report_lines.append("\n" + "=" * 80)
    report_lines.append("KEY FINDINGS")
    report_lines.append("-" * 40)

    # Find optimal LCC for CRP
    lcc_rates = baseline_data.groupby('lcc')['enters_crp'].mean() * 100
    optimal_lcc = lcc_rates.idxmax()
    report_lines.append(f"1. Optimal LCC for CRP enrollment: {optimal_lcc} ({lcc_rates[optimal_lcc]:.1f}% rate)")

    # Primary source
    primary_source = baseline_crp['startuse'].mode().values[0] if len(baseline_crp) > 0 else None
    if primary_source:
        source_name = MODELING_CODES.get(primary_source, f"Code {primary_source}")
        source_pct = (baseline_crp['startuse'] == primary_source).mean() * 100
        report_lines.append(f"2. Primary CRP source: {source_name} ({source_pct:.1f}% of CRP)")

    # Scenario comparison
    baseline_acres = baseline_data[baseline_data['enters_crp']]['xfact'].sum()
    high_payment_acres = results_dict['high_payment'][results_dict['high_payment']['enters_crp']]['xfact'].sum()
    increase_pct = (high_payment_acres - baseline_acres) / baseline_acres * 100 if baseline_acres > 0 else 0
    report_lines.append(f"3. High payment scenario increases CRP by {increase_pct:.1f}%")

    # Write report
    report_text = "\n".join(report_lines)

    output_file = 'crp_simulation_report.txt'
    with open(output_file, 'w') as f:
        f.write(report_text)

    rprint(f"  ✓ Report saved to {output_file}")

    return report_text


def main():
    """Main execution function."""
    console.print(Panel.fit(
        "[bold green]CRP Enrollment Simulation[/bold green]\n"
        "Simulating Conservation Reserve Program enrollment using real NRI data\n"
        "Based on Land Capability Class (LCC) characteristics",
        title="Simulation Start"
    ))

    # Load data
    data = load_and_prepare_data()

    # Analyze LCC distribution
    rprint("\n[bold cyan]Land Capability Class Distribution[/bold cyan]")
    lcc_dist = data['lcc'].value_counts().sort_index()
    for lcc, count in lcc_dist.items():
        pct = count / len(data) * 100
        rprint(f"  LCC {lcc}: {count:,} parcels ({pct:.1f}%)")

    # Run simulations for different scenarios
    scenarios = ['baseline', 'high_payment', 'conservation_focus']
    results = {}

    for scenario in scenarios:
        scenario_data = simulate_crp_enrollment(data.copy(), scenario=scenario)
        results[scenario] = scenario_data

        # Analyze by LCC for this scenario
        if scenario == 'baseline':
            analyze_crp_by_lcc(scenario_data)

    # Create visualizations
    create_visualizations(results)

    # Generate report
    report = generate_report(results)

    # Final summary
    console.print(Panel.fit(
        "[bold green]Simulation Complete![/bold green]\n\n"
        "Key Results:\n"
        f"• Baseline CRP enrollment: {results['baseline']['enters_crp'].mean() * 100:.2f}%\n"
        f"• High payment enrollment: {results['high_payment']['enters_crp'].mean() * 100:.2f}%\n"
        f"• Conservation focus enrollment: {results['conservation_focus']['enters_crp'].mean() * 100:.2f}%\n\n"
        "Outputs generated:\n"
        "• crp_simulation_results.png - Comprehensive visualizations\n"
        "• crp_simulation_report.txt - Detailed analysis report\n\n"
        "The simulation successfully demonstrates CRP enrollment patterns\n"
        "based on Land Capability Classes in real NRI data.",
        title="Summary"
    ))

    return results


if __name__ == "__main__":
    results = main()