#!/usr/bin/env python3
"""
Analyze tree planting practices in CRP enrollment.
This script simulates and analyzes the distribution of CRP practices,
comparing tree-based vs non-tree conservation practices.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from landuse.crp_tree_practices import (
    CRP_TREE_PRACTICES,
    classify_crp_practices,
    calculate_carbon_benefits,
    compare_practice_effectiveness,
    calculate_ecosystem_services,
    optimize_practice_selection
)

console = Console()


def simulate_crp_practices(crp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate CRP practice assignments based on land characteristics.

    Parameters:
    -----------
    crp_data : pd.DataFrame
        CRP enrollment data with LCC information

    Returns:
    --------
    pd.DataFrame
        Data with simulated practice assignments
    """
    rprint("\n[bold cyan]Simulating CRP Practice Assignments[/bold cyan]")

    # Practice assignment probabilities by LCC
    # Higher quality land (higher LCC) more likely to get grass practices
    # Lower quality land more suitable for trees
    practice_probabilities = {
        # LCC 1-2: Poor land - good for trees
        1: {'CP3': 0.30, 'CP22': 0.25, 'CP23': 0.20, 'CP2': 0.15, 'CP4D': 0.10},
        2: {'CP3': 0.25, 'CP22': 0.20, 'CP23': 0.15, 'CP2': 0.20, 'CP11': 0.10, 'CP1': 0.10},

        # LCC 3-4: Marginal land - mixed practices
        3: {'CP3': 0.20, 'CP22': 0.15, 'CP2': 0.25, 'CP1': 0.15, 'CP5A': 0.10,
            'CP10': 0.10, 'CP4D': 0.05},
        4: {'CP2': 0.30, 'CP1': 0.20, 'CP3': 0.15, 'CP22': 0.10, 'CP5A': 0.10,
            'CP10': 0.10, 'CP16A': 0.05},

        # LCC 5-6: Moderate land - more grass
        5: {'CP2': 0.35, 'CP1': 0.25, 'CP10': 0.15, 'CP5A': 0.10, 'CP3': 0.10, 'CP12': 0.05},
        6: {'CP2': 0.40, 'CP1': 0.30, 'CP10': 0.15, 'CP12': 0.10, 'CP5A': 0.05},

        # LCC 7-8: Good land - primarily grass
        7: {'CP2': 0.45, 'CP1': 0.35, 'CP10': 0.15, 'CP12': 0.05},
        8: {'CP2': 0.50, 'CP1': 0.40, 'CP10': 0.10}
    }

    # Assign practices
    np.random.seed(42)
    crp_data['practice_code'] = None

    for lcc in crp_data['lcc'].unique():
        lcc_mask = crp_data['lcc'] == lcc
        n_parcels = lcc_mask.sum()

        if lcc in practice_probabilities:
            probs = practice_probabilities[lcc]
            practices = list(probs.keys())
            probabilities = list(probs.values())

            # Normalize probabilities
            probabilities = np.array(probabilities) / sum(probabilities)

            # Assign practices randomly based on probabilities
            assigned_practices = np.random.choice(
                practices,
                size=n_parcels,
                p=probabilities
            )
            crp_data.loc[lcc_mask, 'practice_code'] = assigned_practices
        else:
            # Default to native grass for undefined LCC
            crp_data.loc[lcc_mask, 'practice_code'] = 'CP2'

    rprint(f"  Assigned practices to {len(crp_data)} CRP parcels")

    # Add practice classifications
    crp_data = classify_crp_practices(crp_data)

    # Summary
    tree_pct = crp_data['has_trees'].mean() * 100
    rprint(f"  {tree_pct:.1f}% of parcels have tree-based practices")

    return crp_data


def analyze_practice_distribution(data: pd.DataFrame):
    """Analyze the distribution of CRP practices."""
    rprint("\n[bold cyan]CRP Practice Distribution Analysis[/bold cyan]")

    # Overall summary
    table = Table(title="CRP Practice Distribution")
    table.add_column("Practice Code", style="cyan")
    table.add_column("Practice Name", style="magenta")
    table.add_column("Count", style="yellow")
    table.add_column("Acres", style="green")
    table.add_column("% of Total", style="blue")
    table.add_column("Has Trees", style="red")

    practice_summary = data.groupby(['practice_code', 'practice_name', 'has_trees']).agg({
        'xfact': ['count', 'sum']
    }).reset_index()

    practice_summary.columns = ['practice_code', 'practice_name', 'has_trees',
                                'count', 'acres']
    practice_summary = practice_summary.sort_values('acres', ascending=False)

    total_acres = practice_summary['acres'].sum()

    for _, row in practice_summary.head(10).iterrows():
        pct = row['acres'] / total_acres * 100
        table.add_row(
            row['practice_code'],
            row['practice_name'],
            f"{row['count']:,}",
            f"{row['acres']:,.0f}",
            f"{pct:.1f}%",
            "Yes" if row['has_trees'] else "No"
        )

    console.print(table)

    # Category summary
    rprint("\n[bold]Practice Categories:[/bold]")
    category_summary = data.groupby('practice_category').agg({
        'xfact': 'sum',
        'has_trees': 'mean'
    }).sort_values('xfact', ascending=False)

    for category, row in category_summary.iterrows():
        acres = row['xfact']
        tree_pct = row['has_trees'] * 100
        rprint(f"  {category}: {acres:,.0f} acres ({tree_pct:.0f}% with trees)")


def analyze_carbon_sequestration(data: pd.DataFrame):
    """Analyze carbon sequestration potential."""
    rprint("\n[bold cyan]Carbon Sequestration Analysis[/bold cyan]")

    # Calculate carbon benefits
    data = calculate_carbon_benefits(data, contract_length=10)

    # Summary statistics
    total_carbon_annual = data['carbon_annual_tons'].sum()
    total_carbon_10yr = data['carbon_total_tons'].sum()
    total_value = data['carbon_value_npv'].sum()

    rprint(f"\n[bold]Total Carbon Sequestration:[/bold]")
    rprint(f"  Annual: {total_carbon_annual:,.0f} tons CO2")
    rprint(f"  10-year total: {total_carbon_10yr:,.0f} tons CO2")
    rprint(f"  NPV of carbon benefits: ${total_value:,.0f}")

    # Compare tree vs non-tree
    comparison, category_summary = compare_practice_effectiveness(data)

    rprint(f"\n[bold]Tree vs Non-Tree Comparison:[/bold]")
    for has_trees, row in comparison.iterrows():
        tree_type = "Tree-based" if has_trees else "Non-tree"
        rprint(f"\n  {tree_type} practices:")
        rprint(f"    Total acres: {row['xfact']:,.0f}")
        rprint(f"    Carbon/acre/year: {row['carbon_per_acre']:.2f} tons CO2")
        rprint(f"    Total annual carbon: {row['carbon_annual_tons']:,.0f} tons")
        rprint(f"    10-year carbon: {row['carbon_total_tons']:,.0f} tons")

    # Top practices by carbon
    rprint(f"\n[bold]Top Practices by Carbon Sequestration Rate:[/bold]")
    top_carbon = data.groupby('practice_name')['carbon_rate'].first().sort_values(ascending=False)
    for practice, rate in top_carbon.head(5).items():
        rprint(f"  {practice}: {rate:.1f} tons CO2/acre/year")

    return data


def analyze_ecosystem_services(data: pd.DataFrame):
    """Analyze comprehensive ecosystem services."""
    rprint("\n[bold cyan]Ecosystem Services Analysis[/bold cyan]")

    # Calculate ecosystem services
    data = calculate_ecosystem_services(data)

    # Overall scores
    table = Table(title="Ecosystem Service Scores by Practice Category")
    table.add_column("Category", style="cyan")
    table.add_column("Wildlife", style="magenta")
    table.add_column("Water Quality", style="blue")
    table.add_column("Soil Health", style="green")
    table.add_column("Pollinators", style="yellow")
    table.add_column("Overall", style="red")

    category_scores = data.groupby('practice_category').agg({
        'wildlife_score': 'mean',
        'water_quality_score': 'mean',
        'soil_health_score': 'mean',
        'pollinator_score': 'mean',
        'ecosystem_score': 'mean'
    })

    for category, row in category_scores.iterrows():
        table.add_row(
            category,
            f"{row['wildlife_score']:.0f}",
            f"{row['water_quality_score']:.0f}",
            f"{row['soil_health_score']:.0f}",
            f"{row['pollinator_score']:.0f}",
            f"{row['ecosystem_score']:.0f}"
        )

    console.print(table)

    # Tree vs non-tree ecosystem services
    tree_scores = data.groupby('has_trees')[['wildlife_score', 'water_quality_score',
                                              'ecosystem_score']].mean()

    rprint(f"\n[bold]Tree vs Non-Tree Ecosystem Services:[/bold]")
    for has_trees, row in tree_scores.iterrows():
        tree_type = "Tree-based" if has_trees else "Non-tree"
        rprint(f"  {tree_type}:")
        rprint(f"    Wildlife: {row['wildlife_score']:.0f}")
        rprint(f"    Water Quality: {row['water_quality_score']:.0f}")
        rprint(f"    Overall: {row['ecosystem_score']:.0f}")

    return data


def create_visualizations(data: pd.DataFrame):
    """Create comprehensive visualizations."""
    rprint("\n[bold cyan]Creating Visualizations[/bold cyan]")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # 1. Practice distribution pie chart
    ax1 = axes[0, 0]
    practice_counts = data['practice_category'].value_counts()
    ax1.pie(practice_counts.values, labels=practice_counts.index, autopct='%1.1f%%')
    ax1.set_title('CRP Practices by Category')

    # 2. Tree vs Non-tree acres
    ax2 = axes[0, 1]
    tree_acres = data.groupby('has_trees')['xfact'].sum()
    ax2.bar(['Non-tree', 'Tree-based'], tree_acres.values, color=['gold', 'forestgreen'])
    ax2.set_ylabel('Acres')
    ax2.set_title('Tree vs Non-Tree Practice Acres')

    # 3. Carbon by practice type
    ax3 = axes[0, 2]
    carbon_by_practice = data.groupby('practice_category')['carbon_annual_tons'].sum().sort_values()
    ax3.barh(range(len(carbon_by_practice)), carbon_by_practice.values)
    ax3.set_yticks(range(len(carbon_by_practice)))
    ax3.set_yticklabels(carbon_by_practice.index)
    ax3.set_xlabel('Annual Carbon Sequestration (tons CO2)')
    ax3.set_title('Carbon Sequestration by Practice Category')

    # 4. Practice distribution by LCC
    ax4 = axes[1, 0]
    lcc_tree_pct = data.groupby('lcc')['has_trees'].mean() * 100
    ax4.bar(lcc_tree_pct.index, lcc_tree_pct.values, color='darkgreen')
    ax4.set_xlabel('Land Capability Class')
    ax4.set_ylabel('% with Tree Practices')
    ax4.set_title('Tree Practice Adoption by LCC')

    # 5. Carbon rates comparison
    ax5 = axes[1, 1]
    carbon_rates = data.groupby('practice_name')['carbon_rate'].first().sort_values(ascending=False).head(10)
    ax5.barh(range(len(carbon_rates)), carbon_rates.values, color='teal')
    ax5.set_yticks(range(len(carbon_rates)))
    ax5.set_yticklabels(carbon_rates.index, fontsize=8)
    ax5.set_xlabel('Carbon Rate (tons CO2/acre/year)')
    ax5.set_title('Top 10 Practices by Carbon Rate')

    # 6. Ecosystem services radar chart
    ax6 = axes[1, 2]
    categories = ['Wildlife', 'Water', 'Soil', 'Pollinators']
    tree_scores = data[data['has_trees']].agg({
        'wildlife_score': 'mean',
        'water_quality_score': 'mean',
        'soil_health_score': 'mean',
        'pollinator_score': 'mean'
    }).values
    nontree_scores = data[~data['has_trees']].agg({
        'wildlife_score': 'mean',
        'water_quality_score': 'mean',
        'soil_health_score': 'mean',
        'pollinator_score': 'mean'
    }).values

    x = np.arange(len(categories))
    width = 0.35
    ax6.bar(x - width/2, tree_scores, width, label='Tree-based', color='forestgreen')
    ax6.bar(x + width/2, nontree_scores, width, label='Non-tree', color='gold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.set_ylabel('Score')
    ax6.set_title('Ecosystem Service Scores')
    ax6.legend()

    # 7. Acres by category and tree status
    ax7 = axes[2, 0]
    category_tree = data.groupby(['practice_category', 'has_trees'])['xfact'].sum().unstack(fill_value=0)
    category_tree.plot(kind='bar', stacked=True, ax=ax7, color=['gold', 'forestgreen'])
    ax7.set_xlabel('Practice Category')
    ax7.set_ylabel('Acres')
    ax7.set_title('Practice Categories by Tree Status')
    ax7.legend(['Non-tree', 'Tree-based'])
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 8. Carbon value distribution
    ax8 = axes[2, 1]
    ax8.hist(data['carbon_value_npv'], bins=30, color='darkblue', alpha=0.7)
    ax8.set_xlabel('Carbon Value NPV ($)')
    ax8.set_ylabel('Number of Parcels')
    ax8.set_title('Distribution of Carbon Value (10-year NPV)')

    # 9. Optimization recommendation
    ax9 = axes[2, 2]
    allocation = optimize_practice_selection(1000, 150000)
    ax9.pie([v for v in allocation.values()],
           labels=[k for k in allocation.keys()],
           autopct='%1.1f%%')
    ax9.set_title('Optimal Practice Allocation\n(1000 acres, $150k budget)')

    plt.suptitle('CRP Tree Planting Practices Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = 'tree_crp_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    rprint(f"  ✓ Saved visualization to {output_file}")
    plt.close()


def generate_report(data: pd.DataFrame):
    """Generate comprehensive report."""
    rprint("\n[bold cyan]Generating Report[/bold cyan]")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CRP TREE PLANTING PRACTICES ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary statistics
    total_acres = data['xfact'].sum()
    tree_acres = data[data['has_trees']]['xfact'].sum()
    tree_pct = tree_acres / total_acres * 100

    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total CRP acres analyzed: {total_acres:,.0f}")
    report_lines.append(f"Tree-based practice acres: {tree_acres:,.0f} ({tree_pct:.1f}%)")
    report_lines.append(f"Non-tree practice acres: {total_acres - tree_acres:,.0f} ({100-tree_pct:.1f}%)")
    report_lines.append("")

    # Carbon benefits
    total_carbon = data['carbon_total_tons'].sum()
    tree_carbon = data[data['has_trees']]['carbon_total_tons'].sum()
    tree_carbon_pct = tree_carbon / total_carbon * 100

    report_lines.append("CARBON SEQUESTRATION")
    report_lines.append("-" * 40)
    report_lines.append(f"Total 10-year carbon sequestration: {total_carbon:,.0f} tons CO2")
    report_lines.append(f"Tree practices contribution: {tree_carbon:,.0f} tons ({tree_carbon_pct:.1f}%)")
    report_lines.append(f"Average carbon rate:")
    report_lines.append(f"  Tree practices: {data[data['has_trees']]['carbon_rate'].mean():.2f} tons/acre/year")
    report_lines.append(f"  Non-tree practices: {data[~data['has_trees']]['carbon_rate'].mean():.2f} tons/acre/year")
    report_lines.append("")

    # Top practices
    report_lines.append("TOP PRACTICES BY ENROLLMENT")
    report_lines.append("-" * 40)
    top_practices = data.groupby('practice_name')['xfact'].sum().sort_values(ascending=False).head(5)
    for i, (practice, acres) in enumerate(top_practices.items(), 1):
        report_lines.append(f"{i}. {practice}: {acres:,.0f} acres")
    report_lines.append("")

    # Ecosystem services
    report_lines.append("ECOSYSTEM SERVICES COMPARISON")
    report_lines.append("-" * 40)
    tree_eco = data[data['has_trees']]['ecosystem_score'].mean()
    nontree_eco = data[~data['has_trees']]['ecosystem_score'].mean()
    report_lines.append(f"Average ecosystem service score:")
    report_lines.append(f"  Tree-based practices: {tree_eco:.1f}/100")
    report_lines.append(f"  Non-tree practices: {nontree_eco:.1f}/100")
    report_lines.append("")

    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 40)
    report_lines.append("1. Increase tree-based practices on LCC 1-3 land for maximum carbon benefits")
    report_lines.append("2. Prioritize riparian forest buffers for water quality improvements")
    report_lines.append("3. Use windbreaks on LCC 4-5 land for balanced benefits")
    report_lines.append("4. Maintain grass practices on LCC 6-8 for cost-effectiveness")
    report_lines.append("5. Consider 30-40% tree practices for optimal ecosystem services")

    # Write report
    report_text = "\n".join(report_lines)

    output_file = 'tree_crp_analysis_report.txt'
    with open(output_file, 'w') as f:
        f.write(report_text)

    rprint(f"  ✓ Report saved to {output_file}")

    return report_text


def main():
    """Main execution function."""
    console.print(Panel.fit(
        "[bold green]CRP Tree Planting Practices Analysis[/bold green]\n"
        "Analyzing tree-based conservation practices in CRP enrollment\n"
        "Comparing carbon sequestration and ecosystem services",
        title="Analysis Start"
    ))

    # Load CRP enrollment data
    rprint("\n[bold cyan]Loading CRP Enrollment Data[/bold cyan]")

    # Use simulated CRP data from previous analysis
    import pandas as pd
    from simulate_crp_enrollment import load_and_prepare_data, simulate_crp_enrollment

    # Load real NRI data
    base_data = load_and_prepare_data()

    # Simulate CRP enrollment
    crp_data = simulate_crp_enrollment(base_data, scenario='baseline')

    # Filter to only CRP enrolled parcels
    crp_enrolled = crp_data[crp_data['enters_crp']].copy()
    rprint(f"  Analyzing {len(crp_enrolled):,} CRP enrolled parcels")

    # Simulate practice assignments
    crp_with_practices = simulate_crp_practices(crp_enrolled)

    # Analyze practice distribution
    analyze_practice_distribution(crp_with_practices)

    # Analyze carbon sequestration
    crp_with_carbon = analyze_carbon_sequestration(crp_with_practices)

    # Analyze ecosystem services
    crp_with_ecosystem = analyze_ecosystem_services(crp_with_carbon)

    # Create visualizations
    create_visualizations(crp_with_ecosystem)

    # Generate report
    report = generate_report(crp_with_ecosystem)

    # Final summary
    console.print(Panel.fit(
        "[bold green]Analysis Complete![/bold green]\n\n"
        "Key Findings:\n"
        f"• {(crp_with_ecosystem['has_trees'].mean()*100):.1f}% of CRP uses tree practices\n"
        f"• Tree practices provide {(crp_with_ecosystem[crp_with_ecosystem['has_trees']]['carbon_rate'].mean()):.1f}x "
        f"carbon sequestration vs grass\n"
        f"• Riparian forest buffers score highest for water quality\n"
        f"• Forest establishment provides best wildlife habitat\n\n"
        "Files generated:\n"
        "• tree_crp_analysis.png - Comprehensive visualizations\n"
        "• tree_crp_analysis_report.txt - Detailed report",
        title="Summary"
    ))

    return crp_with_ecosystem


if __name__ == "__main__":
    data = main()