"""
CRP Tree Planting Practices Analysis Module

This module analyzes tree-based Conservation Reserve Program practices,
including riparian buffers, windbreaks, and forest establishment.
Tree-based CRP practices provide enhanced environmental benefits,
particularly for carbon sequestration and wildlife habitat.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# CRP Practice Types with tree planting components
CRP_TREE_PRACTICES = {
    # Riparian and Buffer Practices
    'CP21': {
        'name': 'Filter Strips',
        'tree_component': False,
        'description': 'Grass filter strips along water bodies'
    },
    'CP22': {
        'name': 'Riparian Forest Buffer',
        'tree_component': True,
        'tree_density': 'high',
        'carbon_rate': 3.5,  # tons CO2/acre/year
        'description': 'Trees planted along streams and water bodies'
    },
    'CP23': {
        'name': 'Wetland Restoration',
        'tree_component': 'partial',
        'tree_density': 'low',
        'carbon_rate': 1.5,
        'description': 'Wetland with some woody vegetation'
    },
    'CP23A': {
        'name': 'Wetland Restoration - Non-Floodplain',
        'tree_component': 'partial',
        'tree_density': 'low',
        'carbon_rate': 1.2,
        'description': 'Upland wetland restoration'
    },

    # Tree Planting Practices
    'CP3': {
        'name': 'Tree Planting',
        'tree_component': True,
        'tree_density': 'high',
        'carbon_rate': 4.0,
        'description': 'General tree planting for conservation'
    },
    'CP3A': {
        'name': 'Hardwood Tree Planting',
        'tree_component': True,
        'tree_density': 'high',
        'carbon_rate': 4.5,
        'description': 'Hardwood forest establishment'
    },
    'CP11': {
        'name': 'Vegetative Cover - Trees',
        'tree_component': True,
        'tree_density': 'medium',
        'carbon_rate': 3.0,
        'description': 'Tree cover for wildlife habitat'
    },

    # Windbreaks and Shelterbelts
    'CP5A': {
        'name': 'Field Windbreak',
        'tree_component': True,
        'tree_density': 'linear',
        'carbon_rate': 2.0,
        'description': 'Single or multiple row windbreaks'
    },
    'CP16A': {
        'name': 'Shelterbelt Establishment',
        'tree_component': True,
        'tree_density': 'linear',
        'carbon_rate': 2.5,
        'description': 'Multiple row windbreaks for farmstead protection'
    },
    'CP17A': {
        'name': 'Living Snow Fence',
        'tree_component': True,
        'tree_density': 'linear',
        'carbon_rate': 1.8,
        'description': 'Trees/shrubs to control snow drifting'
    },

    # Wildlife Habitat with Trees
    'CP4D': {
        'name': 'Wildlife Habitat - Trees',
        'tree_component': True,
        'tree_density': 'medium',
        'carbon_rate': 3.2,
        'description': 'Permanent wildlife habitat with trees'
    },
    'CP12': {
        'name': 'Wildlife Food Plot',
        'tree_component': False,
        'description': 'Annual food plots for wildlife'
    },

    # Grass Practices (for comparison)
    'CP1': {
        'name': 'Permanent Introduced Grasses',
        'tree_component': False,
        'carbon_rate': 0.8,
        'description': 'Non-native grass establishment'
    },
    'CP2': {
        'name': 'Permanent Native Grasses',
        'tree_component': False,
        'carbon_rate': 1.0,
        'description': 'Native grass establishment'
    },
    'CP10': {
        'name': 'Vegetative Cover - Grass',
        'tree_component': False,
        'carbon_rate': 0.9,
        'description': 'Grass cover already established'
    }
}


@dataclass
class TreePracticeMetrics:
    """Metrics for tree-based CRP practices."""
    practice_code: str
    practice_name: str
    acres_enrolled: float
    carbon_sequestration_annual: float  # tons CO2/year
    carbon_sequestration_total: float   # tons CO2 over contract
    establishment_cost: float
    annual_payment: float
    cost_per_ton_co2: float
    wildlife_habitat_score: float  # 0-100
    water_quality_score: float     # 0-100


def classify_crp_practices(data: pd.DataFrame,
                           practice_column: str = 'practice_code') -> pd.DataFrame:
    """
    Classify CRP practices as tree-based or non-tree based.

    Parameters:
    -----------
    data : pd.DataFrame
        CRP enrollment data with practice codes
    practice_column : str
        Column name containing practice codes

    Returns:
    --------
    pd.DataFrame
        Data with additional classification columns
    """
    data = data.copy()

    # Add practice classification
    data['has_trees'] = False
    data['tree_density'] = 'none'
    data['carbon_rate'] = 0.8  # Default for grass
    data['practice_name'] = 'Unknown'

    for code, info in CRP_TREE_PRACTICES.items():
        mask = data[practice_column] == code
        if mask.any():
            data.loc[mask, 'practice_name'] = info['name']

            if info.get('tree_component') == True:
                data.loc[mask, 'has_trees'] = True
                data.loc[mask, 'tree_density'] = info.get('tree_density', 'medium')
            elif info.get('tree_component') == 'partial':
                data.loc[mask, 'has_trees'] = True
                data.loc[mask, 'tree_density'] = info.get('tree_density', 'low')

            if 'carbon_rate' in info:
                data.loc[mask, 'carbon_rate'] = info['carbon_rate']

    # Create practice categories
    data['practice_category'] = 'Other'
    data.loc[data['practice_name'].str.contains('Buffer|Riparian', na=False),
             'practice_category'] = 'Riparian/Buffer'
    data.loc[data['practice_name'].str.contains('Tree Planting|Hardwood', na=False),
             'practice_category'] = 'Forest Establishment'
    data.loc[data['practice_name'].str.contains('Windbreak|Shelterbelt|Snow Fence', na=False),
             'practice_category'] = 'Windbreak/Shelterbelt'
    data.loc[data['practice_name'].str.contains('Grass', na=False),
             'practice_category'] = 'Grassland'
    data.loc[data['practice_name'].str.contains('Wildlife', na=False),
             'practice_category'] = 'Wildlife Habitat'
    data.loc[data['practice_name'].str.contains('Wetland', na=False),
             'practice_category'] = 'Wetland'

    return data


def calculate_carbon_benefits(data: pd.DataFrame,
                              contract_length: int = 10,
                              discount_rate: float = 0.03) -> pd.DataFrame:
    """
    Calculate carbon sequestration benefits of CRP practices.

    Parameters:
    -----------
    data : pd.DataFrame
        CRP data with practice classifications
    contract_length : int
        CRP contract length in years
    discount_rate : float
        Discount rate for NPV calculations

    Returns:
    --------
    pd.DataFrame
        Data with carbon benefit calculations
    """
    data = data.copy()

    # Calculate annual carbon sequestration
    if 'xfact' in data.columns:
        data['carbon_annual_tons'] = data['xfact'] * data['carbon_rate']
    else:
        # Assume 100 acres if no acreage data
        data['carbon_annual_tons'] = 100 * data['carbon_rate']

    # Calculate total carbon over contract period
    data['carbon_total_tons'] = data['carbon_annual_tons'] * contract_length

    # Calculate NPV of carbon benefits (assuming $50/ton CO2)
    carbon_price = 50  # $/ton CO2
    data['carbon_value_annual'] = data['carbon_annual_tons'] * carbon_price

    # NPV calculation
    npv_factor = (1 - (1 + discount_rate) ** -contract_length) / discount_rate
    data['carbon_value_npv'] = data['carbon_value_annual'] * npv_factor

    # Additional benefits for tree practices
    data['growth_multiplier'] = 1.0
    tree_mask = data['has_trees'] == True

    # Trees increase carbon sequestration over time
    if tree_mask.any():
        # Simple growth curve: increases to 150% by year 10
        avg_growth = 1.25  # Average over contract period
        data.loc[tree_mask, 'growth_multiplier'] = avg_growth
        data.loc[tree_mask, 'carbon_total_tons'] *= avg_growth

    return data


def compare_practice_effectiveness(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compare effectiveness of tree vs non-tree CRP practices.

    Parameters:
    -----------
    data : pd.DataFrame
        CRP data with practice classifications and carbon calculations

    Returns:
    --------
    pd.DataFrame
        Comparison summary by practice type
    """
    # Group by tree/non-tree
    comparison = data.groupby('has_trees').agg({
        'xfact': 'sum',  # Total acres
        'carbon_annual_tons': 'sum',
        'carbon_total_tons': 'sum',
        'carbon_value_npv': 'sum',
        'practice_code': 'count'  # Number of enrollments
    }).rename(columns={'practice_code': 'enrollment_count'})

    # Calculate averages
    comparison['carbon_per_acre'] = (comparison['carbon_annual_tons'] /
                                     comparison['xfact'])
    comparison['avg_carbon_per_enrollment'] = (comparison['carbon_total_tons'] /
                                                comparison['enrollment_count'])

    # Group by practice category
    category_summary = data.groupby('practice_category').agg({
        'xfact': 'sum',
        'carbon_annual_tons': 'sum',
        'carbon_total_tons': 'sum',
        'has_trees': 'mean',  # Proportion with trees
        'practice_code': 'count'
    }).rename(columns={
        'practice_code': 'count',
        'has_trees': 'tree_proportion'
    })

    category_summary['carbon_rate_avg'] = (category_summary['carbon_annual_tons'] /
                                           category_summary['xfact'])

    return comparison, category_summary


def calculate_ecosystem_services(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive ecosystem service benefits.

    Parameters:
    -----------
    data : pd.DataFrame
        CRP data with practice classifications

    Returns:
    --------
    pd.DataFrame
        Data with ecosystem service scores
    """
    data = data.copy()

    # Initialize scores
    data['wildlife_score'] = 50  # Base score
    data['water_quality_score'] = 50
    data['soil_health_score'] = 50
    data['pollinator_score'] = 50

    # Tree practices provide enhanced wildlife habitat
    tree_mask = data['has_trees'] == True
    data.loc[tree_mask, 'wildlife_score'] = 80

    # Specific practice benefits
    riparian_mask = data['practice_category'] == 'Riparian/Buffer'
    data.loc[riparian_mask, 'water_quality_score'] = 95
    data.loc[riparian_mask, 'wildlife_score'] = 85

    forest_mask = data['practice_category'] == 'Forest Establishment'
    data.loc[forest_mask, 'wildlife_score'] = 90
    data.loc[forest_mask, 'soil_health_score'] = 75

    windbreak_mask = data['practice_category'] == 'Windbreak/Shelterbelt'
    data.loc[windbreak_mask, 'soil_health_score'] = 80
    data.loc[windbreak_mask, 'wildlife_score'] = 70

    grass_mask = data['practice_category'] == 'Grassland'
    data.loc[grass_mask, 'pollinator_score'] = 85
    data.loc[grass_mask, 'soil_health_score'] = 70

    wetland_mask = data['practice_category'] == 'Wetland'
    data.loc[wetland_mask, 'water_quality_score'] = 90
    data.loc[wetland_mask, 'wildlife_score'] = 95

    # Calculate overall ecosystem service score
    data['ecosystem_score'] = (data['wildlife_score'] +
                               data['water_quality_score'] +
                               data['soil_health_score'] +
                               data['pollinator_score']) / 4

    return data


def optimize_practice_selection(available_acres: float,
                               budget: float,
                               objectives: Dict[str, float] = None) -> Dict[str, float]:
    """
    Optimize CRP practice selection given constraints.

    Parameters:
    -----------
    available_acres : float
        Total acres available for enrollment
    budget : float
        Total budget available
    objectives : Dict[str, float]
        Weights for different objectives (carbon, wildlife, water)

    Returns:
    --------
    Dict[str, float]
        Recommended allocation by practice type
    """
    if objectives is None:
        objectives = {
            'carbon': 0.4,
            'wildlife': 0.3,
            'water_quality': 0.3
        }

    # Practice costs and benefits (simplified)
    practices = {
        'Riparian Forest Buffer': {
            'cost_per_acre': 150,
            'carbon_rate': 3.5,
            'wildlife_score': 85,
            'water_score': 95
        },
        'Tree Planting': {
            'cost_per_acre': 120,
            'carbon_rate': 4.0,
            'wildlife_score': 90,
            'water_score': 60
        },
        'Native Grass': {
            'cost_per_acre': 80,
            'carbon_rate': 1.0,
            'wildlife_score': 70,
            'water_score': 50
        },
        'Wetland Restoration': {
            'cost_per_acre': 200,
            'carbon_rate': 1.5,
            'wildlife_score': 95,
            'water_score': 90
        }
    }

    # Calculate benefit scores
    allocations = {}
    remaining_acres = available_acres
    remaining_budget = budget

    # Score each practice
    practice_scores = {}
    for name, info in practices.items():
        score = (objectives['carbon'] * info['carbon_rate'] / 4.0 +  # Normalize
                objectives['wildlife'] * info['wildlife_score'] / 100 +
                objectives['water_quality'] * info['water_score'] / 100)

        # Adjust for cost-effectiveness
        cost_effectiveness = score / (info['cost_per_acre'] / 100)
        practice_scores[name] = cost_effectiveness

    # Allocate based on scores (greedy approach)
    sorted_practices = sorted(practice_scores.items(), key=lambda x: x[1], reverse=True)

    for practice_name, score in sorted_practices:
        if remaining_acres <= 0 or remaining_budget <= 0:
            break

        practice = practices[practice_name]
        max_acres = min(
            remaining_acres,
            remaining_budget / practice['cost_per_acre'],
            available_acres * 0.4  # Max 40% in any one practice
        )

        allocations[practice_name] = max_acres
        remaining_acres -= max_acres
        remaining_budget -= max_acres * practice['cost_per_acre']

    return allocations


# Example usage
if __name__ == "__main__":
    print("CRP Tree Planting Practices Module")
    print("-" * 40)

    # Show available practices
    print("\nTree-based CRP Practices:")
    for code, info in CRP_TREE_PRACTICES.items():
        if info.get('tree_component'):
            print(f"  {code}: {info['name']}")
            if 'carbon_rate' in info:
                print(f"    Carbon sequestration: {info['carbon_rate']} tons CO2/acre/year")

    # Example optimization
    print("\nExample Practice Optimization:")
    print("Available: 1000 acres, $150,000 budget")

    allocation = optimize_practice_selection(1000, 150000)
    print("\nRecommended allocation:")
    for practice, acres in allocation.items():
        print(f"  {practice}: {acres:.0f} acres")