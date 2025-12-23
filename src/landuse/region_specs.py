"""
Region-specific model specifications for land use transition models.

Each region may require a different specification based on:
- Data variation available in that region
- Economic validity of coefficient signs
- Convergence behavior

Specification format:
    'start_use': 'variables'

Where variables can be:
    'lcc'                    - LCC only
    'lcc + nr_ur'            - LCC + urban net returns
    'lcc + nr_ps'            - LCC + pasture net returns
    'lcc + nr_ur + nr_ps'    - LCC + urban + pasture net returns
    etc.

Note: Urban start is never modeled (urban development is irreversible).
"""

# Default specification (used when region not specified)
# Based on analysis: urban returns (nr_ur) show correct signs most consistently
DEFAULT_SPEC = {
    'cr_irr': 'lcc + nr_ur',
    'cr_dry': 'lcc + nr_ur',
    'pasture': 'lcc + nr_ur',
    'range': 'lcc',
    'forest': 'lcc',
}

# Region-specific specifications
# These override the default for specific regions based on data analysis

REGION_SPECS = {
    # Southeast - default spec works well, all signs correct
    'SE': {
        'cr_irr': 'lcc + nr_ur',
        'cr_dry': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Delta - pasture has violation with nr_ur, use LCC only
    'DL': {
        'cr_irr': 'lcc + nr_ur',
        'cr_dry': 'lcc + nr_ur',
        'pasture': 'lcc',  # nr_ur has wrong sign in DL
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Northern Plains - cr_irr has violation with nr_ur
    'NP': {
        'cr_irr': 'lcc',  # nr_ur has wrong sign in NP
        'cr_dry': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Corn Belt - cr_irr has violation with nr_ur, no rangeland
    'CB': {
        'cr_irr': 'lcc',  # nr_ur has wrong sign in CB
        'cr_dry': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',  # Will be skipped (no data)
        'forest': 'lcc',
    },

    # Appalachian - cr_irr has violation, no rangeland
    'AP': {
        'cr_irr': 'lcc',  # nr_ur has wrong sign in AP (only 6 urban transitions)
        'cr_dry': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Lake States - no rangeland
    'LS': {
        'cr_irr': 'lcc + nr_ur',
        'cr_dry': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Northeast - cr_dry has violation, no rangeland
    'NE': {
        'cr_irr': 'lcc + nr_ur',
        'cr_dry': 'lcc',  # nr_ur has wrong sign in NE
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Mountain - cr_irr and forest have violations
    'MT': {
        'cr_irr': 'lcc',  # nr_ur has wrong sign in MT
        'cr_dry': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',  # nr_ur has wrong sign in MT
    },

    # Pacific Coast - cr_dry has violation, forest has convergence issues
    'PC': {
        'cr_irr': 'lcc + nr_ur',
        'cr_dry': 'lcc',  # nr_ur has wrong sign in PC
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',  # Convergence issues with nr_ur
    },

    # Southern Plains - cr_irr has violation
    'SP': {
        'cr_irr': 'lcc',  # nr_ur has wrong sign in SP
        'cr_dry': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },
}


def get_region_spec(region: str) -> dict:
    """Get the specification for a specific region."""
    return REGION_SPECS.get(region, DEFAULT_SPEC)


def get_all_region_specs() -> dict:
    """Get all region specifications."""
    return REGION_SPECS
