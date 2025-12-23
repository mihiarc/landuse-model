"""
Region-specific model specifications for 5-category land use transition model.

Each region may require a different specification based on:
- Data variation available in that region
- Economic validity of coefficient signs
- Convergence behavior

Categories:
    CR (1): Cropland (combined irrigated + non-irrigated)
    PS (2): Pasture
    RG (3): Rangeland
    FR (4): Forest
    UR (5): Urban (excluded from estimation - irreversible)

Specification format:
    'start_use': 'variables'

Where variables can be:
    'lcc'             - LCC only
    'lcc + nr_ur'     - LCC + urban net returns

Note: Urban start is never modeled (urban development is irreversible).
"""

# RPA Subregion names
RPA_SUBREGIONS = {
    'NE': 'Northeast',
    'LS': 'Lake States',
    'CB': 'Corn Belt',
    'NP': 'Northern Plains',
    'AP': 'Appalachian',
    'SE': 'Southeast',
    'DL': 'Delta',
    'SP': 'Southern Plains',
    'MT': 'Mountain',
    'PC': 'Pacific Coast',
}

# Default specification (used when region not specified)
# Based on analysis: all crop/pasture models show correct signs with nr_ur
DEFAULT_SPEC = {
    'crop': 'lcc + nr_ur',
    'pasture': 'lcc + nr_ur',
    'range': 'lcc',
    'forest': 'lcc',
}

# Region-specific specifications
# Based on coefficient sign analysis - all 18 models passed validation
REGION_SPECS = {
    # Southeast - all signs correct
    'SE': {
        'crop': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Delta - all signs correct
    'DL': {
        'crop': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Northern Plains - all signs correct
    'NP': {
        'crop': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Corn Belt - all signs correct, no rangeland
    'CB': {
        'crop': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Appalachian - all signs correct, no rangeland
    'AP': {
        'crop': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Lake States - all signs correct, no rangeland
    'LS': {
        'crop': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Northeast - all signs correct, no rangeland
    'NE': {
        'crop': 'lcc + nr_ur',
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Mountain - crop uses LCC only (small negative coef with nr_ur)
    'MT': {
        'crop': 'lcc',  # nr_ur coef was -0.025
        'pasture': 'lcc + nr_ur',
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Pacific Coast - pasture uses LCC only (essentially zero coef)
    'PC': {
        'crop': 'lcc + nr_ur',
        'pasture': 'lcc',  # nr_ur coef was ~0
        'range': 'lcc',
        'forest': 'lcc',
    },

    # Southern Plains - all signs correct
    'SP': {
        'crop': 'lcc + nr_ur',
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
