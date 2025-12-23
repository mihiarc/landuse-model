"""
NRI BROAD Land Cover/Use Codes
Based on 2017 NRI CSV File Layout

This module defines the standard NRI BROAD codes for land use classification.
"""

# NRI BROAD Land Use Codes
NRI_BROAD_CODES = {
    1: "Cultivated cropland",
    2: "Noncultivated cropland",
    3: "Pastureland",
    4: "Rangeland",
    5: "Forest land",
    6: "Minor land",
    7: "Urban and built-up land",
    8: "Rural transportation",
    9: "Small water areas",
    10: "Large water areas",
    11: "Federal land",
    12: "Conservation Reserve Program (CRP)"
}

# Simplified mapping for modeling (focusing on major land uses)
# We'll focus on the main agricultural and forest transitions
MODELING_CODES = {
    1: "Crop",        # Cultivated cropland
    3: "Pasture",     # Pastureland
    5: "Forest",      # Forest land
    7: "Urban",       # Urban and built-up land
    12: "CRP"         # Conservation Reserve Program
}

# Reverse mapping for easy lookup
CODE_TO_NAME = {
    "crop": 1,
    "cultivated": 1,
    "pasture": 3,
    "pastureland": 3,
    "forest": 5,
    "forestland": 5,
    "urban": 7,
    "built-up": 7,
    "crp": 12,
    "conservation": 12
}

def get_land_use_name(code: int) -> str:
    """Get the land use name for a given NRI BROAD code."""
    return NRI_BROAD_CODES.get(code, f"Unknown ({code})")

def get_simplified_name(code: int) -> str:
    """Get the simplified modeling name for a given code."""
    return MODELING_CODES.get(code, "Other")

def get_code_from_name(name: str) -> int:
    """Get the NRI BROAD code from a land use name."""
    name_lower = name.lower()
    return CODE_TO_NAME.get(name_lower, None)

# Land quality relationships
# Higher LCC = Better quality land for agriculture
LAND_QUALITY_PREFERENCES = {
    1: "high",     # Crop prefers high quality (high LCC) land
    3: "medium",   # Pasture can use medium quality land
    5: "low",      # Forest typically on lower quality (low LCC) land
    7: "any"       # Urban development less sensitive to agricultural quality
}


# =============================================================================
# 4-Category Model (Crop-Pasture-Urban with Irrigation Split)
# =============================================================================

# 4-category land use codes for crop/pasture/urban model
NRI_4CAT_CODES = {
    1: "Irrigated cropland",
    2: "Non-irrigated cropland",
    3: "Pastureland",
    4: "Urban and built-up land",
}

MODELING_CODES_4CAT = {
    1: "CR_IRR",   # Irrigated cropland
    2: "CR_DRY",   # Non-irrigated cropland (dryland)
    3: "PS",       # Pasture
    4: "UR",       # Urban
}

# Reverse mapping for 4-category model
CODE_TO_NAME_4CAT = {
    "cr_irr": 1,
    "irrigated": 1,
    "irrigated_crop": 1,
    "cr_dry": 2,
    "dryland": 2,
    "nonirrigated": 2,
    "nonirrigated_crop": 2,
    "ps": 3,
    "pasture": 3,
    "ur": 4,
    "urban": 4,
}

# Net returns column names for 4-category model
NR_COLUMNS_4CAT = ['nr_cr_irr', 'nr_cr_dry', 'nr_ps', 'nr_ur']

# Land quality preferences for 4-category model
LAND_QUALITY_PREFERENCES_4CAT = {
    1: "high",     # Irrigated crop - best land, with irrigation investment
    2: "high",     # Non-irrigated crop - still prefers good quality land
    3: "medium",   # Pasture can use medium quality land
    4: "any",      # Urban development less sensitive to agricultural quality
}


# =============================================================================
# 5-Category Model (Combined Cropland-Pasture-Range-Forest-Urban)
# =============================================================================

# 5-category land use codes (combined cropland)
NRI_5CAT_CODES = {
    1: "Cropland",
    2: "Pastureland",
    3: "Rangeland",
    4: "Forest land",
    5: "Urban and built-up land",
}

MODELING_CODES_5CAT = {
    1: "CR",       # Combined cropland (irrigated + non-irrigated)
    2: "PS",       # Pasture
    3: "RG",       # Rangeland
    4: "FR",       # Forest
    5: "UR",       # Urban
}

# Reverse mapping for 5-category model
CODE_TO_NAME_5CAT = {
    "cr": 1,
    "crop": 1,
    "cropland": 1,
    "ps": 2,
    "pasture": 2,
    "rg": 3,
    "range": 3,
    "rangeland": 3,
    "fr": 4,
    "forest": 4,
    "forestland": 4,
    "ur": 5,
    "urban": 5,
}

# Net returns column names for 5-category model
# Note: Only crop, pasture, and urban have net returns; range and forest use LCC only
NR_COLUMNS_5CAT = ['nr_cr', 'nr_ps', 'nr_ur']

# Which land uses have net returns data (for mixed specification) - 5-cat
USES_WITH_NET_RETURNS_5CAT = {1, 2, 5}  # CR, PS, UR
USES_LCC_ONLY_5CAT = {3, 4}  # RG, FR

# Land quality preferences for 5-category model
LAND_QUALITY_PREFERENCES_5CAT = {
    1: "high",     # Cropland prefers good quality land
    2: "medium",   # Pasture can use medium quality land
    3: "low",      # Rangeland - typically marginal land
    4: "low",      # Forest - typically on less suitable ag land
    5: "any",      # Urban development less sensitive to agricultural quality
}


def nri_to_5cat(broad_code: int, irrtyp: int = None) -> int:
    """
    Convert NRI BROAD code to 5-category code (combined cropland).

    Args:
        broad_code: NRI BROAD land use code (1=crop, 3=pasture, 4=range, 5=forest, 7=urban)
        irrtyp: Irrigation type (ignored - cropland is combined)

    Returns:
        5-category code (1-5) or None if not in categories
    """
    if broad_code == 1:  # Cultivated cropland
        return 1  # Combined cropland
    elif broad_code == 3:  # Pastureland
        return 2
    elif broad_code == 4:  # Rangeland
        return 3
    elif broad_code == 5:  # Forest
        return 4
    elif broad_code == 7:  # Urban
        return 5
    else:
        return None


def get_5cat_name(code: int) -> str:
    """Get the 5-category name for a given code."""
    return MODELING_CODES_5CAT.get(code, f"Unknown ({code})")


def get_5cat_full_name(code: int) -> str:
    """Get the full descriptive name for a 5-category code."""
    return NRI_5CAT_CODES.get(code, f"Unknown ({code})")


def nri_to_4cat(broad_code: int, irrtyp: int) -> int:
    """
    Convert NRI BROAD code + irrigation type to 4-category code.

    Args:
        broad_code: NRI BROAD land use code (1=crop, 3=pasture, 7=urban)
        irrtyp: Irrigation type (0 = non-irrigated, >0 = irrigated)

    Returns:
        4-category code (1=CR_IRR, 2=CR_DRY, 3=PS, 4=UR) or None if not in categories
    """
    if broad_code == 1:  # Cultivated cropland
        return 1 if irrtyp > 0 else 2  # Irrigated vs dryland
    elif broad_code == 3:  # Pastureland
        return 3
    elif broad_code == 7:  # Urban
        return 4
    else:
        return None


def get_4cat_name(code: int) -> str:
    """Get the 4-category name for a given code."""
    return MODELING_CODES_4CAT.get(code, f"Unknown ({code})")


def get_4cat_full_name(code: int) -> str:
    """Get the full descriptive name for a 4-category code."""
    return NRI_4CAT_CODES.get(code, f"Unknown ({code})")