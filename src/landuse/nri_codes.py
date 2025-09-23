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