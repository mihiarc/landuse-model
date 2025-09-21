"""
Pytest configuration and fixtures for landuse-modeling tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def sample_georef_data() -> pd.DataFrame:
    """Create sample geographic reference data."""
    np.random.seed(42)
    n_counties = 100

    fips_values = list(range(1001, 1001 + n_counties))
    return pd.DataFrame({
        'county_fips': fips_values,
        'fips': fips_values,  # Include both for compatibility
        'subregion': np.random.choice(['NO', 'SO'], n_counties),
        'region': np.random.choice(['NO', 'SO'], n_counties),
        'state': np.random.choice(['AL', 'GA', 'FL', 'NC', 'SC'], n_counties)
    })


@pytest.fixture
def sample_net_returns_data() -> pd.DataFrame:
    """Create sample net returns data."""
    np.random.seed(42)
    n_obs = 1000

    return pd.DataFrame({
        'fips': np.random.choice(range(1001, 1101), n_obs),
        'year': np.random.choice([2008, 2009, 2010], n_obs),
        'nr_cr': np.random.normal(500, 100, n_obs),  # Crop net returns
        'nr_ps': np.random.normal(200, 50, n_obs),   # Pasture net returns
        'nr_fr': np.random.normal(100, 30, n_obs),   # Forest net returns
        'nr_ur': np.random.normal(1000, 200, n_obs)  # Urban net returns
    })


@pytest.fixture
def sample_start_data() -> pd.DataFrame:
    """Create sample starting land use data."""
    np.random.seed(42)
    n_obs = 800

    return pd.DataFrame({
        'fips': np.random.choice(range(1001, 1101), n_obs),
        'year': np.random.choice([2010, 2011, 2012], n_obs),
        'riad_id': range(1, n_obs + 1),
        'lcc': np.random.choice([1, 2, 3, 4], n_obs),  # Land capability class
        'enduse': np.random.choice([1, 2, 3, 4], n_obs),  # End use (crop, pasture, forest, urban)
        'xfact': np.random.uniform(0.5, 2.0, n_obs),  # Expansion factor
        'startuse': np.random.choice([1, 2, 3], n_obs),  # Starting use
        'region': np.random.choice(['NO', 'SO'], n_obs)
    })


@pytest.fixture
def sample_climate_impact_data() -> pd.DataFrame:
    """Create sample climate change impact data."""
    np.random.seed(42)
    n_counties = 50

    return pd.DataFrame({
        'fips': range(1001, 1001 + n_counties),
        'crnr_obs': np.random.normal(500, 100, n_counties),
        'crnr_impact': np.random.normal(0, 50, n_counties),
        'frnr_obs': np.random.normal(100, 30, n_counties),
        'frnr_impact': np.random.normal(0, 15, n_counties),
        'urnr_obs': np.random.normal(1000, 200, n_counties),
        'urnr_impact': np.random.normal(0, 100, n_counties)
    })


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create sample configuration for testing."""
    return {
        "output_dir": "test_results",
        "run_exploration": True,
        "run_estimation": True,
        "run_marginal_effects": True,
        "run_climate_impact": False,  # Skip for basic tests
        "georef_file": "test_georef.csv",
        "nr_data_file": "test_nr_data.csv",
        "start_crop_file": "test_start_crop.csv",
        "start_pasture_file": "test_start_pasture.csv",
        "start_forest_file": "test_start_forest.csv",
        "years": [2010, 2011, 2012]
    }


@pytest.fixture
def temp_test_files(sample_georef_data, sample_net_returns_data, sample_start_data):
    """Create temporary test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Save test data files
        georef_file = tmp_path / "test_georef.csv"
        nr_file = tmp_path / "test_nr_data.csv"
        crop_file = tmp_path / "test_start_crop.csv"
        pasture_file = tmp_path / "test_start_pasture.csv"
        forest_file = tmp_path / "test_start_forest.csv"

        sample_georef_data.to_csv(georef_file, index=False)
        sample_net_returns_data.to_csv(nr_file, index=False)
        sample_start_data.to_csv(crop_file, index=False)
        sample_start_data.to_csv(pasture_file, index=False)
        sample_start_data.to_csv(forest_file, index=False)

        yield {
            'tmp_dir': tmp_path,
            'georef_file': str(georef_file),
            'nr_file': str(nr_file),
            'crop_file': str(crop_file),
            'pasture_file': str(pasture_file),
            'forest_file': str(forest_file)
        }


@pytest.fixture
def known_model_coefficients() -> np.ndarray:
    """Provide known model coefficients for testing calculations."""
    # These are example coefficients for testing mathematical accuracy
    return np.array([
        0.5,   # forest intercept
        -0.1,  # forest lcc coefficient
        0.3,   # pasture intercept
        -0.05, # pasture lcc coefficient
        1.2,   # urban intercept
        -0.2,  # urban lcc coefficient
        0.001, # crop net returns coefficient
        0.0005,# forest net returns coefficient
        0.0008,# pasture net returns coefficient
        0.0015 # urban net returns coefficient
    ])


@pytest.fixture
def numerical_tolerance() -> float:
    """Standard numerical tolerance for floating point comparisons."""
    return 1e-6


@pytest.fixture
def regression_tolerance() -> float:
    """Tolerance for regression testing against R outputs."""
    return 1e-4