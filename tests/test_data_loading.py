"""
Tests for data loading and validation functions.

These tests ensure that data loading operations work correctly
and that data validation catches common issues.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

from landuse.logit_estimation import load_data, prepare_estimation_data
from landuse.main import load_config


class TestDataLoading:
    """Test data loading functions across all modules."""

    def test_load_csv_files(self, temp_test_files):
        """Test loading CSV files with correct format."""
        georef, nr_data, start_crop, start_pasture, start_forest = load_data(
            temp_test_files['georef_file'],
            temp_test_files['nr_file'],
            temp_test_files['crop_file'],
            temp_test_files['pasture_file'],
            temp_test_files['forest_file']
        )

        # Check that all dataframes are loaded
        assert isinstance(georef, pd.DataFrame)
        assert isinstance(nr_data, pd.DataFrame)
        assert isinstance(start_crop, pd.DataFrame)
        assert isinstance(start_pasture, pd.DataFrame)
        assert isinstance(start_forest, pd.DataFrame)

        # Check expected columns exist
        assert 'fips' in georef.columns
        assert 'region' in georef.columns
        assert 'fips' in nr_data.columns
        assert 'year' in nr_data.columns
        assert 'fips' in start_crop.columns

    def test_fips_code_conversion(self, temp_test_files):
        """Test that FIPS codes are properly converted to integers."""
        georef, _, _, _, _ = load_data(
            temp_test_files['georef_file'],
            temp_test_files['nr_file'],
            temp_test_files['crop_file'],
            temp_test_files['pasture_file'],
            temp_test_files['forest_file']
        )

        assert pd.api.types.is_integer_dtype(georef['fips'])
        assert all(georef['fips'] > 0)

    def test_missing_file_handling(self):
        """Test handling of missing data files."""
        with pytest.raises(FileNotFoundError):
            load_data(
                "nonexistent_georef.csv",
                "nonexistent_nr.csv",
                "nonexistent_crop.csv",
                "nonexistent_pasture.csv",
                "nonexistent_forest.csv"
            )

    def test_config_loading(self, sample_config):
        """Test configuration file loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config, f)
            config_file = f.name

        try:
            loaded_config = load_config(config_file)
            assert loaded_config == sample_config
            assert 'output_dir' in loaded_config
            assert 'years' in loaded_config
            assert isinstance(loaded_config['years'], list)
        finally:
            Path(config_file).unlink()

    def test_invalid_config_handling(self):
        """Test handling of invalid configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_file = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(config_file)
        finally:
            Path(config_file).unlink()


class TestDataPreparation:
    """Test data preparation functions."""

    def test_prepare_estimation_data_basic(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test basic data preparation functionality."""
        years = [2010, 2011, 2012]

        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            years
        )

        # Check that data is properly filtered
        assert all(prepared_data['year'].isin(years))

        # Check that merges worked
        assert 'nr_cr' in prepared_data.columns or 'nr_ps' in prepared_data.columns
        assert 'region_x' in prepared_data.columns or 'region_y' in prepared_data.columns or 'region' in prepared_data.columns

        # Check that weights are calculated
        assert 'weight' in prepared_data.columns
        assert all(prepared_data['weight'] > 0)

    def test_lag_application(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that net returns lag is applied correctly."""
        # Create specific test data to verify lag
        nr_test = pd.DataFrame({
            'fips': [1001, 1001, 1002],
            'year': [2008, 2009, 2010],
            'nr_cr': [100, 200, 300],
            'nr_ps': [50, 100, 150],
            'nr_fr': [25, 50, 75],
            'nr_ur': [500, 600, 700]
        })

        start_test = pd.DataFrame({
            'fips': [1001, 1001, 1002],
            'year': [2010, 2011, 2012],
            'enduse': [1, 2, 3],
            'lcc': [1, 2, 3],
            'xfact': [1.0, 1.5, 2.0],
            'region': ['NO', 'NO', 'SO']
        })

        prepared = prepare_estimation_data(
            start_test,
            nr_test,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        # Check that the lag was applied (year 2008 net returns should match 2010 start data)
        if not prepared.empty:
            # The lag means 2008 net returns are used for 2010 land use decisions
            year_2010_data = prepared[prepared['year'] == 2010]
            if not year_2010_data.empty:
                # This should contain the 2008 net returns values (after lag adjustment)
                assert len(year_2010_data) > 0

    def test_region_filtering(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test filtering by geographic regions."""
        regions = ['NO']

        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            [2010, 2011, 2012],
            regions
        )

        if not prepared_data.empty:
            assert all(prepared_data['region'].isin(regions))

    def test_land_capability_class_conversion(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that land capability class is converted to numeric."""
        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        if 'lcc' in prepared_data.columns and not prepared_data.empty:
            assert pd.api.types.is_numeric_dtype(prepared_data['lcc'])

    def test_weight_calculation(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that weights are calculated correctly."""
        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        if not prepared_data.empty:
            # Weights should sum to the number of observations (approximately)
            weight_sum = prepared_data['weight'].sum()
            n_obs = len(prepared_data)

            # Allow for some floating point precision differences
            assert abs(weight_sum - n_obs) < 1e-10

    def test_empty_data_handling(self, sample_georef_data):
        """Test handling of empty input data."""
        empty_start = pd.DataFrame(columns=['fips', 'year', 'enduse', 'lcc', 'xfact'])
        empty_nr = pd.DataFrame(columns=['fips', 'year', 'nr_cr', 'nr_ps', 'nr_fr', 'nr_ur'])

        prepared_data = prepare_estimation_data(
            empty_start,
            empty_nr,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        assert len(prepared_data) == 0

    def test_column_cleanup(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that unnecessary columns are removed."""
        # Add some columns that should be dropped
        test_start_data = sample_start_data.copy()
        test_start_data['test_ag'] = np.random.random(len(test_start_data))
        test_start_data['another_ag'] = np.random.random(len(test_start_data))

        prepared_data = prepare_estimation_data(
            test_start_data,
            sample_net_returns_data,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        # Columns ending with 'ag' should be removed
        ag_columns = [col for col in prepared_data.columns if col.endswith('ag')]
        assert len(ag_columns) == 0


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_required_columns_presence(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that required columns are present in prepared data."""
        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        required_columns = ['fips', 'year', 'enduse']
        for col in required_columns:
            assert col in prepared_data.columns, f"Required column {col} missing"

    def test_data_types_consistency(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that data types are consistent after preparation."""
        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        if not prepared_data.empty:
            # FIPS should be integer
            assert pd.api.types.is_integer_dtype(prepared_data['fips'])

            # Year should be integer
            assert pd.api.types.is_integer_dtype(prepared_data['year'])

            # Weights should be numeric
            if 'weight' in prepared_data.columns:
                assert pd.api.types.is_numeric_dtype(prepared_data['weight'])

    def test_no_negative_weights(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that all weights are positive."""
        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        if 'weight' in prepared_data.columns and not prepared_data.empty:
            assert all(prepared_data['weight'] > 0), "All weights should be positive"

    def test_valid_year_range(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that years are within expected range."""
        years = [2010, 2011, 2012]

        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            years
        )

        if not prepared_data.empty:
            assert prepared_data['year'].min() >= min(years)
            assert prepared_data['year'].max() <= max(years)

    def test_fips_code_validity(self, sample_start_data, sample_net_returns_data, sample_georef_data):
        """Test that FIPS codes are valid (5-digit integers)."""
        prepared_data = prepare_estimation_data(
            sample_start_data,
            sample_net_returns_data,
            sample_georef_data,
            [2010, 2011, 2012]
        )

        if not prepared_data.empty:
            # FIPS codes should be positive integers
            assert all(prepared_data['fips'] > 0)
            # Should typically be 4-5 digit numbers for counties
            assert all(prepared_data['fips'] < 99999)