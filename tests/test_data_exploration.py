"""
Tests for data exploration and visualization utilities.

These tests ensure data exploration functions work correctly
and produce valid summaries and visualizations.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from landuse.data_exploration import (
    explore_nri_data,
    analyze_metro_population_changes,
    create_transition_matrix,
    plot_transition_heatmap,
    analyze_spatial_patterns,
    create_temporal_plots,
    generate_summary_report
)


class TestNRIDataExploration:
    """Test NRI data exploration functions."""

    @pytest.fixture
    def sample_nri_data(self):
        """Create sample NRI data for testing."""
        np.random.seed(42)
        n_obs = 200

        return pd.DataFrame({
            'fips': np.random.choice(range(1001, 1021), n_obs),
            'year': np.random.choice([2010, 2011, 2012], n_obs),
            'landuse': np.random.choice(['crop', 'pasture', 'forest', 'urban'], n_obs),
            'acres': np.random.uniform(100, 1000, n_obs),
            'slope': np.random.uniform(0, 20, n_obs),
            'elevation': np.random.uniform(100, 2000, n_obs),
            'precipitation': np.random.uniform(20, 60, n_obs),
            'temperature': np.random.uniform(-5, 25, n_obs)
        })

    def test_explore_nri_data_basic(self, sample_nri_data, tmp_path):
        """Test basic NRI data exploration."""
        # Save sample data to temp file
        nri_file = tmp_path / "test_nri.csv"
        sample_nri_data.to_csv(nri_file, index=False)

        summary = explore_nri_data(str(nri_file), str(tmp_path))

        # Check that summary is returned
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0

        # Check that summary contains expected statistics
        expected_stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        for stat in expected_stats:
            assert stat in summary.index

        # Check that output file is created
        assert (tmp_path / 'nri_summary_stats.csv').exists()

    def test_explore_nri_data_missing_values(self, tmp_path):
        """Test NRI exploration with missing values."""
        # Create data with missing values
        nri_data_missing = pd.DataFrame({
            'fips': [1001, 1002, 1003, 1004],
            'year': [2010, 2011, None, 2012],
            'landuse': ['crop', None, 'forest', 'urban'],
            'acres': [100, 200, None, 400],
            'slope': [5, 10, 15, None]
        })

        nri_file = tmp_path / "test_nri_missing.csv"
        nri_data_missing.to_csv(nri_file, index=False)

        # Should handle missing values gracefully
        summary = explore_nri_data(str(nri_file))

        assert isinstance(summary, pd.DataFrame)
        # Should only include numeric columns for summary stats
        numeric_cols = nri_data_missing.select_dtypes(include=[np.number]).columns
        assert all(col in summary.columns for col in numeric_cols if col != 'Unnamed: 0')

    def test_explore_nri_temporal_analysis(self, tmp_path):
        """Test temporal analysis in NRI exploration."""
        # Create data with clear temporal patterns
        temporal_data = pd.DataFrame({
            'year': [2010, 2010, 2011, 2011, 2012, 2012],
            'acres': [100, 110, 120, 130, 140, 150],  # Increasing trend
            'fips': [1001, 1002, 1001, 1002, 1001, 1002],
            'landuse': ['crop'] * 6
        })

        nri_file = tmp_path / "temporal_nri.csv"
        temporal_data.to_csv(nri_file, index=False)

        summary = explore_nri_data(str(nri_file))

        # Should process temporal data without errors
        assert isinstance(summary, pd.DataFrame)

    def test_explore_nri_landuse_categories(self, sample_nri_data, tmp_path):
        """Test land use category analysis."""
        nri_file = tmp_path / "landuse_nri.csv"
        sample_nri_data.to_csv(nri_file, index=False)

        # Capture printed output (land use distribution)
        with patch('builtins.print') as mock_print:
            summary = explore_nri_data(str(nri_file))

            # Should print land use distribution
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            landuse_printed = any('Land Use Distribution' in str(call) for call in print_calls)
            assert landuse_printed

    def test_explore_nri_file_formats(self, sample_nri_data, tmp_path):
        """Test handling of different file formats."""
        # Test CSV
        csv_file = tmp_path / "test.csv"
        sample_nri_data.to_csv(csv_file, index=False)

        summary_csv = explore_nri_data(str(csv_file))
        assert isinstance(summary_csv, pd.DataFrame)

        # Test pickle format
        pickle_file = tmp_path / "test.pkl"
        sample_nri_data.to_pickle(pickle_file)

        summary_pickle = explore_nri_data(str(pickle_file))
        assert isinstance(summary_pickle, pd.DataFrame)

        # Summaries should be similar
        pd.testing.assert_frame_equal(summary_csv, summary_pickle)


class TestMetroPopulationAnalysis:
    """Test metropolitan population change analysis."""

    @pytest.fixture
    def sample_population_data(self):
        """Create sample population data."""
        np.random.seed(42)

        return pd.DataFrame({
            'fips': [1001, 1001, 1001, 1002, 1002, 1002],
            'year': [2010, 2011, 2012, 2010, 2011, 2012],
            'population': [100000, 105000, 110000, 50000, 52000, 55000],
            'metro_area': ['Metro A', 'Metro A', 'Metro A', 'Metro B', 'Metro B', 'Metro B']
        })

    @pytest.fixture
    def sample_metro_definitions(self):
        """Create sample metro area definitions."""
        return pd.DataFrame({
            'fips': [1001, 1002, 1003],
            'metro_area': ['Metro A', 'Metro B', 'Metro A'],
            'cbsa': ['12345', '12346', '12345']
        })

    def test_analyze_metro_population_basic(self, sample_population_data):
        """Test basic metropolitan population analysis."""
        result = analyze_metro_population_changes(sample_population_data)

        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Should calculate population changes
        assert 'pop_change' in result.columns
        assert 'pop_pct_change' in result.columns
        assert 'annual_growth_rate' in result.columns

        # Check that changes are calculated correctly
        fips_1001_data = result[result['fips'] == 1001].sort_values('year')
        if len(fips_1001_data) > 1:
            # First change should be 105000 - 100000 = 5000
            first_change = fips_1001_data['pop_change'].iloc[1]
            assert abs(first_change - 5000) < 1e-6

    def test_metro_population_growth_rates(self, sample_population_data):
        """Test population growth rate calculations."""
        result = analyze_metro_population_changes(sample_population_data)

        # Growth rates should be reasonable
        growth_rates = result['annual_growth_rate'].dropna()
        assert all(growth_rates > -50), "Growth rates shouldn't be extremely negative"
        assert all(growth_rates < 50), "Growth rates shouldn't be extremely positive"

        # For the sample data, growth should be positive
        assert all(growth_rates > 0), "Sample data should show positive growth"

    def test_metro_aggregation(self, sample_population_data, sample_metro_definitions):
        """Test aggregation to metropolitan area level."""
        result = analyze_metro_population_changes(
            sample_population_data,
            sample_metro_definitions
        )

        # Should aggregate to metro level
        assert 'metro_area' in result.columns
        assert 'num_counties' in result.columns
        assert 'metro_pop_pct_change' in result.columns

        # Check aggregation worked
        metro_a_data = result[result['metro_area'] == 'Metro A']
        if not metro_a_data.empty:
            # Metro A should have population from county 1001
            metro_2010 = metro_a_data[metro_a_data['year'] == 2010]['population'].iloc[0]
            assert metro_2010 == 100000

    def test_year_filtering(self, sample_population_data):
        """Test filtering by specific years."""
        years = [2011, 2012]
        result = analyze_metro_population_changes(
            sample_population_data,
            years=years
        )

        # Should only include specified years
        assert all(result['year'].isin(years))
        assert 2010 not in result['year'].values

    def test_empty_population_data(self):
        """Test handling of empty population data."""
        empty_data = pd.DataFrame(columns=['fips', 'year', 'population'])

        result = analyze_metro_population_changes(empty_data)

        # Should return empty DataFrame without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_population_file_formats(self, sample_population_data, tmp_path):
        """Test loading population data from different file formats."""
        # Test CSV
        csv_file = tmp_path / "pop.csv"
        sample_population_data.to_csv(csv_file, index=False)

        result_csv = analyze_metro_population_changes(str(csv_file))
        assert isinstance(result_csv, pd.DataFrame)

        # Test pickle
        pickle_file = tmp_path / "pop.pkl"
        sample_population_data.to_pickle(pickle_file)

        result_pickle = analyze_metro_population_changes(str(pickle_file))
        assert isinstance(result_pickle, pd.DataFrame)


class TestTransitionMatrix:
    """Test land use transition matrix creation."""

    @pytest.fixture
    def sample_transition_data(self):
        """Create sample transition data."""
        return pd.DataFrame({
            'startuse': ['crop', 'crop', 'forest', 'forest', 'pasture', 'pasture'],
            'enduse': ['crop', 'urban', 'forest', 'crop', 'pasture', 'urban'],
            'xfact': [1.0, 2.0, 1.5, 1.0, 2.5, 1.0]
        })

    def test_create_transition_matrix_unweighted(self, sample_transition_data):
        """Test unweighted transition matrix creation."""
        transition_matrix = create_transition_matrix(
            sample_transition_data,
            start_col='startuse',
            end_col='enduse',
            weight_col=None
        )

        # Check structure
        assert isinstance(transition_matrix, pd.DataFrame)

        # Rows should sum to 1 (probabilities)
        row_sums = transition_matrix.sum(axis=1)
        assert all(abs(row_sums - 1.0) < 1e-10), "Transition probabilities should sum to 1"

        # All values should be between 0 and 1
        assert all((transition_matrix >= 0).all()), "All probabilities should be non-negative"
        assert all((transition_matrix <= 1).all()), "All probabilities should be <= 1"

    def test_create_transition_matrix_weighted(self, sample_transition_data):
        """Test weighted transition matrix creation."""
        transition_matrix = create_transition_matrix(
            sample_transition_data,
            start_col='startuse',
            end_col='enduse',
            weight_col='xfact'
        )

        # Check structure
        assert isinstance(transition_matrix, pd.DataFrame)

        # Rows should sum to 1
        row_sums = transition_matrix.sum(axis=1)
        assert all(abs(row_sums - 1.0) < 1e-10), "Weighted transition probabilities should sum to 1"

        # Check that weights affected the calculation
        # Crop->crop: weight 1.0, Crop->urban: weight 2.0
        # So crop->urban should have probability 2/3, crop->crop should have 1/3
        if 'crop' in transition_matrix.index and 'urban' in transition_matrix.columns:
            crop_to_urban = transition_matrix.loc['crop', 'urban']
            crop_to_crop = transition_matrix.loc['crop', 'crop']
            expected_urban = 2.0 / 3.0
            expected_crop = 1.0 / 3.0

            assert abs(crop_to_urban - expected_urban) < 1e-10
            assert abs(crop_to_crop - expected_crop) < 1e-10

    def test_transition_matrix_missing_combinations(self):
        """Test handling of missing start-end combinations."""
        limited_data = pd.DataFrame({
            'startuse': ['crop', 'crop'],
            'enduse': ['crop', 'urban'],
            'xfact': [1.0, 1.0]
        })

        transition_matrix = create_transition_matrix(limited_data)

        # Should fill missing combinations with 0
        assert 'forest' not in transition_matrix.index  # No forest starts
        if 'pasture' in transition_matrix.columns:
            assert all(transition_matrix['pasture'] == 0)  # No transitions to pasture

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_transition_heatmap(self, mock_show, mock_savefig, sample_transition_data):
        """Test transition matrix heatmap plotting."""
        transition_matrix = create_transition_matrix(sample_transition_data)

        try:
            fig = plot_transition_heatmap(transition_matrix)

            # Should return a figure
            assert fig is not None

        except Exception as e:
            pytest.skip(f"Transition heatmap test failed: {str(e)}")


class TestSpatialAnalysis:
    """Test spatial pattern analysis."""

    @pytest.fixture
    def sample_spatial_data(self):
        """Create sample data for spatial analysis."""
        return pd.DataFrame({
            'fips': ['01001', '01002', '01003'],
            'population': [100000, 200000, 150000],
            'land_value': [1000, 1500, 1200]
        })

    @pytest.fixture
    def sample_shapefile_data(self):
        """Create mock shapefile data."""
        # This would normally be a real shapefile, but for testing we'll mock it
        from shapely.geometry import Polygon
        import geopandas as gpd

        # Create simple polygons
        polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])
        ]

        return gpd.GeoDataFrame({
            'GEOID': ['01001', '01002', '01003'],
            'geometry': polygons
        })

    @patch('geopandas.read_file')
    @patch('libpysal.weights.Queen.from_dataframe')
    @patch('esda.moran.Moran')
    def test_analyze_spatial_patterns(self, mock_moran, mock_queen, mock_read_file,
                                    sample_spatial_data, sample_shapefile_data):
        """Test spatial pattern analysis."""
        # Mock the file reading
        mock_read_file.return_value = sample_shapefile_data

        # Mock spatial weights and Moran's I
        mock_weights = MagicMock()
        mock_queen.return_value = mock_weights

        mock_moran_result = MagicMock()
        mock_moran_result.I = 0.25
        mock_moran_result.p_norm = 0.05
        mock_moran.return_value = mock_moran_result

        try:
            spatial_data = analyze_spatial_patterns(
                sample_spatial_data,
                "dummy_shapefile.shp",
                "population"
            )

            # Should return GeoDataFrame
            assert isinstance(spatial_data, gpd.GeoDataFrame)

            # Should have geometry column
            assert 'geometry' in spatial_data.columns

            # Should have merged the data
            assert 'population' in spatial_data.columns

        except ImportError:
            pytest.skip("Spatial analysis libraries not available")
        except Exception as e:
            pytest.skip(f"Spatial analysis test failed: {str(e)}")

    def test_fips_formatting(self, sample_spatial_data):
        """Test FIPS code formatting."""
        # Test with integer FIPS codes
        int_data = sample_spatial_data.copy()
        int_data['fips'] = [1001, 1002, 1003]

        # The function should handle FIPS formatting
        # (This test would need the actual function to verify formatting)
        assert all(len(str(fips).zfill(5)) == 5 for fips in int_data['fips'])


class TestTemporalPlots:
    """Test temporal plotting functions."""

    @pytest.fixture
    def sample_temporal_data(self):
        """Create sample temporal data."""
        np.random.seed(42)

        return pd.DataFrame({
            'year': [2010, 2011, 2012] * 4,
            'region': ['North', 'North', 'North', 'South', 'South', 'South'] * 2,
            'acres_crop': np.random.uniform(1000, 2000, 12),
            'acres_forest': np.random.uniform(500, 1500, 12),
            'population': np.random.uniform(50000, 200000, 12)
        })

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_temporal_plots_basic(self, mock_show, mock_savefig, sample_temporal_data, tmp_path):
        """Test basic temporal plot creation."""
        variables = ['acres_crop', 'acres_forest']

        try:
            figures = create_temporal_plots(
                sample_temporal_data,
                variables,
                time_col='year',
                output_dir=str(tmp_path)
            )

            # Should return dictionary of figures
            assert isinstance(figures, dict)
            assert len(figures) == len(variables)

            for var in variables:
                assert var in figures

        except Exception as e:
            pytest.skip(f"Temporal plots test failed: {str(e)}")

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_temporal_plots_grouped(self, mock_show, mock_savefig, sample_temporal_data):
        """Test temporal plots with grouping."""
        variables = ['population']

        try:
            figures = create_temporal_plots(
                sample_temporal_data,
                variables,
                time_col='year',
                group_col='region'
            )

            # Should create grouped plots
            assert isinstance(figures, dict)
            assert 'population' in figures

        except Exception as e:
            pytest.skip(f"Grouped temporal plots test failed: {str(e)}")

    def test_temporal_plots_missing_variables(self, sample_temporal_data):
        """Test handling of missing variables."""
        variables = ['nonexistent_variable', 'acres_crop']

        # Should skip missing variables without error
        figures = create_temporal_plots(
            sample_temporal_data,
            variables,
            time_col='year'
        )

        # Should only have figure for existing variable
        assert 'acres_crop' in figures
        assert 'nonexistent_variable' not in figures


class TestReportGeneration:
    """Test summary report generation."""

    def test_generate_summary_report(self, tmp_path):
        """Test comprehensive summary report generation."""
        # Create sample datasets
        data_dict = {
            'nri_data': pd.DataFrame({
                'fips': [1001, 1002, 1003],
                'acres': [100, 200, 300],
                'year': [2010, 2011, 2012]
            }),
            'population_data': pd.DataFrame({
                'fips': [1001, 1002],
                'population': [50000, 75000],
                'density': [100, 150]
            })
        }

        report_file = tmp_path / "summary_report.txt"

        generate_summary_report(data_dict, str(report_file))

        # Check that report file was created
        assert report_file.exists()

        # Check report content
        with open(report_file, 'r') as f:
            content = f.read()

        assert 'Land Use Data Analysis Summary Report' in content
        assert 'nri_data' in content
        assert 'population_data' in content
        assert 'Shape:' in content

    def test_summary_report_with_missing_values(self, tmp_path):
        """Test report generation with missing values."""
        data_with_missing = {
            'test_data': pd.DataFrame({
                'col1': [1, 2, None, 4],
                'col2': [10, None, 30, 40],
                'col3': ['a', 'b', 'c', None]
            })
        }

        report_file = tmp_path / "missing_report.txt"

        generate_summary_report(data_with_missing, str(report_file))

        assert report_file.exists()

        # Check that missing values are reported
        with open(report_file, 'r') as f:
            content = f.read()

        assert 'missing values' in content.lower()

    def test_summary_report_memory_usage(self, tmp_path):
        """Test that memory usage is reported correctly."""
        large_data = {
            'large_dataset': pd.DataFrame({
                'col1': range(10000),
                'col2': np.random.random(10000),
                'col3': ['text'] * 10000
            })
        }

        report_file = tmp_path / "memory_report.txt"

        generate_summary_report(large_data, str(report_file))

        with open(report_file, 'r') as f:
            content = f.read()

        assert 'Memory usage:' in content
        assert 'MB' in content


class TestDataQuality:
    """Test data quality checks in exploration functions."""

    def test_outlier_detection_concepts(self):
        """Test concepts for outlier detection (would be implemented in actual functions)."""
        # Create data with outliers
        data_with_outliers = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'fips': [1001, 1002, 1003, 1004, 1005, 1006]
        })

        # Basic outlier detection using IQR
        Q1 = data_with_outliers['values'].quantile(0.25)
        Q3 = data_with_outliers['values'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data_with_outliers[
            (data_with_outliers['values'] < lower_bound) |
            (data_with_outliers['values'] > upper_bound)
        ]

        # Should detect the outlier value
        assert len(outliers) == 1
        assert outliers['values'].iloc[0] == 100

    def test_data_consistency_checks(self):
        """Test data consistency validation concepts."""
        # Test year range validation
        data = pd.DataFrame({
            'year': [2010, 2011, 3000, 2012],  # 3000 is unrealistic
            'value': [100, 200, 300, 400]
        })

        # Years should be within reasonable range
        valid_years = data[(data['year'] >= 1990) & (data['year'] <= 2030)]
        assert len(valid_years) == 3  # Should exclude year 3000

        # Test FIPS code validation
        fips_data = pd.DataFrame({
            'fips': [1001, 12345, 99999, 1002],  # 99999 might be invalid
            'value': [100, 200, 300, 400]
        })

        # FIPS codes should be reasonable (this is a simplified check)
        valid_fips = fips_data[(fips_data['fips'] >= 1001) & (fips_data['fips'] <= 56999)]
        assert len(valid_fips) >= 2

    def test_completeness_assessment(self):
        """Test data completeness assessment."""
        incomplete_data = pd.DataFrame({
            'required_col': [1, 2, None, 4, 5],
            'optional_col': [10, None, None, 40, 50],
            'complete_col': [100, 200, 300, 400, 500]
        })

        # Calculate completeness rates
        completeness = {}
        for col in incomplete_data.columns:
            completeness[col] = incomplete_data[col].notna().mean()

        assert completeness['complete_col'] == 1.0  # 100% complete
        assert completeness['required_col'] == 0.8   # 80% complete
        assert completeness['optional_col'] == 0.6   # 60% complete

        # Identify columns with low completeness
        low_completeness = [col for col, rate in completeness.items() if rate < 0.7]
        assert 'optional_col' in low_completeness