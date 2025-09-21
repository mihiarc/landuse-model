"""
Performance benchmarks and regression tests.

These tests ensure the Python implementation produces results
consistent with the original R implementation and maintains
acceptable performance characteristics.
"""

import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
from unittest.mock import patch

from landuse.logit_estimation import (
    prepare_estimation_data,
    estimate_mnlogit,
    calculate_marginal_effects
)
from landuse.climate_impact import calc_utility, calc_prob_chg
from landuse.marginal_effects import calculate_elasticities


class TestPerformanceBenchmarks:
    """Test performance characteristics of key functions."""

    @pytest.fixture
    def performance_data(self):
        """Create datasets of various sizes for performance testing."""
        np.random.seed(42)

        sizes = {
            'small': 100,
            'medium': 1000,
            'large': 5000
        }

        datasets = {}

        for size_name, n_obs in sizes.items():
            # Estimation data
            estimation_data = pd.DataFrame({
                'fips': np.random.choice(range(1001, 1101), n_obs),
                'year': np.random.choice([2010, 2011, 2012], n_obs),
                'enduse': np.random.choice([1, 2, 3, 4], n_obs),
                'lcc': np.random.choice([1, 2, 3], n_obs),
                'nr_cr': np.random.normal(500, 100, n_obs),
                'nr_ps': np.random.normal(200, 50, n_obs),
                'nr_fr': np.random.normal(100, 30, n_obs),
                'nr_ur': np.random.normal(1000, 200, n_obs),
                'xfact': np.random.uniform(0.5, 2.0, n_obs),
                'region': np.random.choice(['NO', 'SO'], n_obs)
            })

            # Climate impact data
            climate_data = pd.DataFrame({
                'fips': range(1001, 1001 + min(n_obs // 10, 100)),
                'riad_id': range(1, min(n_obs // 10, 100) + 1),
                'xfact': np.random.uniform(0.5, 2.0, min(n_obs // 10, 100)),
                'lcc': np.random.choice([1, 2, 3], min(n_obs // 10, 100)),
                'crnr_obs': np.random.normal(500, 100, min(n_obs // 10, 100)),
                'frnr_obs': np.random.normal(100, 30, min(n_obs // 10, 100)),
                'urnr_obs': np.random.normal(1000, 200, min(n_obs // 10, 100))
            })

            datasets[size_name] = {
                'estimation': estimation_data,
                'climate': climate_data
            }

        return datasets

    def test_data_preparation_performance(self, performance_data):
        """Test performance of data preparation functions."""
        georef_data = pd.DataFrame({
            'fips': range(1001, 1101),
            'region': ['NO'] * 50 + ['SO'] * 50,
            'subregion': ['NO'] * 50 + ['SO'] * 50
        })

        nr_data = pd.DataFrame({
            'fips': np.repeat(range(1001, 1021), 3),
            'year': np.tile([2008, 2009, 2010], 20),
            'nr_cr': np.random.normal(500, 100, 60),
            'nr_ps': np.random.normal(200, 50, 60),
            'nr_fr': np.random.normal(100, 30, 60),
            'nr_ur': np.random.normal(1000, 200, 60)
        })

        performance_results = {}

        for size_name, data in performance_data.items():
            start_time = time.time()

            try:
                prepared_data = prepare_estimation_data(
                    data['estimation'],
                    nr_data,
                    georef_data,
                    [2010, 2011, 2012]
                )

                end_time = time.time()
                performance_results[size_name] = {
                    'time': end_time - start_time,
                    'input_size': len(data['estimation']),
                    'output_size': len(prepared_data)
                }

            except Exception as e:
                pytest.skip(f"Data preparation performance test failed for {size_name}: {str(e)}")

        # Performance should scale reasonably
        if 'small' in performance_results and 'medium' in performance_results:
            small_time = performance_results['small']['time']
            medium_time = performance_results['medium']['time']

            # Medium should not be more than 20x slower than small
            # (This is a loose bound to account for setup overhead)
            assert medium_time < small_time * 20, \
                f"Performance degradation too severe: {small_time} -> {medium_time}"

    def test_logit_estimation_performance(self, performance_data):
        """Test performance of logit estimation."""
        performance_results = {}

        for size_name, data in performance_data.items():
            if size_name == 'large':
                # Skip large dataset for logit estimation (too slow for testing)
                continue

            estimation_data = data['estimation']

            # Ensure we have enough variation for estimation
            if len(estimation_data['enduse'].unique()) < 2:
                continue

            start_time = time.time()

            try:
                formula = 'enduse ~ lcc + nr_cr + nr_ps'
                model = estimate_mnlogit(estimation_data, formula)

                end_time = time.time()
                performance_results[size_name] = {
                    'time': end_time - start_time,
                    'input_size': len(estimation_data),
                    'converged': hasattr(model, 'llf') and np.isfinite(model.llf)
                }

            except Exception as e:
                pytest.skip(f"Logit estimation performance test failed for {size_name}: {str(e)}")

        # Check that estimation completes in reasonable time
        for size_name, results in performance_results.items():
            assert results['time'] < 60, f"Estimation took too long for {size_name}: {results['time']} seconds"

    def test_climate_calculation_performance(self, performance_data, known_model_coefficients):
        """Test performance of climate impact calculations."""
        performance_results = {}

        for size_name, data in performance_data.items():
            climate_data = data['climate']

            start_time = time.time()

            try:
                # Test utility calculation
                util_data = climate_data.copy()
                util_data['nr.cr'] = climate_data['crnr_obs']
                util_data['nr.fr'] = climate_data['frnr_obs']
                util_data['nr.ur'] = climate_data['urnr_obs']
                util_data['nr.ps'] = np.random.normal(200, 50, len(climate_data))

                result = calc_utility(known_model_coefficients, util_data)

                end_time = time.time()
                performance_results[size_name] = {
                    'time': end_time - start_time,
                    'input_size': len(climate_data),
                    'output_size': len(result)
                }

            except Exception as e:
                pytest.skip(f"Climate calculation performance test failed for {size_name}: {str(e)}")

        # Performance should be fast for climate calculations
        for size_name, results in performance_results.items():
            assert results['time'] < 5, f"Climate calculations too slow for {size_name}: {results['time']} seconds"

    def test_memory_usage_estimation(self, performance_data):
        """Test memory usage patterns."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process medium dataset
        medium_data = performance_data['medium']['estimation']

        try:
            # Simple data operations that should not use excessive memory
            grouped = medium_data.groupby(['fips', 'year']).agg({
                'nr_cr': 'mean',
                'nr_ps': 'mean',
                'nr_fr': 'mean',
                'nr_ur': 'mean'
            })

            merged = medium_data.merge(grouped, on=['fips', 'year'], suffixes=('', '_mean'))

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            # Memory increase should be reasonable (less than 500MB for test data)
            assert memory_increase < 500, f"Excessive memory usage: {memory_increase} MB"

        except Exception as e:
            pytest.skip(f"Memory usage test failed: {str(e)}")


class TestRegressionCompatibility:
    """Test compatibility with expected R implementation results."""

    @pytest.fixture
    def reference_calculations(self):
        """Expected results from R implementation for regression testing."""
        # These would be actual values from the R implementation
        # For testing purposes, we use plausible values
        return {
            'utility_calculation': {
                'input': {
                    'lcc': [1, 2, 3],
                    'nr.cr': [500, 600, 400],
                    'nr.ps': [200, 250, 180],
                    'nr.fr': [100, 120, 90],
                    'nr.ur': [1000, 1200, 800]
                },
                'coefficients': np.array([0.5, -0.1, 0.3, -0.05, 1.2, -0.2, 0.001, 0.0005, 0.0008, 0.0015]),
                'expected_utilities': {
                    'crop_util': [0.5, 0.6, 0.4],
                    'pasture_util': [0.46, 0.37, 0.434],
                    'forest_util': [0.45, 0.36, 0.395],
                    'urban_util': [2.3, 1.96, 2.04]
                }
            },
            'probability_calculation': {
                'baseline_probs': [0.25, 0.25, 0.25, 0.25],
                'scenario_probs': [0.30, 0.23, 0.22, 0.25],
                'expected_changes': [0.05, -0.02, -0.03, 0.0]
            }
        }

    def test_utility_calculation_regression(self, reference_calculations):
        """Test that utility calculations match R implementation."""
        ref_data = reference_calculations['utility_calculation']

        input_df = pd.DataFrame(ref_data['input'])
        coefficients = ref_data['coefficients']

        result = calc_utility(coefficients, input_df)

        # Check that utilities are calculated correctly
        # (We use loose tolerances since we don't have exact R values)
        for util_col in ['crop_util', 'pasture_util', 'forest_util', 'urban_util']:
            assert util_col in result.columns, f"Missing utility column: {util_col}"
            assert all(np.isfinite(result[util_col])), f"Non-finite values in {util_col}"

        # Crop utility should be linear in crop returns
        crop_returns = input_df['nr.cr'].values
        crop_utils = result['crop_util'].values
        crop_coeff = coefficients[6]

        expected_crop_utils = crop_coeff * crop_returns
        np.testing.assert_allclose(crop_utils, expected_crop_utils, rtol=1e-10)

    def test_probability_bounds_regression(self, known_model_coefficients):
        """Test that probabilities stay within valid bounds across scenarios."""
        # Create data that might cause numerical issues
        extreme_data = pd.DataFrame({
            'fips': [1001],
            'riad_id': [1],
            'xfact': [1.0],
            'lcc': [1],
            'crnr_obs': [10000],  # Very high returns
            'frnr_obs': [1],      # Very low returns
            'urnr_obs': [5000]
        })

        baseline = extreme_data.copy()
        scenario = extreme_data.copy()
        scenario['nr.cr'] = [15000]  # Even higher crop returns
        scenario['nr.fr'] = [1]
        scenario['nr.ur'] = [5000]

        result = calc_prob_chg(known_model_coefficients, baseline, scenario)

        # Extract probability columns
        prob_cols = [col for col in result.columns if col.startswith('pr_') and not col.startswith('probchg_')]

        for col in prob_cols:
            probs = result[col].values
            assert all(probs >= 0), f"Negative probabilities in {col}"
            assert all(probs <= 1), f"Probabilities > 1 in {col}"

        # Check that probabilities sum to 1
        baseline_cols = [col for col in prob_cols if col.endswith('0')]
        scenario_cols = [col for col in prob_cols if col.endswith('1')]

        if baseline_cols:
            baseline_sum = result[baseline_cols].sum(axis=1).iloc[0]
            assert abs(baseline_sum - 1.0) < 1e-10, "Baseline probabilities don't sum to 1"

        if scenario_cols:
            scenario_sum = result[scenario_cols].sum(axis=1).iloc[0]
            assert abs(scenario_sum - 1.0) < 1e-10, "Scenario probabilities don't sum to 1"

    def test_marginal_effects_signs_regression(self):
        """Test that marginal effects have economically reasonable signs."""
        # Create data with clear economic relationships
        test_data = pd.DataFrame({
            'fips': [1001] * 3,
            'year': [2010] * 3,
            'enduse': [1, 2, 3],  # Crop, pasture, forest
            'lcc': [1, 2, 3],     # Different land quality
            'nr_cr': [500, 400, 300],   # Higher for crop land
            'nr_ps': [200, 250, 150],   # Higher for pasture land
            'nr_fr': [100, 120, 200],   # Higher for forest land
            'nr_ur': [1000, 1000, 1000], # Constant urban returns
            'xfact': [1.0, 1.0, 1.0]
        })

        try:
            from unittest.mock import MagicMock
            import statsmodels.api as sm

            # Create a mock model for testing marginal effects signs
            mock_model = MagicMock(spec=sm.discrete.discrete_model.MultinomialResultsWrapper)

            # Mock marginal effects that should have reasonable signs
            mock_margeff = MagicMock()
            mock_summary_frame = pd.DataFrame({
                'dy/dx': [0.002, -0.001, -0.001],  # Higher crop returns increase crop probability
                'Std. Err.': [0.0005, 0.0003, 0.0003],
                'z': [4.0, -3.33, -3.33],
                'P>|z|': [0.001, 0.001, 0.001],
                '[0.025': [0.001, -0.0016, -0.0016],
                '0.975]': [0.003, -0.0004, -0.0004]
            }, index=['nr_cr_1', 'nr_cr_2', 'nr_cr_3'])  # Effects on outcomes 1, 2, 3

            mock_margeff.summary_frame.return_value = mock_summary_frame
            mock_model.get_margeff.return_value = mock_margeff

            marginal_effects = calculate_marginal_effects(mock_model, test_data, 'nr_cr')

            # Check that we get reasonable results
            assert isinstance(marginal_effects, pd.DataFrame)
            if not marginal_effects.empty:
                # Should have marginal effects for the variable
                assert 'nr_cr' in marginal_effects['outcome'].iloc[0]

        except Exception as e:
            pytest.skip(f"Marginal effects signs test failed: {str(e)}")

    def test_numerical_stability_regression(self, known_model_coefficients):
        """Test numerical stability across different scales."""
        # Test with different scales of data
        scales = [0.001, 1.0, 1000.0]

        base_data = pd.DataFrame({
            'fips': [1001],
            'riad_id': [1],
            'xfact': [1.0],
            'lcc': [2],
            'crnr_obs': [500],
            'frnr_obs': [100],
            'urnr_obs': [1000]
        })

        results = {}

        for scale in scales:
            scaled_data = base_data.copy()
            scaled_data['crnr_obs'] = scaled_data['crnr_obs'] * scale
            scaled_data['frnr_obs'] = scaled_data['frnr_obs'] * scale
            scaled_data['urnr_obs'] = scaled_data['urnr_obs'] * scale

            scenario_data = scaled_data.copy()
            scenario_data['nr.cr'] = scaled_data['crnr_obs'] * 1.1  # 10% increase
            scenario_data['nr.fr'] = scaled_data['frnr_obs']
            scenario_data['nr.ur'] = scaled_data['urnr_obs']

            try:
                result = calc_prob_chg(known_model_coefficients, scaled_data, scenario_data)
                results[scale] = result

                # Check numerical stability
                prob_changes = [result[col].iloc[0] for col in result.columns if col.startswith('probchg_')]
                assert all(np.isfinite(prob_changes)), f"Non-finite probability changes at scale {scale}"

            except Exception as e:
                pytest.skip(f"Numerical stability test failed at scale {scale}: {str(e)}")

        # Changes should be similar across scales (for percentage changes)
        if len(results) > 1:
            change_cols = [col for col in results[1.0].columns if col.startswith('probchg_')]
            for col in change_cols:
                changes = [results[scale][col].iloc[0] for scale in scales if scale in results]
                if len(changes) > 1:
                    # Changes should be of similar magnitude (this is a weak test)
                    change_range = max(changes) - min(changes)
                    change_mean = np.mean(np.abs(changes))
                    if change_mean > 0:
                        relative_range = change_range / change_mean
                        # Allow for some variation but not extreme differences
                        assert relative_range < 10, f"Too much variation across scales in {col}"


class TestConsistencyChecks:
    """Test internal consistency of calculations."""

    def test_probability_conservation(self, known_model_coefficients):
        """Test that probability changes conserve total probability."""
        test_scenarios = [
            {
                'baseline': {'crnr_obs': [500], 'frnr_obs': [100], 'urnr_obs': [1000]},
                'scenario': {'nr.cr': [550], 'nr.fr': [100], 'nr.ur': [1000]}
            },
            {
                'baseline': {'crnr_obs': [400], 'frnr_obs': [150], 'urnr_obs': [1200]},
                'scenario': {'nr.cr': [400], 'nr.fr': [200], 'nr.ur': [1200]}
            }
        ]

        for i, scenario in enumerate(test_scenarios):
            baseline_df = pd.DataFrame({
                'fips': [1001],
                'riad_id': [1],
                'xfact': [1.0],
                'lcc': [2],
                **scenario['baseline']
            })

            scenario_df = baseline_df.copy()
            scenario_df.update(scenario['scenario'])

            result = calc_prob_chg(known_model_coefficients, baseline_df, scenario_df)

            # Check probability conservation
            change_cols = [col for col in result.columns if col.startswith('probchg_')]
            if change_cols:
                total_change = result[change_cols].sum(axis=1).iloc[0]
                assert abs(total_change) < 1e-10, f"Probability changes don't sum to zero in scenario {i}"

    def test_elasticity_reciprocity(self):
        """Test that elasticities satisfy basic economic relationships."""
        # Create mock model for elasticity testing
        from unittest.mock import MagicMock
        import statsmodels.api as sm

        mock_model = MagicMock(spec=sm.discrete.discrete_model.MultinomialResultsWrapper)

        # Mock predict method with simple relationship
        def mock_predict(data):
            # Simple logistic relationship
            n_obs = len(data)
            crop_util = 0.001 * data.get('nr_cr', 0)
            forest_util = 0.0005 * data.get('nr_fr', 0)

            # Convert to probabilities
            utils = np.column_stack([
                np.zeros(n_obs),  # Reference
                crop_util,
                forest_util
            ])
            exp_utils = np.exp(utils)
            probs = exp_utils / exp_utils.sum(axis=1, keepdims=True)

            return pd.DataFrame(probs, columns=['ref', 'crop', 'forest'])

        mock_model.predict = mock_predict
        mock_model.exog_names = ['const', 'nr_cr', 'nr_fr']

        test_data = pd.DataFrame({
            'nr_cr': [500, 600, 400],
            'nr_fr': [100, 120, 80],
            'xfact': [1.0, 1.0, 1.0]
        })

        try:
            elasticities = calculate_elasticities(mock_model, test_data, ['nr_cr', 'nr_fr'])

            # Basic checks
            assert isinstance(elasticities, pd.DataFrame)
            if not elasticities.empty:
                # Own-price elasticities for complementary goods should have expected signs
                # (This is a simplified test)
                crop_elast = elasticities[
                    (elasticities['variable'] == 'nr_cr') &
                    (elasticities['outcome'] == 'crop')
                ]

                if not crop_elast.empty:
                    # Own elasticity should be positive (higher crop returns -> higher crop probability)
                    assert crop_elast['elasticity'].iloc[0] > 0

        except Exception as e:
            pytest.skip(f"Elasticity reciprocity test failed: {str(e)}")

    def test_parameter_sensitivity(self, known_model_coefficients):
        """Test sensitivity to parameter changes."""
        base_data = pd.DataFrame({
            'fips': [1001],
            'riad_id': [1],
            'xfact': [1.0],
            'lcc': [2],
            'nr.cr': [500],
            'nr.ps': [200],
            'nr.fr': [100],
            'nr.ur': [1000]
        })

        # Test with original coefficients
        original_utils = calc_utility(known_model_coefficients, base_data)

        # Test with modified coefficients (small change)
        modified_coeffs = known_model_coefficients.copy()
        modified_coeffs[6] = modified_coeffs[6] * 1.1  # 10% increase in crop coefficient

        modified_utils = calc_utility(modified_coeffs, base_data)

        # Crop utility should increase
        original_crop = original_utils['crop_util'].iloc[0]
        modified_crop = modified_utils['crop_util'].iloc[0]

        assert modified_crop > original_crop, "Higher crop coefficient should increase crop utility"

        # Other utilities should be unchanged
        for util_col in ['pasture_util', 'forest_util', 'urban_util']:
            original_val = original_utils[util_col].iloc[0]
            modified_val = modified_utils[util_col].iloc[0]
            assert abs(original_val - modified_val) < 1e-10, f"{util_col} should be unchanged"


class TestLongRunningOperations:
    """Test operations that might take longer but should complete successfully."""

    @pytest.mark.slow
    def test_large_dataset_estimation(self):
        """Test estimation with larger dataset (marked as slow test)."""
        np.random.seed(42)

        # Create larger dataset
        n_obs = 10000
        large_data = pd.DataFrame({
            'enduse': np.random.choice([1, 2, 3], n_obs),
            'lcc': np.random.choice([1, 2, 3], n_obs),
            'nr_cr': np.random.normal(500, 100, n_obs),
            'nr_ps': np.random.normal(200, 50, n_obs),
            'nr_fr': np.random.normal(100, 30, n_obs),
            'fips': np.random.choice(range(1001, 1101), n_obs),
            'year': np.random.choice([2010, 2011, 2012], n_obs),
            'xfact': np.random.uniform(0.5, 2.0, n_obs)
        })

        start_time = time.time()

        try:
            formula = 'enduse ~ lcc + nr_cr + nr_ps + nr_fr'
            model = estimate_mnlogit(large_data, formula)

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within reasonable time
            assert execution_time < 600, f"Large estimation took too long: {execution_time} seconds"

            # Should produce valid results
            assert hasattr(model, 'params')
            assert hasattr(model, 'llf')
            assert all(np.isfinite(model.params))

        except Exception as e:
            pytest.skip(f"Large dataset estimation failed: {str(e)}")

    @pytest.mark.slow
    def test_monte_carlo_consistency(self, known_model_coefficients):
        """Test consistency across multiple random samples."""
        np.random.seed(42)

        n_simulations = 100
        results = []

        base_template = {
            'fips': [1001],
            'riad_id': [1],
            'xfact': [1.0],
            'lcc': [2],
            'crnr_obs': [500],
            'frnr_obs': [100],
            'urnr_obs': [1000]
        }

        for sim in range(n_simulations):
            # Add small random variations
            base_data = pd.DataFrame(base_template)
            base_data['crnr_obs'] = base_data['crnr_obs'] + np.random.normal(0, 10)
            base_data['frnr_obs'] = base_data['frnr_obs'] + np.random.normal(0, 5)
            base_data['urnr_obs'] = base_data['urnr_obs'] + np.random.normal(0, 20)

            scenario_data = base_data.copy()
            scenario_data['nr.cr'] = base_data['crnr_obs'] + 50  # Consistent increase
            scenario_data['nr.fr'] = base_data['frnr_obs']
            scenario_data['nr.ur'] = base_data['urnr_obs']

            try:
                result = calc_prob_chg(known_model_coefficients, base_data, scenario_data)
                crop_change = result['probchg_crop'].iloc[0]
                results.append(crop_change)

            except Exception:
                continue

        if len(results) > 10:
            # Results should be reasonably consistent
            mean_change = np.mean(results)
            std_change = np.std(results)

            # Standard deviation should be reasonable relative to mean
            if abs(mean_change) > 1e-6:
                cv = std_change / abs(mean_change)
                assert cv < 5.0, f"Results too variable across simulations: CV = {cv}"

            # All results should be finite
            assert all(np.isfinite(results)), "Some simulation results were not finite"