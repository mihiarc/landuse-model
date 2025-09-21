# Land Use Modeling Test Suite

This directory contains comprehensive tests for the landuse-modeling Python package, ensuring reliability and correctness of the statistical models and data processing pipeline.

## Test Structure

### Core Test Modules

1. **`test_data_loading.py`** - Data loading and validation
   - File format handling (CSV, RDS, pickle)
   - FIPS code validation and conversion
   - Configuration management
   - Data type consistency

2. **`test_logit_estimation.py`** - Multinomial logit model estimation
   - Model convergence and parameter validation
   - Prediction accuracy and probability bounds
   - Statistical significance testing
   - Edge case handling

3. **`test_marginal_effects.py`** - Marginal effects and elasticity calculations
   - Mathematical accuracy of derivatives
   - Weighted vs unweighted marginal effects
   - Elasticity computation and unit independence
   - Temporal analysis consistency

4. **`test_climate_impact.py`** - Climate change impact analysis
   - Utility calculation accuracy
   - Probability change conservation
   - Scenario impact calculations
   - Spatial aggregation functions

5. **`test_data_exploration.py`** - Data exploration utilities
   - NRI data analysis
   - Metropolitan population analysis
   - Transition matrix generation
   - Spatial pattern analysis

6. **`test_integration.py`** - Full pipeline integration tests
   - End-to-end workflow validation
   - Configuration management
   - Error handling and recovery
   - Output file generation

7. **`test_performance_regression.py`** - Performance and regression tests
   - Performance benchmarks
   - Regression against R implementation
   - Numerical stability checks
   - Memory usage monitoring

### Test Configuration

- **`conftest.py`** - Shared fixtures and test data generators
- **`pytest.ini`** - Pytest configuration with markers and settings

## Running Tests

### Basic Test Execution

```bash
# Install test dependencies
uv pip install pytest pytest-cov

# Run all tests
uv run python -m pytest

# Run with coverage
uv run python -m pytest --cov=landuse --cov-report=html

# Run specific test module
uv run python -m pytest tests/test_data_loading.py

# Run specific test
uv run python -m pytest tests/test_data_loading.py::TestDataLoading::test_fips_code_conversion
```

### Test Categories

Tests are organized using pytest markers:

```bash
# Run only unit tests (default)
uv run python -m pytest -m "not slow and not integration"

# Run integration tests
uv run python -m pytest -m integration

# Run performance tests
uv run python -m pytest -m performance

# Run regression tests
uv run python -m pytest -m regression

# Skip slow tests
uv run python -m pytest -m "not slow"
```

### Parallel Execution

For faster execution on larger test suites:

```bash
# Install pytest-xdist for parallel execution
uv pip install pytest-xdist

# Run tests in parallel
uv run python -m pytest -n auto
```

## Test Data and Fixtures

### Generated Test Data

Tests use automatically generated synthetic data that mimics the structure of real land use datasets:

- **Geographic Reference Data**: County-level FIPS codes, regions, and classifications
- **Net Returns Data**: Economic returns for different land uses across years
- **Land Use Transition Data**: Starting and ending land use classifications
- **Climate Impact Data**: Simulated climate change effects on land use returns

### Key Fixtures

- `sample_georef_data`: Geographic reference data for testing
- `sample_net_returns_data`: Net returns across land use types
- `sample_start_data`: Land use transition starting conditions
- `temp_test_files`: Temporary files for I/O testing
- `known_model_coefficients`: Reference coefficients for mathematical validation

## Critical Test Coverage Areas

### 1. Statistical Correctness
- Multinomial logit parameter estimation accuracy
- Marginal effects calculation validation
- Probability conservation laws
- Elasticity computation correctness

### 2. Data Processing Reliability
- File format compatibility (CSV, RDS, pickle)
- Missing value handling
- Data type consistency
- FIPS code validation

### 3. Mathematical Properties
- Utility function linearity
- Probability bounds (0 ≤ p ≤ 1)
- Conservation laws (Σ prob changes = 0)
- Numerical stability across scales

### 4. Integration and Performance
- Full pipeline execution
- Memory usage optimization
- Execution time benchmarks
- Error handling and recovery

### 5. Regression Testing
- Consistency with R implementation results
- Parameter sign validation
- Output format compatibility
- Numerical precision maintenance

## Quality Assurance Standards

### Testing Principles
1. **Real API Calls**: Tests use actual model estimation, not just mocks
2. **Mathematical Validation**: Statistical properties are verified
3. **Edge Case Coverage**: Boundary conditions and error states tested
4. **Performance Monitoring**: Execution time and memory usage tracked
5. **Data Quality Checks**: Input validation and output verification

### Expected Coverage Targets
- **Core Functions**: >90% line coverage
- **Statistical Models**: >85% branch coverage
- **Data Processing**: >95% line coverage
- **Integration Tests**: All major workflows covered

### Continuous Integration
Tests are designed to run reliably in CI environments with:
- Deterministic random seeds for reproducibility
- Timeout protection for long-running operations
- Graceful handling of missing dependencies
- Clear error reporting and debugging information

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `uv pip install -e ".[dev]"`
2. **File Not Found**: Tests create temporary files; ensure write permissions
3. **Numerical Precision**: Some tests may fail due to platform-specific floating point differences
4. **Memory Issues**: Large dataset tests may require sufficient RAM

### Debug Mode

Run tests with verbose output and debug information:

```bash
uv run python -m pytest -v -s --tb=long tests/test_logit_estimation.py
```

### Test-Specific Environment Variables

Some tests can be configured via environment variables:
- `LANDUSE_TEST_TIMEOUT`: Override default test timeout
- `LANDUSE_TEST_TOLERANCE`: Set numerical comparison tolerance
- `LANDUSE_SKIP_SLOW`: Skip performance benchmarks

## Contributing New Tests

When adding new functionality:

1. **Add Unit Tests**: Test individual functions in isolation
2. **Add Integration Tests**: Test component interactions
3. **Update Fixtures**: Add new test data as needed
4. **Document Test Cases**: Include docstrings explaining test purpose
5. **Mark Appropriately**: Use pytest markers for test categorization
6. **Validate Coverage**: Ensure new code is adequately tested

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<specific_functionality>`
- Fixtures: `<descriptive_name>_data` or `sample_<data_type>`

## Pre-Merge Testing Checklist

Before merging the Python port:

- [ ] All unit tests pass
- [ ] Integration tests validate full pipeline
- [ ] Performance tests show acceptable execution times
- [ ] Regression tests confirm R compatibility
- [ ] Statistical properties are mathematically validated
- [ ] Edge cases and error conditions are handled
- [ ] Code coverage meets minimum thresholds
- [ ] No memory leaks in long-running operations