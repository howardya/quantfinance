# Test Coverage Analysis

## Current State

The project enforces **100% line coverage** in CI (`pytest --cov --cov-fail-under=100`).
However, there is a significant discrepancy: the `.coveragerc` file sets
`source = tensorflow_quant` while the actual package is named `quantfinance`. This
means **coverage is likely being measured against the wrong (or empty) package name**,
and the 100% gate may be passing vacuously. This should be verified and fixed first.

### Existing Test Inventory

| Source Module | Test File | Active Test Count |
|---|---|---|
| `covariance.py` (225 lines) | `test_covariance.py` | 13 |
| `data.py` (22 lines) | `test_data.py` | 4 |
| `gaussian_process_lvm.py` (172 lines) | `test_gaussian_process_lvm.py` | 3 |
| `optimization.py` (86 lines) | `test_optimization.py` | 11 |
| `portfolio_ica.py` (14 lines) | `test_portfolio_ica.py` | 2 |
| `portfolio_optimization.py` (156 lines) | `test_portfolio_optimization.py` | 7 |
| `positive_pca.py` (58 lines) | `test_positive_pca.py` | 2 |
| `probabilistic_pca.py` (173 lines) | `test_probabilistic_pca.py` | 7 |
| `strategies/learn_to_rank.py` (147 lines) | `test_learn_to_rank_stocks.py` | 5 |
| `strategies/tiny_cta.py` (31 lines) | `test_tiny_cta.py` | 1 |
| `tensorflow_quant.py` (5 lines) | `test_tensorflow_quant.py` | 1 |
| `univariate_gmm_constraints.py` (194 lines) | `test_univariate_gmm_constraints.py` | 3 |
| **`library.py` (1,406 lines)** | **None** | **0** |
| **`strategies/trend_MA.py` (9 lines)** | **None** | **0** |
| **`strategies/usbonds_from_jpeq.py` (129 lines)** | **None** | **0** |
| **`utils/get_sharpe_ratio.py` (6 lines)** | **None** | **0** |

---

## Priority 1: Infrastructure / Configuration Issues

### 1. `.coveragerc` references wrong package name

`.coveragerc` specifies `source = tensorflow_quant`, but the package directory is
`quantfinance/`. This means `pytest --cov` is probably not measuring coverage of the
actual source code at all. The CI gate of `--cov-fail-under=100` may be passing
trivially.

**Fix:** Change `.coveragerc` to `source = quantfinance` and re-run to get a true
baseline.

### 2. Docker smoke test references wrong image tag

`.github/workflows/test.yml` line 41 runs
`docker run --rm tensorflow_quant:test 10` but the image is built as
`quantfinance:test` on line 37. This smoke test will always fail.

---

## Priority 2: Completely Untested Modules

### 1. `library.py` (1,406 lines) — No tests

This is the largest file in the project. It contains only constant data (ticker
symbol lists: `DOW_30_TICKER`, `NAS_100_TICKER`, `SP_500_TICKER`, `HSI_50_TICKER`,
`SSE_50_TICKER`, `CSI_300_TICKER`, `CAC_40_TICKER`, `DAX_30_TICKER`,
`TECDAX_TICKER`, `MDAX_50_TICKER`, `SDAX_50_TICKER`, `FX_TICKER`,
`LQ45_TICKER`).

While there is no functional logic to test, this module is imported and used (e.g.,
`test_tiny_cta.py` imports `DOW_30_TICKER`). Basic sanity tests would help catch
data integrity issues:

- Assert each ticker list is non-empty and has the expected length
- Assert no duplicates within each list
- Note: `FX_TICKER` currently has duplicates (e.g., `AUDUSD=X` appears 7 times,
  `EURCHF=X` appears 3 times, `GBPAUD=X` appears 3 times, `EURTRY=X` appears
  2 times). This is likely a data bug.

### 2. `strategies/usbonds_from_jpeq.py` (129 lines) — No tests

Contains three functions and a `main()` entry point:
- `get_data_investpy()` — Fetches ETF data from investpy
- `get_data_yf()` — Fetches ETF data from Yahoo Finance
- `backtest(data, rebal_frequency)` — Runs a trend-following backtest on US bonds
  using Japan equity signals, with both weekly and daily rebalancing modes
- `main()` — Orchestrates the full pipeline

The `backtest()` function has meaningful logic (SMA calculation, signal generation,
weekly/daily rebalancing paths) that should be tested with synthetic data. The data
fetching functions can be tested with mocks or integration tests.

### 3. `strategies/trend_MA.py` (9 lines) — No tests

Contains a single `get_data()` function that calls `get_stocks_prices_yahoo`. This
is a thin wrapper but is completely untested. It also is not exported from the
strategies `__init__.py`.

### 4. `utils/get_sharpe_ratio.py` (6 lines) — No tests

Contains `get_sharpe_ratio(s)` which computes annualized Sharpe ratio. Despite being
simple, it is used by other modules (`usbonds_from_jpeq.py`) and should have basic
unit tests to verify the math (known inputs -> expected outputs, handling of zero
standard deviation, etc.).

---

## Priority 3: Gaps in Existing Test Coverage

### 1. `covariance.py` — Missing invalid method test

`get_covariance()` silently falls through if an unsupported `method` string is
passed (no `else` clause). There is no test for invalid method names. The function
will reach `diagonal_inv_vol` with an undefined `covariance` variable, causing an
`UnboundLocalError`. A test should verify this edge case, and the code should raise
a `ValueError` for unknown methods.

### 2. `covariance.py` — No test for correlated data

All `generate_multivariate_data()` calls use `correlated=False`. The helper supports
`correlated=True` (generating correlated data via Cholesky LKJ), but this path is
never exercised in tests. DCC-GARCH and CCC-GARCH are specifically designed for
correlated time series, so testing with correlated inputs is important.

### 3. `covariance.py` — No test for `get_covariance` with higher dimensions

All `get_covariance` tests use `data_dim=2`. Testing with higher dimensional data
(e.g., 5-10 dimensions) would be valuable to verify matrix operations scale
correctly.

### 4. `data.py` — No test for invalid frequency

`get_stocks_prices_yahoo()` raises `NotImplementedError` for invalid frequency
strings, but there is no test covering this branch.

### 5. `data.py` — No test for multiple symbols

All tests pass a single symbol string `"AAPL"`. Other test files (e.g.,
`test_tiny_cta.py`) do pass lists, but `test_data.py` itself never tests the
multi-symbol path explicitly.

### 6. `optimization.py` — No higher-dimensional tests

All tests use 1D or low-dimensional functions. Testing with higher-dimensional
optimization problems would exercise different code paths in the underlying
optimizers.

### 7. `gaussian_process_lvm.py` — No test for invalid kernel type

The function raises `NotImplementedError` for unsupported `kernel_type` values, but
no test covers this branch.

### 8. `gaussian_process_lvm.py` — No test for custom kernel

The function accepts a user-provided `kernel` object, but all tests pass
`kernel=None` and rely on built-in kernels.

### 9. `univariate_gmm_constraints.py` — Commented-out tests for 3, 4, 5 mixtures

Tests for 3+ mixtures are all commented out (lines 71-187 and 238-284 in
`test_univariate_gmm_constraints.py`). This means only 2-mixture scenarios are
tested, leaving the more complex multi-component GMM fitting untested.

### 10. `portfolio_optimization.py` — No test for `sharpe_loss_generator` edge cases

The Sharpe loss function divides by volatility, which could be zero for constant
returns. No test exercises this edge case.

### 11. `tiny_cta.py` — `osc()` function not directly tested

The `osc()` oscillator function is only tested implicitly through
`tiny_cta_backtest()`. A direct unit test with known inputs would be valuable.

### 12. `positive_pca.py` — No test for convergence failure

The function handles `results.success == False` with a print statement but the path
is never tested. Also no test for `n_components=None`.

### 13. `probabilistic_pca.py` — No test with custom optimizer

The `optimizer` parameter is accepted but never tested with a non-default value.

---

## Priority 4: Test Quality Improvements

### 1. Tests rely on live network calls

Multiple test files call `get_stocks_prices_yahoo()` with real tickers and date
ranges, making tests dependent on Yahoo Finance API availability. Flaky network
conditions can cause CI failures. Consider:
- Adding `@pytest.mark.integration` markers to separate network-dependent tests
- Using `unittest.mock.patch` or fixtures with cached data for unit tests
- Creating a shared test fixture with pre-downloaded sample data

Affected test files: `test_data.py`, `test_tiny_cta.py`,
`test_learn_to_rank_stocks.py`, `test_portfolio_optimization.py`,
`test_positive_pca.py`, `test_portfolio_ica.py`

### 2. Loose assertions with wide tolerance

Many tests use `abs(x - expected) < 0.1` or `< 0.2` as the only assertion. While
understandable for stochastic algorithms, some deterministic properties could be
asserted exactly:
- Covariance matrices should be symmetric
- Correlation diagonals should be exactly 1.0
- Portfolio weights from softmax should sum to 1.0
- PCA components should be orthogonal

### 3. No negative / error-path tests for most modules

Error handling is only tested in `optimization.py` and `covariance.py`. Most other
modules have no negative tests at all.

### 4. No property-based testing

For mathematical functions, property-based testing (e.g., with `hypothesis`) could
verify invariants:
- Covariance matrices are always positive semi-definite
- Portfolio weights are always non-negative after softmax
- PCA reconstructions preserve total variance

### 5. Test data generation is not reproducible in all files

Some test files set `tf.random.set_seed()` / `np.random.seed()` and some do not
(e.g., `test_positive_pca.py`, `test_portfolio_ica.py`). This can cause
intermittent failures.

---

## Summary of Recommended Actions

| Priority | Action | Impact |
|---|---|---|
| **P0** | Fix `.coveragerc` to use `source = quantfinance` | Enables real coverage measurement |
| **P0** | Fix Docker smoke test image tag in CI | Fixes broken CI step |
| **P1** | Add tests for `utils/get_sharpe_ratio.py` | Covers untested utility |
| **P1** | Add tests for `strategies/usbonds_from_jpeq.py` `backtest()` | Covers untested strategy logic (129 lines) |
| **P1** | Add data integrity tests for `library.py` ticker lists | Catches duplicate ticker bugs |
| **P1** | Add invalid method test for `get_covariance()` | Catches silent failure bug |
| **P2** | Add invalid frequency test for `get_stocks_prices_yahoo()` | Covers error branch |
| **P2** | Add invalid kernel_type test for `gaussian_process_lvm()` | Covers error branch |
| **P2** | Uncomment or rewrite 3+ mixture GMM tests | Covers multi-component GMM |
| **P2** | Add correlated data tests for covariance module | Tests the primary use case |
| **P2** | Add `osc()` direct unit test | Covers function independently |
| **P3** | Mock network calls in unit tests | Reduces flakiness |
| **P3** | Add property-based invariant assertions | Increases assertion quality |
| **P3** | Standardize random seed usage across all test files | Improves reproducibility |
