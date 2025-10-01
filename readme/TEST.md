# Bitcoin Data Analysis Testing

This repository contains Python modules and tests for Bitcoin data analysis, feature engineering, dataset preparation, and model evaluation. The tests use **pytest**, **pandas**, **polars**, **numpy**, and **scikit-learn**.

---

## Key Python Modules

### Classes under test:

- **DataFrameLoader**: Load CSV data into `pandas` or `polars` DataFrames.
- **CryptoFeatureEngineer**: Compute basic, lag, and rolling features plus advanced indicators (RSI, MACD, Bollinger Bands).
- **CryptoDatasetLoader**: Prepare features and targets for machine learning; supports scaling, PCA, subsetting, and null filtering.
- **ModelEvaluator**: Evaluate regression models with metrics, residuals, feature importance, and plots.

---

## Pytest Fixtures

The tests use reusable fixtures:

- **sample_csv**: Temporary CSV file with enough rows for system tests.
- **ohlcv_df**: Polars DataFrame with OHLCV data for feature engineering tests.
- **dataset_df**: Polars DataFrame for CryptoDatasetLoader tests.
- **dummy_model_data**: Small dataset with a fitted LinearRegression model.
- **dummy_model_df**: Polars DataFrame including timestamp for residual-over-time tests.

---
## Test Overview

| Category | Test Function | Description |
|----------|---------------|-------------|
| **DataFrameLoader Tests** | `test_load_pandas` | Load CSV as Pandas DataFrame. |
|  | `test_load_polars` | Load CSV as Polars DataFrame. |
|  | `test_invalid_library` | Raise error for unsupported library. |
| **CryptoFeatureEngineer Tests** | `test_basic_features` | Compute return, spreads, and candle features. |
|  | `test_lag_features` | Generate lagged features. |
|  | `test_rsi_macd_bollinger` | Compute technical indicators (RSI, MACD, Bollinger Bands). |
| **CryptoDatasetLoader Tests** | `test_create_target` | Verify next-period Close as target. |
|  | `test_filter_nulls` | Drop rows with null values. |
|  | `test_subset` | Keep last N rows. |
|  | `test_scaling` | Normalize features and target to mean 0, std 1. |
|  | `test_pca` | Apply PCA and retain ≥95% variance. |
| **ModelEvaluator Tests** | `test_predict_and_residuals` | Verify predictions and residuals. |
|  | `test_compute_metrics` | Validate RMSE and R² for perfect fit. |
|  | `test_feature_importance_plot` | Plot feature importances. |
|  | `test_residual_plots` | Plot residuals without exceptions. |
|  | `test_residuals_over_time` | Plot residuals over time with timestamp. |
|  | `test_residuals_over_time_missing_timestamp` | Handle missing timestamp gracefully. |
| **System Tests** | `test_full_pipeline_system` | End-to-end test for full data loading, feature engineering, dataset preparation, model training, and evaluation. |

---

## How to Run Tests

### 1. Using Python directly:

```bash
cd Week3
pip install -r requirements.txt
pytest -vv --cov=Bitcoin_DataAnalysis test_BitcoinDataAnalysis.py
```

### 2. Using Makefile:

The Makefile includes convenient targets for installing dependencies, running tests, formatting code, and more.

### 2. Using the Makefile

The Makefile provides convenient targets to install dependencies, run tests, format code, and more.

1. Install dependencies:

    ```bash
    make install
    ```

2. Run the tests with coverage:

    ```bash
    make test
    ```
