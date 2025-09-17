# IDS-706-DataEngineering-Week3
# Index

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
  - [Dev Container (VS Code)](#dev-container-vs-code)
  - [Docker Container](#docker-container)
- [Bitcoin Data Analysis Testing](#bitcoin-data-analysis-testing)
  - [Key Python Modules](#key-python-modules)
  - [Pytest Fixtures](#pytest-fixtures)
  - [Test Overview](#test-overview)
- [How to Run Tests](#how-to-run-tests)
  - [1. Using Python directly](#1-using-python-directly)
  - [2. Using Makefile](#2-using-makefile)

---
# Project Structure
```
├── .devcontainer
├── Week1/
├── Week2/
│   └── Bitcoin_DataAnalysis.py
├── Week3/
│   ├── test_BitcoinDataAnalysis.py
│   └── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```


- **Week2/Bitcoin_DataAnalysis.py**: Contains the main classes for data loading, feature engineering, dataset preparation, and model evaluation.
- **Week3/test_BitcoinDataAnalysis.py**: Contains unit tests and system tests.
- **requirements.txt**: Python dependencies for Week3.
- **Dockerfile / docker-compose.yml**: Optional containerized environment for reproducible testing.
- **.devcontainer/**: Contains VS Code Dev Container configuration for a reproducible development environment.

---

# Environment Setup

## Dev Container (VS Code)

This project provides a **Dev Container** configuration for reproducible development and testing using VS Code. You can run all tests and Python scripts in a containerized environment without polluting your local machine.

### Steps to Use Dev Container

1. **Install VS Code Extensions**  
   Make sure you have the following installed in VS Code:
   - **Remote - Containers**  
     (Official extension to work with Dev Containers)

2. **Open the Project in VS Code**  
   - Open the folder containing this repository in VS Code.

3. **Open in Dev Container**  
   - Press `F1` → type `Remote-Containers: Open Folder in Container` → select the project folder.  
   - VS Code will build the container using the included `devcontainer.json` and `Dockerfile`.

4. **Post-create Commands**  
   - The Dev Container automatically installs dependencies from `Week3/requirements.txt`.

5. **Run Tests Inside Container**  
   Open a terminal inside VS Code (within the container) and run:

   ```bash
   cd Week3
   pytest -vv --cov=Bitcoin_DataAnalysis test_BitcoinDataAnalysis.py
    ```
---

## Docker Container

This project can also be run using a **Docker container** without relying on VS Code Dev Containers. This ensures a reproducible Python environment on any system with Docker installed.

### Steps to Use Docker Container

1. **Install Docker**  
   Ensure Docker is installed and running:

   ```bash
   docker --version
   ```
2. Build the Docker Image from the project root:
    ```bash
    docker build -t python_week3 .
    ```
    This uses the Dockerfile to create a container with Python 3.12 and dependencies from Week3/requirements.txt.

3. Run the Container
    ```bash
    docker run -it --name python_week3 -v $(pwd):/app python_week3
    ```

- `-v $(pwd):/app` mounts the project folder into the container.  
- It opens an interactive terminal.

4. To enter the running container, use:  
    ```bash
    docker exec -it python_week3 /bin/bash
    ```

5. Alternative: Using Docker Compose

    You can also use docker-compose for easier management:
    ```bash
    docker-compose up -d
    ```
    - This builds and starts the container in detached mode.

    Access the container with:

    ```bash
    docker compose exec python /bin/bash
    ```

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
