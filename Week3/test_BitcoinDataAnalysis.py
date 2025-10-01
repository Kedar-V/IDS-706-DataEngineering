import os
import pytest
import pandas as pd
import polars as pl
import numpy as np
from tempfile import NamedTemporaryFile
from sklearn.linear_model import LinearRegression

from Week2.Bitcoin_DataAnalysis import (
    DataFrameLoader,
    CryptoFeatureEngineer,
    CryptoDatasetLoader,
    ModelEvaluator,
)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_csv():
    """Create temporary CSV with enough rows for system test."""
    data = "Timestamp,Open,High,Low,Close,Volume\n"
    for i in range(1, 21):
        data += f"{i},{100+i},{105+i},{95+i},{102+i},{10*i}\n"
    tmp = NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(data.encode("utf-8"))
    tmp.close()
    yield tmp.name
    os.remove(tmp.name)


@pytest.fixture
def ohlcv_df():
    """Polars DataFrame with OHLCV data for feature engineering tests."""
    return pl.DataFrame(
        {
            "Timestamp": [1, 2, 3, 4, 5],
            "Open": [100, 102, 104, 106, 108],
            "High": [105, 107, 109, 111, 113],
            "Low": [95, 97, 99, 101, 103],
            "Close": [102, 104, 106, 108, 110],
            "Volume": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def dataset_df():
    """Polars DataFrame with OHLCV for CryptoDatasetLoader tests."""
    return pl.DataFrame(
        {
            "Timestamp": [1, 2, 3, 4, 5],
            "Open": [100, 101, 102, 103, 104],
            "High": [110, 111, 112, 113, 114],
            "Low": [90, 91, 92, 93, 94],
            "Close": [105, 106, 107, 108, 109],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }
    )


@pytest.fixture
def dummy_model_data():
    """Return small dataset and a fitted LinearRegression model for ModelEvaluator tests."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = LinearRegression()
    model.fit(X, y)
    return X, y, model


@pytest.fixture
def dummy_model_df(dummy_model_data):
    """Polars DataFrame with timestamp column for residual-over-time tests."""
    X, y, _ = dummy_model_data
    return pl.DataFrame(
        {"Timestamp": [1, 2, 3, 4, 5], "Feature": X.flatten(), "Target": y}
    )


# -----------------------------
# DataFrameLoader Tests
# -----------------------------
class TestDataFrameLoader:
    """Tests for DataFrameLoader class."""

    def test_load_pandas(self, sample_csv):
        """Test loading CSV into a Pandas DataFrame."""
        loader = DataFrameLoader(sample_csv)
        df = loader.load("pandas")

        assert isinstance(df, pd.DataFrame)
        assert "Timestamp" in df.columns
        assert df["Timestamp"].dtype == "int64"
        assert df.loc[0, "Open"] == 101

    def test_load_polars(self, sample_csv):
        """Test loading CSV into a Polars DataFrame."""
        loader = DataFrameLoader(sample_csv)
        df = loader.load("polars")

        assert isinstance(df, pl.DataFrame)
        assert "Timestamp" in df.columns
        assert df["Timestamp"].dtype == pl.Int64
        assert df["Open"][0] == 101

    def test_invalid_library(self, sample_csv):
        """Test that an unsupported library raises ValueError."""
        loader = DataFrameLoader(sample_csv)
        with pytest.raises(ValueError, match="Unsupported library"):
            loader.load("random")


# -----------------------------
# CryptoFeatureEngineer Tests
# -----------------------------
class TestCryptoFeatureEngineer:
    """Tests for CryptoFeatureEngineer class."""

    def test_basic_features(self, ohlcv_df):
        """Test basic feature calculations: return, spreads, candle info."""
        fe = CryptoFeatureEngineer(ohlcv_df).basic_features()
        df = fe.get_df()

        expected_cols = [
            "return",
            "high_low_spread",
            "open_close_diff",
            "candle_body",
            "candle_upper_shadow",
            "candle_lower_shadow",
        ]
        for col in expected_cols:
            assert col in df.columns

        # Sanity check: high-low spread
        assert df["high_low_spread"][0] == 10

    def test_lag_features(self, ohlcv_df):
        """Test lagged feature creation."""
        fe = CryptoFeatureEngineer(ohlcv_df).lag_features(lags=[1, 2])
        df = fe.get_df()
        assert "Close_lag_1" in df.columns
        assert "Volume_lag_2" in df.columns
        # Lag value matches previous row
        assert df["Close_lag_1"][1] == df["Close"][0]

    def test_rsi_macd_bollinger(self, ohlcv_df):
        """Test advanced technical indicators: RSI, MACD, Bollinger Bands."""
        fe = (
            CryptoFeatureEngineer(ohlcv_df)
            .rsi(window=3)
            .macd()
            .bollinger_bands(window=3)
        )
        df = fe.get_df()
        assert "RSI_3" in df.columns
        assert "MACD" in df.columns
        assert "MACD_signal" in df.columns
        assert "BB_upper" in df.columns


# -----------------------------
# CryptoDatasetLoader Tests
# -----------------------------
class TestCryptoDatasetLoader:
    """Tests for CryptoDatasetLoader class."""

    def test_create_target(self, dataset_df):
        """Test that next-period Close is created as target."""
        loader = CryptoDatasetLoader(dataset_df).create_target()
        assert "target" in loader.df.columns
        assert loader.df["target"][0] == dataset_df["Close"][1]

    def test_filter_nulls(self, dataset_df):
        """Test that rows with nulls in essential features are dropped."""
        df_with_null = dataset_df.with_columns(
            pl.Series("Close", [105, None, 107, 108, 109])
        )
        loader = CryptoDatasetLoader(df_with_null).create_target().filter_nulls()
        assert loader.df.height < df_with_null.height
        assert loader.df["Close"].is_null().sum() == 0

    def test_subset(self, dataset_df):
        """Test subset functionality: only last N rows are kept."""
        loader = CryptoDatasetLoader(dataset_df, subset_size=3).create_target()
        X, y = loader.get_features_targets()
        assert X.shape[0] == 3
        assert y.shape[0] == 3

    def test_scaling(self, dataset_df):
        """Test that scaling normalizes features and target to mean 0, std 1."""
        loader = CryptoDatasetLoader(dataset_df, scale=True).create_target()
        X, y = loader.get_features_targets()
        assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
        assert np.allclose(X.std(axis=0), 1, atol=1e-8)
        assert np.allclose(y.mean(), 0, atol=1e-8)

    def test_pca(self, dataset_df):
        """Test PCA reduces dimensionality while retaining ~95% variance."""
        loader = CryptoDatasetLoader(dataset_df, apply_pca=True).create_target()
        X, y = loader.get_features_targets()
        assert loader.pca is not None
        assert X.shape[1] <= len(dataset_df.columns) - 2  # exclude Timestamp & target
        explained = loader.pca.explained_variance_ratio_.sum()
        assert explained >= 0.95


# -----------------------------
# ModelEvaluator Tests
# -----------------------------
class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_predict_and_residuals(self, dummy_model_data):
        """Test predictions and residual calculation."""
        X, y, model = dummy_model_data
        evaluator = ModelEvaluator(model, X, y)
        y_pred = evaluator.predict()
        residuals = evaluator.residuals
        np.testing.assert_array_almost_equal(residuals, y - y_pred)

    def test_compute_metrics(self, dummy_model_data):
        """Test RMSE and R² metrics for perfect linear fit."""
        X, y, model = dummy_model_data
        evaluator = ModelEvaluator(model, X, y)
        rmse, r2 = evaluator.compute_metrics()
        assert np.isclose(rmse, 0)
        assert np.isclose(r2, 1.0)

    def test_feature_importance_plot(self, dummy_model_data):
        """Test plotting of feature importances (with mock attribute)."""
        X, y, model = dummy_model_data
        model.feature_importances_ = np.array([0.1])
        evaluator = ModelEvaluator(model, X, y, feature_names=["Feature"])
        evaluator.plot_feature_importance(top_n=1)

    def test_residual_plots(self, dummy_model_data):
        """Test that residual plots run without exceptions."""
        X, y, model = dummy_model_data
        evaluator = ModelEvaluator(model, X, y)
        evaluator.plot_residuals()

    def test_residuals_over_time(self, dummy_model_data, dummy_model_df):
        """Test residuals over time plotting with timestamp column."""
        X, y, model = dummy_model_data
        evaluator = ModelEvaluator(
            model, X, y, df=dummy_model_df, timestamp_col="Timestamp"
        )
        evaluator.plot_residuals_over_time()

    def test_residuals_over_time_missing_timestamp(self, dummy_model_data):
        """Test residuals over time gracefully handles missing timestamp."""
        X, y, model = dummy_model_data
        evaluator = ModelEvaluator(model, X, y, df=None, timestamp_col="Timestamp")
        evaluator.plot_residuals_over_time()


class TestSystem:
    def test_full_pipeline_system(self, sample_csv):
        # Load data
        df = DataFrameLoader(sample_csv).load("polars")

        # Feature engineering
        fe = (
            CryptoFeatureEngineer(df).basic_features().lag_features().rolling_features()
        )
        df_fe = fe.get_df()

        # Dataset loader
        loader = CryptoDatasetLoader(df_fe, test_size=0.3).create_target()
        X, y = loader.get_features_targets()
        X_train, X_test, y_train, y_test = loader.train_test_split()

        # Train a simple model
        model = LinearRegression().fit(X_train, y_train)

        # Evaluate
        evaluator = ModelEvaluator(model, X_test, y_test, df=df_fe)
        rmse, r2 = evaluator.compute_metrics()

        # Simple assertions
        assert rmse >= 0
        assert -1 <= r2 <= 1
