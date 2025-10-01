import sys
import pickle
import os

sys.path.append("../../../Week2")
from Bitcoin_DataAnalysis import (
    DataFrameLoader,
    CryptoFeatureEngineer,
    CryptoDatasetLoader,
    RandomForestRegressor,
)


class ModelTrainer:
    def __init__(self, model_path):
        self.loader = DataFrameLoader()
        self.model_path = model_path

    def prepare_features(self):
        df = self.loader.load("sql")
        print(df.head())
        fe = CryptoFeatureEngineer(df)
        self.df_feat = (
            fe.basic_features()
            .aggregated_features(group_col="Day")
            .lag_features()
            .rolling_features()
            .rsi()
            .macd()
            .bollinger_bands()
            .get_df()
        )
        print(self.df_feat.tail(10).to_pandas())

    def prepare_dataset(self):
        dataloader = CryptoDatasetLoader(
            self.df_feat,
            target_col="Close",
            scale=False,
            apply_pca=False,
            pca_variance=0.95,
            subset_size=100000,  # only use last 100000 rows
        )
        dataloader.create_target().filter_nulls()
        self.X_train, self.X_test, self.y_train, self.y_test = (
            dataloader.train_test_split()
        )

    def train_model(self):
        rf_model = RandomForestRegressor(
            n_estimators=100,  # fewer trees for a quick test
            max_depth=10,  # shallower trees, faster
            random_state=42,
            n_jobs=-1,  # use all CPU cores
            verbose=1,  # prints minimal progress info
        )
        rf_model.fit(self.X_train, self.y_train)
        print(self.df_feat.columns)
        with open(self.model_path, "wb") as file:
            pickle.dump(rf_model, file)


if __name__ == "__main__":
    model_path = os.path.join(
        os.path.dirname(__file__), "../model/random_forest_model.pkl"
    )
    trainer = ModelTrainer(model_path)
    trainer.prepare_features()
    trainer.prepare_dataset()
    trainer.train_model()
