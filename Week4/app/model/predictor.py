import pickle
import pandas as pd
import sys

sys.path.append("../..")

from Week2.Bitcoin_DataAnalysis import CryptoFeatureEngineer, DataFrameLoader


class Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        with open(self.model_path, "rb") as file:
            model = pickle.load(file)
        return model

    def predict(self, input_data):
        feature_engineer = CryptoFeatureEngineer(input_data)
        processed_data = (
            feature_engineer.basic_features()
            .aggregated_features(group_col="Day")
            .lag_features()
            .rolling_features()
            .rsi()
            .macd()
            .bollinger_bands()
            .get_df()
        )
        processed_data = processed_data.drop("Close")
        print(self.model.predict(processed_data)[0])
        return self.model.predict(processed_data)[0]


if __name__ == "__main__":

    # Sample input data
    loader = DataFrameLoader()
    df = loader.load("sql")

    predictor = Predictor(model_path="random_forest_model.pkl")
    predictions = predictor.predict(df)

    print("Predictions:", predictions)
