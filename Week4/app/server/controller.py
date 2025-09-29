import pandas as pd

class BitcoinOHLCController:
    def __init__(self, model):
        self.model = model

    def get_ohlc_df(self):
        rows, error = self.model.fetch_all()
        if error:
            return pd.DataFrame(), error
        if rows:
            df = pd.DataFrame(rows, columns=["id", "timestamp", "open", "high", "low", "close", "volume"])
            return df, None
        return pd.DataFrame(), None