import yfinance as yf


class BitcoinHistoryFetcher:
    """
    OOP class to fetch historical BTC-USD data using yfinance.
    """

    def __init__(self, ticker="BTC-USD"):
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)

    def fetch_history(self, period="1y", interval="1d"):
        return self.yf_ticker.history(period=period, interval=interval)
