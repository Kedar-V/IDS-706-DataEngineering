import sys
import os
sys.path.append("../data")
sys.path.append("../dao")
from bitcoin_scraper import BitcoinHistoryFetcher
from btc_dao import BitcoinOHLCDAO
from datetime import datetime

class BitcoinPriceIngestor:
    def __init__(self, dao_host='btc-mysql', dao_user='root', dao_password='example'):
        self.fetcher = BitcoinHistoryFetcher()
        self.dao = BitcoinOHLCDAO(host=dao_host, user=dao_user, password=dao_password)

    def ingest(self, period='3d', interval='1d'):
        hist = self.fetcher.fetch_history(period=period, interval=interval)
        today = datetime.now().date()
        filtered_hist = hist[hist.index.date < today]
        rows = []
        for idx, row in filtered_hist.iterrows():
            timestamp = idx.to_pydatetime()
            open_ = float(row['Open'])
            high = float(row['High'])
            low = float(row['Low'])
            close = float(row['Close'])
            volume = float(row['Volume'])
            rows.append((timestamp, open_, high, low, close, volume))
        self.dao.insert_data(rows)
        print(f"Inserted {len(rows)} rows into MySQL.")

    def close(self):
        self.dao.close()

if __name__ == "__main__":
    ingestor = BitcoinPriceIngestor()
    ingestor.ingest(period='1y', interval='1d')
    ingestor.close()