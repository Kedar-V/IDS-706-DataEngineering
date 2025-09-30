import sys
import os
sys.path.append('../dao')
from btc_dao import BitcoinOHLCDAO

class BitcoinOHLCModel:
    def __init__(self, host='btc-mysql', user='root', password='example'):
        try:
            self.dao = BitcoinOHLCDAO(host=host, user=user, password=password)
        except Exception as e:
            self.dao = None
            self.error = str(e)
            print(self.error)

    def fetch_all(self):
        if not self.dao:
            return [], getattr(self, 'error', 'DAO not available')
        try:
            return self.dao.fetch_all(), None
        except Exception as e:
            return [], str(e)

    def close(self):
        if self.dao:
            self.dao.close()