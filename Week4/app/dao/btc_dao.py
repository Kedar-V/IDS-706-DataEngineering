# import MySQL connector
import mysql.connector


class BitcoinOHLCDAO:
    def __init__(self, host, user, password, database="btc"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self.table = "bitcoin_daily_price"
        self._connect()
        self._create_schema()

    def _connect(self):
        self.conn = mysql.connector.connect(
            host=self.host, user=self.user, password=self.password
        )
        self._create_database()
        self.conn.database = self.database

    def _create_database(self):
        cursor = self.conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        cursor.close()

    def _create_schema(self):
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                open DECIMAL(18,8) NOT NULL,
                high DECIMAL(18,8) NOT NULL,
                low DECIMAL(18,8) NOT NULL,
                close DECIMAL(18,8) NOT NULL,
                volume DECIMAL(18,8) NOT NULL,
                UNIQUE KEY unique_timestamp (timestamp)
            )
        """
        )
        cursor.close()

    def insert_data(self, rows):
        """
        Inserts the provided rows into the database.

        :param rows: A list of tuples, where each tuple contains
                                 (timestamp, open, high, low, close, volume)
        """
        cursor = self.conn.cursor()
        cursor.executemany(
            f"""
            INSERT IGNORE INTO {self.table} (timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        self.conn.commit()
        cursor.close()

    def fetch_all(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {self.table}")
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def close(self):
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    dao = BitcoinOHLCDAO(host="btc-mysql", user="root", password="example")
    dao._create_database()
    dao._create_schema()
    dao.insert_dummy_data()
    rows = dao.fetch_all()
    for row in rows:
        print(row)

    dao.close()
