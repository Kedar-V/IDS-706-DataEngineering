# import MySQL connector
import mysql.connector
import json


class NewsDAO:
    def __init__(self, host, user, password, database="btc"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self.table = "daily_news"
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
                title VARCHAR(255) NOT NULL,
                link TEXT NOT NULL,
                author VARCHAR(255),
                published_date DATETIME,
                key_takeaways TEXT,
                UNIQUE KEY unique_link (link(255))  -- Specified key length for the `link` column
            )
        """
        )
        cursor.close()

    def insert_data(self, rows):
        """
        Inserts the provided rows into the database.

        :param rows: A list of tuples, where each tuple contains
                     (title, link, author, published_date, key_takeaways)
                     key_takeaways should be a list of strings.
        """
        cursor = self.conn.cursor()
        rows = [
            (title, link, author, published_date, json.dumps(key_takeaways))
            for title, link, author, published_date, key_takeaways in rows
        ]
        cursor.executemany(
            f"""
            INSERT IGNORE INTO {self.table} (title, link, author, published_date, key_takeaways)
            VALUES (%s, %s, %s, %s, %s)
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
        # Deserialize key_takeaways from JSON string to list
        return [
            (id, title, link, author, published_date, json.loads(key_takeaways))
            for id, title, link, author, published_date, key_takeaways in rows
        ]

    def close(self):
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    dao = NewsDAO(host="btc-mysql", user="root", password="example")

    # Example data to insert
    example_rows = [
        (
            "Bitcoin hits new high",
            "https://example.com/bitcoin-high",
            "John Doe",
            "2025-09-30 10:00:00",
            [
                "Bitcoin reached an all-time high.",
                "Institutional adoption is increasing.",
            ],
        ),
        (
            "Ethereum upgrade announced",
            "https://example.com/eth-upgrade",
            "Jane Smith",
            "2025-09-30 11:00:00",
            [
                "Ethereum's upgrade improves scalability.",
                "Transaction fees are reduced.",
            ],
        ),
    ]

    try:
        dao.insert_data(example_rows)
        rows = dao.fetch_all()
        for row in rows:
            print(row)
    finally:
        dao.close()
