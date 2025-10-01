import sys

sys.path.append("../data")
sys.path.append("../dao")
from news_scrapper import NewsScraper
from news_dao import NewsDAO
from datetime import datetime


class NewsIngestor:
    def __init__(self, scraper: NewsScraper, dao: NewsDAO):
        self.scraper = scraper
        self.dao = dao

    def ingest_news(self, url):
        try:

            articles = self.scraper.scrape_articles(url)

            for article in articles:
                article["takeaways"] = self.scraper.scrape_key_takeaways(
                    article["link"]
                )

            rows = [
                (
                    article["title"],
                    article["link"],
                    article["author"],
                    datetime.now(),
                    article["takeaways"],
                )
                for article in articles
            ]

            self.dao.insert_data(rows)

        except Exception as e:
            print(f"Error during ingestion: {e}")


if __name__ == "__main__":
    url = "https://cointelegraph.com/tags/bitcoin"
    scraper = NewsScraper()
    dao = NewsDAO(host="btc-mysql", user="root", password="example")
    ingestor = NewsIngestor(scraper, dao)

    try:
        ingestor.ingest_news(url)
        rows = dao.fetch_all()
        for row in rows:
            print(row)
    finally:
        dao.close()
