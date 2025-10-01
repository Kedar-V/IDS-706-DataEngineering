import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


class NewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/114.0.0.0 Safari/537.36"
                )
            }
        )

    def scrape_articles(self, url):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("a.post-card-inline__figure-link")
        scraped_data = []

        for link_tag in articles:
            parent = link_tag.find_parent("li")
            if not parent:
                continue

            title_tag = parent.select_one(".post-card-inline__title")
            author_tag = parent.select_one(".post-card-inline__author")

            if not (title_tag and link_tag and author_tag):
                continue

            title = title_tag.get_text(strip=True)
            link = urljoin(url, link_tag.get("href"))
            author = author_tag.get_text(strip=True)
            scraped_data.append({"title": title, "link": link, "author": author})

        return scraped_data

    def scrape_key_takeaways(self, article_url):
        try:
            response = self.session.get(article_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch {article_url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        takeaways_header = soup.find(
            lambda tag: tag.name == "h2"
            and "key takeaways" in tag.get_text(strip=True).lower()
        )
        if not takeaways_header:
            return []

        ul_tag = takeaways_header.find_next_sibling("ul")
        if not ul_tag:
            return []

        return [li.get_text(strip=True) for li in ul_tag.find_all("li")]


if __name__ == "__main__":
    url = "https://cointelegraph.com/tags/bitcoin"
    scraper = NewsScraper()
    data = {}

    articles = scraper.scrape_articles(url)
    for article in articles:
        print(article["title"], article["link"], article["author"])
        data[article["link"]] = article

    for link in data.keys():
        data[link]["takeaways"] = scraper.scrape_key_takeaways(link)

    print("---------------")
    print(data)
