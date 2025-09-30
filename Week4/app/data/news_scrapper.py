import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class NewsScraper:
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

    def scrape_articles(self, url):
        self.driver.get(url)
        articles = self.driver.find_elements(By.CSS_SELECTOR, "li[data-testid='posts-listing__item']")
        scraped_data = []

        for article in articles:
            title = article.find_element(By.CSS_SELECTOR, ".post-card-inline__title").text
            link = article.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
            author = article.find_element(By.CSS_SELECTOR, ".post-card-inline__author").text
            scraped_data.append({"title": title, "link": link, "author": author})

        return scraped_data
    
    def scrape_key_takeaways(self, article_url):
        try:
            self.driver.get(article_url)
            time.sleep(2)  # wait for content to load (tune as needed)

            # Look for Key takeaways section
            header = self.driver.find_elements(By.XPATH, "//h2[contains(text(),'Key takeaways')]")
            if not header:
                return []

            # get list items under Key takeaways
            takeaways = self.driver.find_elements(By.XPATH, "//h2[contains(text(),'Key takeaways')]/following-sibling::ul[1]/li")
            return [t.text for t in takeaways]

        except Exception as e:
            print(f"Error scraping {article_url}: {e}")
            return []

    def close_driver(self):
        self.driver.quit()

if __name__ == "__main__":
    url = 'https://cointelegraph.com/tags/bitcoin'
    scraper = NewsScraper()
    data = {}
    try:
        articles = scraper.scrape_articles(url)
        for article in articles:
            print(article["title"], article["link"], article["author"])
            data[article["link"]] = article

        links = data.keys()
        for link in links:
            data[link]['takeaways'] = scraper.scrape_key_takeaways(link)
        print('---------------')
        print(data)
    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        scraper.close_driver()
