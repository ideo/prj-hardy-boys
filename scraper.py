import random
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd
# from tqdm import tqdm
from stqdm import stqdm as tqdm


from directories import DATA_DIR


class Scraper:
    def __init__(self):
        self.base_url = "https://www.contractortalk.com/"


    def _get_page_source(self, url):
        response = requests.get(url, headers={'Cache-Control': 'no-cache'})
        soup = BeautifulSoup(response.content, features="html.parser")
        return soup        


    def scrape_search_results(self, search_terms):
        """Search on Contractor Talk and Scrape all result page links"""
        search_terms = "+".join(search_terms)
        search_id = random.randint(111111, 999999)

        search_url = self._format_search_url(search_terms, search_id)
        page_links, total_page_count = self._scrape_search_result_page_links(search_url)
        results = pd.DataFrame(columns=["post link"], data=page_links)

        _range = range(2, total_page_count+1)
        desc = f"Scraping {total_page_count} pages of results"
        for page_count in tqdm(_range, desc=desc):
            search_url = self._format_search_url(search_terms, search_id, page=page_count)
            page_links, _ = self._scrape_search_result_page_links(search_url)
            df = pd.DataFrame(columns=["post link"], data=page_links)
            results = pd.concat([results, df])

        filename = f"{search_terms}.csv"
        filepath = DATA_DIR / filename
        results.to_csv(filepath)
        return filepath
    

    def scrape_posts_from_result_links(self, filepath):
        df = pd.read_csv(filepath)
        tqdm.pandas(desc=f"Scraping {df.shape[0]} posts")
        df[["title", "text"]] = df["post link"].progress_apply(lambda x: self._get_post_contents(x))
        df.to_csv(filepath)


    def _get_post_contents(self, post_link):
        post_url = urljoin(self.base_url, post_link)
        soup = self._get_page_source(post_url)

        itemid = post_url.rsplit("/", 1)[0] + "/"
        post_div = soup.find("div", attrs={"itemid": itemid})
        
        title = post_div.find("h1", attrs={"class": "MessageCard__thread-title"}).text.strip()
        text = post_div.find("article", attrs={"qid": "post-text"}).text.strip()
        
        # return title, text
        return pd.Series([title, text])


    def _format_search_url(self, search_terms, search_id, page=1):
        search_url = urljoin(self.base_url, f"search/{search_id}/")

        if page > 1:
            params = f"?page={page}&q={search_terms}&o=relevance"
            search_url = urljoin(search_url, params)
        else:
            params = f"?q={search_terms}&o=relevance"
            search_url = urljoin(search_url, params)

        print(search_url)
        return search_url
    

    def _scrape_search_result_page_links(self, search_url):
        soup = self._get_page_source(search_url)
        posts = soup.find_all("a", attrs={"qid": "search-results-title"})
        page_links = [post["href"] for post in posts]

        # Should we continue
        # next_page_btn = soup.find("a", attrs={"qid": "page-nav-next-button"})
        # next_page_exists = next_page_btn["aria-disabled"] == "false"

        # How mnay pages are there?
        last_page_btn = soup.find_all("a", attrs={"qid": "page-nav-other-page"})[-1]
        total_page_count = int(last_page_btn.text.strip())

        return page_links, total_page_count




if __name__ == "__main__":
    scraper = Scraper()

    # search_terms = ["hardie", "siding"]
    # results = scraper.scrape_search_results(search_terms)
    # filename = "+".join(search_terms) + ".csv"
    # results.to_csv(DATA_DIR / filename)

    filepath = "data/hardie+siding.csv"
    scraper.scrape_posts_from_result_links(filepath)