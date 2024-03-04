from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

from directories import DATA_DIR



class Scraper:
    def __init__(self):
        self.base_url = "https://www.contractortalk.com/"


    def scrape_search_results(self, search_terms):
        """Search on Contractor Talk and Scrape all result page links"""
        search_terms = "+".join(search_terms)
        # results = pd.DataFrame()

        # next_page_exists = True
        # page_count = 1
        search_url = self._format_search_url(search_terms)
        page_links, total_page_count = self._scrape_search_result_page_links(search_url)
        results = pd.DataFrame(columns=[search_terms], data=page_links)
        # results = pd.concat([results, df])

        # while next_page_exists:
        for page_count in tqdm(range(2,total_page_count+1)):
            search_url = self._format_search_url(search_terms, page=page_count)
            page_links, _ = self._scrape_search_result_page_links(search_url)
            df = pd.DataFrame(columns=[search_terms], data=page_links)
            results = pd.concat([results, df])
            
            # print(page_count)
            page_count += 1

        return results


    def _format_search_url(self, search_terms, page=1):
        search_url = urljoin(self.base_url, "search/465553/")

        if page > 1:
            params = f"?page={page}&q={search_terms}&c[searchProfileName]=control&o=relevance"
            search_url = urljoin(search_url, params)
        else:
            params = f"?q={search_terms}&c[searchProfileName]=control&o=relevance"
            search_url = urljoin(search_url, params)

        return search_url
    

    def _scrape_search_result_page_links(self, search_url):
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, features="html.parser")
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
    search_terms = ["hardie", "installation"]
    scraper = Scraper()
    results = scraper.scrape_search_results(search_terms)
    filename = "+".join(search_terms) + ".csv"
    results.to_csv(DATA_DIR / filename)