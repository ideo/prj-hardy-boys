import os

import streamlit as st

from scraper import Scraper
from directories import DATA_DIR


def scrape_contractor_talk():
    label = "What would you like to search"
    search_terms = st.text_input(label)
    if search_terms != "":
        _, cntr, _ = st.columns([2.5,2,2.5])
        with cntr:
            label = f"Scrape Search Results"
            clicked = st.button(label)

        if clicked:
            search_terms = [x.strip().lower() for x in search_terms.split(" ")]
            
            scraper = Scraper()
            filepath = scraper.scrape_search_results(search_terms)
            scraper.scrape_posts_from_result_links(filepath)


def dataframe_selector():
    saved_results = [fn for fn in os.listdir(DATA_DIR) if fn.split(".")[-1] == "csv"]
    selections = st.multiselect("Choose Scraping Results", options=saved_results)
    return selections