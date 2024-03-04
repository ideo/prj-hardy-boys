import os

import streamlit as st
import pandas as pd

from scraper import Scraper
from directories import DATA_DIR
from app import utils


def write_text(section_title):
    config = utils.load_config_file()
    for paragraph in config[section_title]:
        st.write(paragraph)


def scrape_contractor_talk():
    col1, col2 = st.columns(2)
    with col1:
        write_text("scraper_one")
        label = "What would you like to search?"
        search_terms = st.text_input(label)

    with col2:
        write_text("scraper_two")
        if search_terms != "":
            search_terms = [x.strip().lower() for x in search_terms.split(" ")]
            st.write(f"Search Terms:\n\r`{search_terms}`")

    if search_terms != "":
        _, cntr, _ = st.columns([2.5,2,2.5])
        with cntr:
            st.write("")
            label = f"Scrape Search Results"
            clicked = st.button(label)

        if clicked:       
            scraper = Scraper()
            filepath = scraper.scrape_search_results(search_terms)
            scraper.scrape_posts_from_result_links(filepath)


def dataframe_selector():
    col1, _, col2 = st.columns([6,1,2])
    with col1:
        saved_results = [fn for fn in os.listdir(DATA_DIR) if fn.split(".")[-1] == "csv"]
        selections = st.multiselect("Choose Scraping Results", options=saved_results)
    
    if len(selections):
        df = pd.concat([pd.read_csv(DATA_DIR / fn) for fn in selections])
        df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
        with col2:
            st.metric("Posts", df.shape[0])
        return df