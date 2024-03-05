import os
from itertools import chain

import streamlit as st
import pandas as pd
import altair as alt

from scraper import Scraper
from topic_modeler import Topic_Modeler
from topic_modeler import get_embeddings
from directories import DATA_DIR, EMBEDDINGS_DIR
from app import utils


def write_text(section_title):
    config = utils.load_config_file()
    col1, col2 = st.columns(2)
    paragraph1, paragraph2 = config[section_title]
    
    with col1:
        st.write(paragraph1)
    with col2:
        st.write(paragraph2)


def scrape_contractor_talk():
    col1, col2 = st.columns(2)
    with col1:
        # write_text("scraper_one")
        label = "What would you like to search?"
        search_terms = st.text_input(label)

    with col2:
        # write_text("scraper_two")
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
    st.write("")
    st.markdown("##### Step #1: Select Data")
    col1, _, col2 = st.columns([6,1,2])
    with col1:
        saved_results = [fn for fn in os.listdir(DATA_DIR) if fn.split(".")[-1] == "csv"]
        label = "Select the datasets you'd like to analyze."
        selections = st.multiselect(label, options=saved_results)
    
    if len(selections):
        df = pd.concat([pd.read_csv(DATA_DIR / fn) for fn in selections], ignore_index=True)
        df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
        df.drop_duplicates(inplace=True)
        with col2:
            st.metric("Posts", df.shape[0])

        search_terms = [x.replace(".csv", "").split("+") for x in selections]
        search_terms = sorted(list(set(chain.from_iterable(search_terms))))
        embeddings_filename = "+".join(search_terms) + ".pkl"

        return df, embeddings_filename
    return None, None


def display_scraped_data(df):
    """
    The concatenated dataframe returned by `dataframe_selector`
    """
    display_df = df.copy().drop(columns=["post link"])
    st.dataframe(display_df)
    

def fetch_embeddings(df, embeddings_filename):
    """Fetch embeddings if they've not already been found"""
    st.write("")
    st.markdown("##### Step #2: Fetch embeddings from OpenAI.")

    embeddings = None
    filepath = EMBEDDINGS_DIR / embeddings_filename

    if not os.path.exists(filepath):
        st.write("")
        label = f"Fetch Embeddings"
        clicked = st.button(label)

        if clicked:
            embeddings = get_embeddings(df)
            embeddings.to_pickle(filepath)

    else:
        embeddings = pd.read_pickle(filepath)
        st.success("Embeddings Retrieved and Saved!")
    
    return embeddings


def visualize_topic_clusters(embeddings, source_data):
    """Reduce with UMAP and cluster"""
    st.write("")
    st.markdown("##### Step #3: Cluster the embedding vectors.")

    with st.spinner("Reducing Embedding Vectors..."):
        model = Topic_Modeler(embeddings, source_data)
        reduction = model.reduce_dimensions()
        topic_labels = model.cluster(reduction, n_clusters=1)

    col1, col2 = st.columns(2)
    with col1:
        msg = """
        Clustering requires some visual inspection. 
        Specify how many clusters you spot in the similarity map below.
        """
        st.write(msg)
    with col2:
        label = "How Many Clusters Do You See?"
        n_clusters = st.number_input(label, value=1, min_value=1, max_value=10)

    with st.spinner("Clustering Topics..."):
        topic_labels = model.cluster(reduction, n_clusters=n_clusters)

    chart_df = pd.DataFrame(reduction, index=embeddings.index)
    chart_df.rename(columns={0:"X", 1:"Y"}, inplace=True)
    chart_df["Topic Label"] = topic_labels
    chart_df = chart_df.join(source_data.set_index("post link"))
    scatter_plot(chart_df)
    return chart_df


def scatter_plot(chart_df):
    """Altair scatter plot"""
    x_min = chart_df["X"].min() - chart_df["X"].mean()*0.1
    x_max = chart_df["X"].max() + chart_df["X"].mean()*0.1
    y_min = chart_df["Y"].min() - chart_df["Y"].mean()*0.1
    y_max = chart_df["Y"].max() + chart_df["Y"].mean()*0.1

    chart = alt.Chart(chart_df).mark_circle(
    ).encode(
        x=alt.X("X", title=None, 
                axis=alt.Axis(labels=False),
                scale=alt.Scale(domain=[x_min,x_max]),
                ),
        y=alt.Y("Y", title=None, 
                axis=alt.Axis(labels=False),
                scale=alt.Scale(domain=[y_min,y_max]),
                ),
        # color=alt.Color("Topic Label:N", legend=alt.Legend(orient="bottom")),
        color=alt.Color("Topic Label:N"),
        href="url",
        tooltip=["title", "text"],
    ).properties(title="Similarity Map from OpenAI Embeddings")
    st.write("")
    st.altair_chart(chart, use_container_width=True)


def expore_topics(center_column, chart_df):
    with center_column:
        # label = "Choose a Topic to Explore"
        col1, _ = st.columns([2,2])
        with col1:
            label = "Read through the posts that have been clustered together."
            options = chart_df["Topic Label"].value_counts().index
            selection = st.selectbox(label, sorted(options))

    display_df = chart_df[chart_df["Topic Label"] == selection].copy()
    display_df.reset_index(inplace=True)
    display_df.drop(columns=["post link", "X", "Y", "Topic Label"], inplace=True)
    display_df.set_index("title", inplace=True)
    st.dataframe(display_df)