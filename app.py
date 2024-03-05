import streamlit as st

from app import logic as lg


st.set_page_config("Hardy Boys", page_icon="🔍", layout="wide")

_, cntr, _ = st.columns([2,6,2])
with cntr:
    st.title("Contractor Talk")

    st.header("Scrape")
    lg.write_text("scraper")
    lg.scrape_contractor_talk()


    st.write("")
    st.write("")
    st.markdown("---")
    st.header("Analyze")
    lg.write_text("analyze")
    
    open_ai_clusters = False
    df, embeddings_filename = lg.dataframe_selector()
    if df is not None:
        lg.display_scraped_data(df)
        embeddings = lg.fetch_embeddings(df, embeddings_filename)

        if embeddings is not None:
            chart_df = lg.visualize_topic_clusters(embeddings, df)
            open_ai_clusters = True

if open_ai_clusters:
    lg.expore_topics(chart_df)
