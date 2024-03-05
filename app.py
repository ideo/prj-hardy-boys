import streamlit as st

from app import logic as lg


st.set_page_config("Hardy Boys", page_icon="🔍")
st.title("Contractor Talk")

st.header("Scrape")
lg.scrape_contractor_talk()


st.write("")
st.write("")
st.markdown("---")
st.header("Analyze")
df, embeddings_filename = lg.dataframe_selector()
if df is not None:
    lg.display_scraped_data(df)
    embeddings = lg.fetch_embeddings(df, embeddings_filename)

    if embeddings is not None:
        lg.visualize_topic_clusters(embeddings, df)
