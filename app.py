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
lg.dataframe_selector()