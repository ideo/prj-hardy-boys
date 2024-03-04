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
df = lg.dataframe_selector()
if df is not None:
    st.dataframe(df)