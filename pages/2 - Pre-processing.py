# PREPROCESSING PAGE
import streamlit as st

st.set_page_config(
    page_title="Preprocessing",
)

st.title("Data Pre-processing & Cleaning")

df = st.session_state["file"]

if df is not None:
    st.write(st.session_state)

st.write(df.head())