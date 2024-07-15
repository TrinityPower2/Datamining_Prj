# CLUSTERING / PREDICTION PAGE
import streamlit as st

st.set_page_config(
    page_title="Clustering / Prediction",
)

st.title("Clustering or Prediction")

df = st.session_state["file"]

if df is not None:
    st.write(st.session_state)

st.write(df.head())

# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")