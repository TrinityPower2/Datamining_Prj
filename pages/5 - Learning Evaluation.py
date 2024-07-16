# LEARNING EVAL PAGE
import streamlit as st

st.set_page_config(
    page_title="Learning Evaluation",
)

st.title("Learning Evaluation")

try:
    df = st.session_state["file"]
except KeyError: # catching the error when no file has been registered
    # Warning when the user has not uploaded a file yet 
    st.subheader(":warning: **Please upload your dataset first in the landing page!**")
else: # case when the file has been uploaded
    st.write(st.session_state)
    st.write(df.head())


# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")