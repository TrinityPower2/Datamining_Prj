# first page test
import streamlit as st

st.set_page_config(
    page_title="First page"
)

st.title("First page")

st.write(st.session_state)

df = st.session_state["file"]
st.write(df.head())