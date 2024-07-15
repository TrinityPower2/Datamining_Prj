# DATA EXPLORATION PAGE
import streamlit as st

st.set_page_config(
    page_title="Data Exploration",
)

st.title("Data Exploration")
df = st.session_state["file"]

if df is not None:
    nb_rows = df.count(0)[0]
    nb_cols = df.count(1)[0]
    nb_na_cols = df.isnull().sum()
    st.write("Number of rows = " + str(nb_rows))
    st.write("Number of columns = " + str(nb_cols))
    st.write("Number of NA per column")
    st.write(nb_na_cols)
    st.write("Basic statistics")
    st.write(df.describe(include="all"))


# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")