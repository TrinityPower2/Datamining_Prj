# DATA EXPLORATION PAGE
import pandas
import streamlit as st

st.set_page_config(
    page_title="Data Exploration",
)

st.title("Data Exploration")

try:
    df = st.session_state["file"]
except KeyError: # catching the error when no file has been registered
    # Warning when the user has not uploaded a file yet 
    st.subheader(":warning: **Please upload your dataset first in the landing page!**")
else: # case when the file has been uploaded
    # count number of rows & columns + add the na containing ones
    nb_rows = df.count(0).iloc[0] + df.isnull().sum().iloc[0]
    nb_cols = df.count(1).iloc[0] + df.isnull().sum(axis=1).iloc[0]
    # number of na values per column
    nb_na_cols = df.isnull().sum()
    # display
    st.write("Number of rows (including na) = " + str(nb_rows))
    st.write("Number of columns (including na) = " + str(nb_cols))
    st.write("Number of NA per column")
    st.write(nb_na_cols)
    # describe all columns to obtain basic stats
    st.write("Basic statistics")
    st.write(df.describe(include="all"))


# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")