import pandas
import streamlit as st
import pandas as pd


def landing_page_display(file, sep=",", head=0):
    if file is not None:
        df = pd.read_csv(file, sep=sep, header=head)
        st.session_state['file'] = df
        st.header("Preview")
        st.subheader("5 first rows")
        st.write(df.head(5))
        st.subheader("5 last rows")
        st.write(df.tail(5))

        st.header("Basic statistics")
        nb_rows = df.count(0)[0]
        nb_cols = df.count(1)[0]
        nb_na_cols = df.isnull().sum()
        st.write("Number of rows = " + str(nb_rows))
        st.write("Number of columns = " + str(nb_cols))
        st.write("Number of NA per column:")
        st.write(nb_na_cols)
        st.write(df.describe(include="all"))

st.set_page_config(
    page_title="landing page"
)

st.write("CAPELLA Jean-Baptiste")
st.write("FAJERMAN Yohan")
st.write("BIA1")

st.title("Data mining project: data mining dedicated app")

st.header("Introduction")
st.write("Welcome to our web app! In here, you will be able to load a tabular file containing data and "
         "perform various analysis to discover your data.")

st.header("File reading")
user_file = st.file_uploader("Please import your file here",accept_multiple_files=False, type=['csv', 'xls', 'data',
                                                                                               'txt'])
separator = st.text_input("Please enter the separator used in your file", value=",", max_chars=2)
hasHeader = st.checkbox("Please tick this box if your file contains a header (keep it unchecked if not)", value=True)

if hasHeader:
    header = 0
else:
    header = None

ok_btn = st.button("OK")

if ok_btn:
    landing_page_display(user_file, separator, head=header)
