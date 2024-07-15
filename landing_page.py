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


st.set_page_config(
    page_title="Landing page"
)

st.title("Data Mining dedicated app")

st.header("Introduction")
st.write("Welcome to our web app! In here, you will be able to load a tabular file containing data and "
         "perform various analysis to discover your data.")

st.header("File reading")
user_file = st.file_uploader("Please import your file here",accept_multiple_files=False, type=['csv', 'xls', 'data',
                                                                                               'txt'])
separator = st.text_input("Please enter the separator used in your file", value=",", max_chars=2)
hasHeader = st.checkbox("Please tick this box if your file contains a header (uncheck if not)", value=True)

if hasHeader:
    header = 0
else:
    header = None

ok_btn = st.button("OK")

if ok_btn:
    landing_page_display(user_file, separator, head=header)


# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")
