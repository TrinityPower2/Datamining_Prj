import streamlit as st
import pandas as pd

st.write("CAPELLA Jean-Baptiste")
st.write("FAJERMAN Yohan")
st.write("BIA1")

st.title("Data mining project: data mining dedicated app")

st.header("Introduction")
st.write("Welcome to our web app! In here, you will be able to load a tabular file containing data and "
         "perform various analysis to discover your data.")

st.header("File reading")
user_file = st.file_uploader("Please import your file here",accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
separator = st.text_input("Please enter the separator used in your file", value=",", max_chars=2)
hasHeader = st.checkbox("Please tick this box if your file contains a header (keep it unchecked if not)")
print(int(hasHeader))
print(user_file.type)
if user_file is not None:
    df = pd.read_csv(user_file, sep=separator, header=int(hasHeader))
    st.write(df.head(5))
    st.write(df.tail(5))

# csv type = application/vnd.ms-excel
# xlsx type = application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
# xls type = application/vnd.ms-excel
