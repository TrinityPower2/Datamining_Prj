# DATAVIZ PAGE
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Data Visualisation",
)

st.title("Visualization of the cleaned data")

try:
    df = st.session_state["file"]
except KeyError: # catching the error when no file has been registered
    # Warning when the user has not uploaded a file yet 
    st.subheader(":warning: **Please upload your dataset first in the landing page!**")
else: # case when the file has been uploaded
    st.header("Histograms")
    select_histo = st.selectbox("**Please select the variable whose histogram you want to display and press the button**",
                                 df.columns)
    histo_ok_btn = st.button("Plot histogram")

    if histo_ok_btn:
        fig = px.histogram(df, x=select_histo)
        st.plotly_chart(fig)

    st.markdown("---")

    st.header("Boxplots")
    select_box = st.selectbox("**Please select the variable whose boxplot you want to display and press the button**",
                                 df.columns)
    box_ok_btn = st.button("Plot boxplot")

    if box_ok_btn:
        fig = px.box(df, x=select_box)
        st.plotly_chart(fig)


# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")