# PREPROCESSING PAGE
import pandas
import streamlit as st
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from scipy.stats import zscore


def delete_na(df_to_clean: pandas.DataFrame, method):
    # perform the deletion according to sent method
    if method == "Rows":
        df_to_clean.dropna(axis=0, inplace=True, how='any')
    elif method == "Columns":
        df_to_clean.dropna(axis=1, inplace=True)
    elif method == "Both":
        df_to_clean.dropna(axis=0, inplace=True)
        df_to_clean.dropna(axis=1, inplace=True)

    # replace the df by the cleaned version in session state storage
    st.session_state["file"] = df_to_clean
    # button management
    set_stage(1)
    st.rerun()


def replace_na(df_to_clean: pandas.DataFrame, method):
    # handling numeric columns according to sent method
    if method == "Mean":
        df_to_clean.fillna(df_to_clean.mean(numeric_only=True, skipna=True), inplace=True)
    elif method == "Median":
        df_to_clean.fillna(df_to_clean.median(numeric_only=True, skipna=True), inplace=True)
    elif method == "Mode":
        df_to_clean.fillna(df_to_clean.mode(numeric_only=True, dropna=True), inplace=True)

    # handling the remaining non-numeric columns by filling in with the mode
    df_to_clean.fillna(df_to_clean.mode(), inplace=True)

    # replace df by cleaned version in session state storage
    st.session_state["file"] = df_to_clean
    # button management
    set_stage(1)
    st.rerun()


def replace_na_algo(df_to_clean: pandas.DataFrame, method):
    # replace na according to sent method
    if method == "KNN":
        knn_imputer = KNNImputer(n_neighbors=1)
        for i in df_to_clean:
            if df[i].dtype != object:
                df[i] = knn_imputer.fit_transform(pandas.DataFrame(df[i]))
    elif method == "Simple imputation":
        df_to_clean.interpolate(inplace=True)

    # replace df by cleaned version in session state storage
    st.session_state["file"] = df_to_clean
    # button management
    set_stage(1)
    st.rerun()


def min_max_norm(df_to_norm):
    # min max normalize the data
    normalizer = MinMaxScaler(copy=False)
    for i in df_to_norm:
        if df[i].dtype != object:
            df[i] = normalizer.fit_transform(pandas.DataFrame(df[i]))

    # replace df by cleaned version in session state storage
    st.session_state["file"] = df_to_norm
    st.rerun()


def z_score_norm(df_to_norm):
    # z_score normalize the data
    for i in df_to_norm:
        if df[i].dtype != object:
            df[i] = zscore(df[i])

    # replace df by cleaned version in session state storage
    st.session_state["file"] = df_to_norm
    st.rerun()


def max_abs_norm(df_to_norm):
    # max_abs normalize the data
    normalizer = MaxAbsScaler(copy=False)
    for i in df_to_norm:
        if df[i].dtype != object:
            df[i] = normalizer.fit_transform(pandas.DataFrame(df[i]))

    # replace df by cleaned version in session state storage
    st.session_state["file"] = df_to_norm
    st.rerun()


# multiple levels of buttons management
def set_stage(i):
    st.session_state["stage"] = i


st.set_page_config(
    page_title="Preprocessing",
)

st.title("Data Pre-processing & Cleaning")

try:
    df = st.session_state["file"]
except KeyError: # catching the error when no file has been registered
    # Warning when the user has not uploaded a file yet 
    st.subheader(":warning: **Please upload your dataset first in the landing page!**")
else:  # case when the file has been uploaded

    # display currently stored data
    st.subheader("Current state of df")
    st.write(df)

    # if not, create & initialize the stage variable (used for button management) in sessions state storage
    if 'stage' not in st.session_state:
        set_stage(1)

    # data cleaning choice & inputs
    st.header("NA values cleaning")

    # data cleaning choice
    na_cleaning_list = ["Delete rows and/or columns with missing values",
                        "Replace missing values with mean/median/mode",
                        "Use a sophisticated imputation algorithm"]
    select_output = st.selectbox("**Please select your NA values cleaning method and press OK**",
                                 na_cleaning_list)
    if st.button("OK") or st.session_state["stage"] == 2:
        # delete na inputs
        if select_output == "Delete rows and/or columns with missing values":
            st.subheader(select_output)

            possibility_list = ["Rows", "Columns", "Both"]
            select_output = st.selectbox("**Please select where to delete the NA values and press Start**",
                                         possibility_list)

            # cleaning start
            if st.button("Start", key="St1", on_click=set_stage(2)):
                delete_na(df, select_output)

        # replace na inputs
        elif select_output == "Replace missing values with mean/median/mode":
            st.subheader(select_output)

            possibility_list = ["Mean", "Median", "Mode"]
            select_output = st.selectbox("**Please select the method to use (mode will always be used for non-numerical"
                                         " columns) and press start**",
                                         possibility_list)

            # cleaning start
            if st.button("Start", key="St2", on_click=set_stage(2)):
                replace_na(df, select_output)

        # replace algo na inputs
        elif select_output == "Use a sophisticated imputation algorithm":
            st.subheader(select_output)

            possibility_list = ["KNN", "Simple imputation"]
            select_output = st.selectbox("**Please select the method to use and press start (note that only NaN from "
                                         "numeric columns will be treated)**",
                                         possibility_list)

            # cleaning start
            if st.button("Start", key="St3", on_click=set_stage(2)):
                replace_na_algo(df, select_output)
        else:
            st.write("Unknown method")

    # normalization method choice
    st.header("Data normalization")

    normalization_list = ["Min-max normalization",
                          "Z-score standardization",
                          "MaxAbs normalization"]
    select_output = st.selectbox("**Please select your normalization method and press OK**",
                                 normalization_list)

    # normalization start
    if st.button("OK", key="OK2"):
        if select_output == "Min-max normalization":
            min_max_norm(df)

        elif select_output == "Z-score standardization":
            z_score_norm(df)

        elif select_output == "MaxAbs normalization":
            max_abs_norm(df)
        else:
            st.write("Unknown method")


# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")
