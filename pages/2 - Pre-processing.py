# PREPROCESSING PAGE
import pandas
import streamlit as st


def delete_na(df_to_clean: pandas.DataFrame):
    st.write("Please select where to delete the NA values")

    dlt_choice_row_btn = st.button("Rows")
    dlt_choice_col_btn = st.button("Columns")
    dlt_choice_both_btn = st.button("Both")
    st.button("Cancel", key="cancel_delete")

    if dlt_choice_row_btn:
        df_to_clean.dropna(axis=0, inplace=True)
    elif dlt_choice_col_btn:
        df_to_clean.dropna(axis=1, inplace=True)
    elif dlt_choice_both_btn:
        df_to_clean.dropna(axis=0, inplace=True)
        df_to_clean.dropna(axis=1, inplace=True)

    return df_to_clean


def replace_na(df_to_clean: pandas.DataFrame):
    st.write("Please select the method to use (mode will always be used for non-numerical columns)")

    rpl_choice_mean_btn = st.button("Mean")
    rpl_choice_med_btn = st.button("Median")
    rpl_choice_mode_btn = st.button("Mode")
    st.button("Cancel", key="cancel_replace")

    if rpl_choice_mean_btn:
        return 1
    elif rpl_choice_med_btn:
        return 1
    elif rpl_choice_mode_btn:
        return 1

    return df_to_clean


def replace_na_algo(df_to_clean: pandas.DataFrame):
    return 1


st.set_page_config(
    page_title="Preprocessing",
)

st.title("Data Pre-processing & Cleaning")

df = st.session_state["file"]

if df is not None:
    clean_df = df
    st.header("NA values cleaning")
    na_cleaning_list = ["Delete rows and/or columns with missing values",
                        "Replace missing values with mean/median/mode",
                        "Use a sophisticated imputation algorithm"]
    select_output = st.selectbox("**Please select your NA values cleaning method and press OK**",
                                 na_cleaning_list)
    na_ok_btn = st.button("OK")

    if na_ok_btn:
        if select_output == "Delete rows and/or columns with missing values":
            st.subheader(select_output)
            clean_df = delete_na(df)
        elif select_output == "Replace missing values with mean/median/mode":
            st.subheader(select_output)
            clean_df = replace_na(df)
        elif select_output == "Use a sophisticated imputation algorithm":
            st.subheader(select_output)
            clean_df = replace_na_algo(df)
        else:
            st.write("Unknown method")

    st.header("Data normalization")

# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")
