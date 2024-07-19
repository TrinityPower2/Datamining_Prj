# DATAVIZ PAGE
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns

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

    # We distinguish the numerical and categorical variables for further plots
    num_vars = df.select_dtypes(include=np.number).columns
    cat_vars = df.select_dtypes(include=['object']).columns

    st.header("Univariate Analysis")
    # HISTOGRAMS
    st.subheader("Histograms")
    select_histo = st.selectbox("**Please select the variable whose histogram you want to display and press the button**",
                                df.columns)
    histo_ok_btn = st.button("Plot histogram")

    if histo_ok_btn: 
        fig = px.histogram(df, x=select_histo)
        st.plotly_chart(fig)

    # BOXPLOTS
    st.markdown("---")
    st.subheader("Boxplots")
    select_box = st.selectbox("**Please select the variable whose boxplot you want to display and press the button**",
                                df.columns)
    box_ok_btn = st.button("Plot boxplot")

    if box_ok_btn:
        fig = px.box(df, y=select_box)
        st.plotly_chart(fig)

    
    # PIE CHARTS
    st.markdown("---")
    st.subheader("Pie Charts")
    select_pie = st.selectbox("**Please select the variable whose pie chart you want to display and press the button**",
                                df.columns)
    pie_ok_btn = st.button("Plot pie chart")

    if pie_ok_btn:
        # condition to display 10 first values if the feature has too many distinct values"
        if df[select_pie].nunique() > 10:
            top_10_values = df[select_pie].value_counts().head(10).index.tolist()
            fig = px.pie(df.loc[df[select_pie].isin(top_10_values)], names=select_pie, values=select_pie)
            st.write("We only display the 10 most represented values of the dataset:")
        else:
            fig = px.pie(df, names=df[select_pie].value_counts().index, values=df[select_pie].value_counts().values)
        st.plotly_chart(fig)



    st.markdown("---")
    st.header("Bivariate Analysis")

    # BAR PLOTS
    st.subheader("Bar Plots")

    col1, col2 = st.columns(2)
    with col1:
        select_bar_x = st.selectbox("**Please select the bar's variable for the x-axis**",
                                    df.columns)
    with col2:
        select_bar_y = st.selectbox("**Please select the bar's variable for the y-axis**",
                                    df.columns)
    bar_ok_btn = st.button("Plot bar plot")

    if bar_ok_btn:
        # condition on the selected variables :  
        if select_bar_x == select_bar_y: # no plot if the selected variables are the same
            st.write(":warning: **Please select two different variables!**")
        #elif (select_bar_x in num_vars and select_bar_y in num_vars) or (select_bar_x in cat_vars and select_bar_y in cat_vars):
            # or if we don't have numeric vs categorical
            #st.write(":warning: **Please select one numerical and one categorical variable for your bar plot!**")
        else:
            sns.barplot(x=select_bar_x, y=select_bar_y, data=df, hue=select_bar_x, legend=False)
            st.pyplot()


    # SCATTER PLOTS
    st.markdown("---")
    st.subheader("Scatter Plots")

    col3, col4 = st.columns(2)
    with col3:
        select_scatter_x = st.selectbox("**Please select the scatter's variable for the x-axis**",
                                    df.columns)
    with col4:
        select_scatter_y = st.selectbox("**Please select the scatter's variable for the y-axis**",
                                    df.columns)
    scatter_ok_btn = st.button("Plot scatter plot")

    if scatter_ok_btn:
        # condition on the selected variables :  
        if select_scatter_x == select_scatter_y:
            st.write(":warning: **Please select two different variables!**")
        elif select_scatter_x in cat_vars and select_scatter_y in cat_vars:
            st.write(":warning: **Please select two numerical variables for your scatter plot!**")
        else:
            fig = px.scatter(df, x=select_scatter_x, y=select_scatter_y)
            st.plotly_chart(fig)


    # HEATMAPS
    st.markdown("---")
    st.subheader("Heatmaps")

    heatmap_ok_btn = st.button("Plot heatmap")

    if heatmap_ok_btn:
        sns.heatmap(df.corr(numeric_only=True).round(decimals=2), annot=True)
        st.pyplot()



# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")