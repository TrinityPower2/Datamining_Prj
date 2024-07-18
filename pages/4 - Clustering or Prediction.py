# CLUSTERING / PREDICTION PAGE
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Clustering / Prediction",
)


def set_status(i):
    st.session_state["status"] = i


def kmeans(data, is_pca=0, k_value=8, rd_state=None):
    model = KMeans(k_value, random_state=rd_state)
    if is_pca:
        st.session_state["is_pca"] = 1
        print("I perform PCA before training")
        pca_data = PCA(n_components=2, random_state=42).fit_transform(data)
        model.fit(pca_data)
        st.session_state["pca_data"] = pca_data
    else:
        st.session_state["is_pca"] = 0
        model.fit(data)

    st.session_state["model"] = model
    set_status(1)
    st.rerun()


def dbscan(data, epsilon=0.5, min_samples=5):
    model = DBSCAN(eps=epsilon, min_samples=min_samples)
    model.fit(data)

    st.session_state["model"] = model
    set_status(1)
    st.rerun()


def linear_reg(data, target, test_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(data[[col for col in df.columns if col != target]],
                                                        data[target], test_size=test_size, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_test_dict = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    st.session_state["model"] = model
    st.session_state["train_test_dict"] = train_test_dict
    set_status(1)
    st.rerun()


def random_forest_classifier(data, target, test_size=0.25, n_esti=100, crit="gini", max_dpt=None, rd_state=None):
    X_train, X_test, y_train, y_test = train_test_split(data[[col for col in df.columns if col != target]],
                                                        data[target], test_size=test_size, random_state=42)

    model = RandomForestClassifier(n_estimators=n_esti, criterion=crit, max_depth=max_dpt, random_state=rd_state)
    model.fit(X_train, y_train)

    train_test_dict = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    st.session_state["model"] = model
    st.session_state["train_test_dict"] = train_test_dict
    set_status(1)
    st.rerun()


st.title("Clustering or Prediction")

try:
    df = st.session_state["file"]
except KeyError: # catching the error when no file has been registered
    # Warning when the user has not uploaded a file yet 
    st.subheader(":warning: **Please upload your dataset first in the landing page!**")
else:  # case when the file has been uploaded
    if 'status' not in st.session_state:
        set_status(1)

    st.header("Clustering models")

    clustering_list = ["KMeans", "DBSCAN"]
    select_output = st.selectbox("**Please select your clustering model and press OK**",
                                 clustering_list)

    if st.button("OK", key="OK_clust") or st.session_state["status"] == 2:
        if select_output == "KMeans":
            st.subheader(select_output)

            pca = st.toggle("Do you want to perform a PCA before running the KMeans?")
            k = st.number_input("Enter the number of clusters", min_value=2)
            random_state = st.number_input("Enter the random state", min_value=0, key="kmeans_rd_state")

            if st.button("Start", key="St1", on_click=set_status(2)):
                kmeans(df, is_pca=pca, k_value=k, rd_state=random_state)

        elif select_output == "DBSCAN":
            st.subheader(select_output)

            eps = st.number_input("Enter the value of parameter epsilon", min_value=0.01)
            min_samp = st.number_input("Enter the value of parameter min_samples", min_value=1)

            if st.button("Start", key="St2", on_click=set_status(2)):
                dbscan(df, epsilon=eps, min_samples=min_samp)

        else:
            st.write("Unknown model")

    st.header("Prediction models")

    prediction_list = ["Linear regression", "Random forest classifier"]
    select_output = st.selectbox("**Please select your prediction model and press OK**",
                                 prediction_list)

    if st.button("OK", key="OK_pred") or st.session_state["status"] == 2:
        if select_output == "Linear regression":
            st.subheader(select_output)

            trgt = st.selectbox("**Please select the column to use as the target**", df.columns)
            test = st.number_input("Enter the test size", min_value=0.1, key="lin_test")

            if st.button("Start", key="St3", on_click=set_status(2)):
                linear_reg(df, target=trgt, test_size=test)

        elif select_output == "Random forest classifier":
            st.subheader(select_output)

            trgt = st.selectbox("Please select the column to use as the target", df.columns)
            test = st.number_input("Enter the test size", min_value=0.1, key="forest_test")
            nb_estimators = st.number_input("Enter the number of trees in the forest", min_value=1)
            criterion_list = ["gini", "entropy", "log_loss"]
            criterion = st.selectbox("Select the criterion to use",criterion_list)
            max_depth = st.number_input("Enter the value for parameter max_depth", min_value=1)
            random_state = st.number_input("Enter the random state", min_value=0, key="forest_rd_state")

            if st.button("Start", key="St4", on_click=set_status(2)):
                random_forest_classifier(df, target=trgt, test_size=test, n_esti=nb_estimators, crit=criterion,
                                         max_dpt=max_depth, rd_state=random_state)

        else:
            st.write("Unknown model")

# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")
