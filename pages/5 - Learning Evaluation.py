# LEARNING EVAL PAGE
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix


def visualize_clusters(data, labels, centers):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    # ax.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)

    ax.set_title('Visualization of the clusters')
    ax.grid()
    st.pyplot(fig)

def calculate_cluster_statistics(labels, model):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_statistics = {}

    for label, count in zip(unique_labels, counts):        
        cluster_statistics[str(label)] = {
            'count': count,
            'center': model.cluster_centers_[label].round(2) if type(model) == KMeans else None,
            'density': model.core_sample_indices_[label] if type(model) == DBSCAN else None
        }
    return cluster_statistics
    
    
st.set_page_config(
    page_title="Learning Evaluation",
)

st.title("Learning Evaluation")

try:
    df = st.session_state["file"]
except KeyError: # catching the error when no file has been registered
    # Warning when the user has not uploaded a file yet 
    st.subheader(":warning: **Please upload your dataset first in the landing page!**")
else: # case when the file has been uploaded

    iris = datasets.load_iris()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(iris.data)
    visualize_clusters(iris.data, kmeans.labels_, kmeans.cluster_centers_)
    cluster_statistics = calculate_cluster_statistics(kmeans.labels_, kmeans)

    # DBSCAN_model = DBSCAN(eps=0.5, min_samples=5)
    # DBSCAN_model.fit(iris.data)
    # visualize_clusters(iris.data, DBSCAN_model.labels_, None)
    # cluster_statistics = calculate_cluster_statistics(DBSCAN_model.labels_, DBSCAN_model)

    st.write(pd.DataFrame(cluster_statistics).T)

    st.write("Silhouette score:", silhouette_score(iris.data, kmeans.labels_))

    st.write("Confusion matrix:")
    st.write(confusion_matrix(iris.target, kmeans.labels_))

# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")