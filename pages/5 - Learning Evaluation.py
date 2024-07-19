# LEARNING EVAL PAGE
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, confusion_matrix, calinski_harabasz_score, davies_bouldin_score, classification_report, r2_score, mean_squared_error


# CLUSTERING METHODS FUNCTIONS
# Function to plot the clusters and the centroids
def visualize_clusters(data, labels, centers):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, s=50, cmap='viridis')
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)

    ax.set_title('Visualization of the clusters')
    ax.grid()
    st.pyplot(fig)

# Function to calculate the statistics of the clusters (number of points, centers, densities)
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


def clustering_evaluation(data, model):
    labels = model.labels_ if type(model) == DBSCAN else model.predict(data)
    centers = model.cluster_centers_ if type(model) == KMeans else None
    
    visualize_clusters(data, labels, centers)
    cluster_statistics = calculate_cluster_statistics(labels, centers)
    st.write(pd.DataFrame(cluster_statistics).T)

    st.markdown("---")
    st.subheader("Clustering evaluation indices: ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Silhouette score:", silhouette_score(data, labels))
    with col2:
        st.write("Calinski-Harabasz score:", calinski_harabasz_score(data, labels))
    with col3:
        st.write("Davies-Bouldin score:", davies_bouldin_score(data, labels))

    # st.write("Confusion matrix:")
    # st.write(confusion_matrix(data, labels))



# SUPERVISED LEARNING FUNCTIONS
def plot_feature_importance(data, model):
        # get the feature importances and names
        importances = model.feature_importances_
        feature_names = data.columns

        # sorted index of the feature importances
        sorted_idx = np.argsort(importances)

        # horizontal bar plot of the feature importances
        fig, ax = plt.subplots()
        ax.barh(sorted_idx, importances[sorted_idx], align='center')
        ax.set_yticks(sorted_idx)
        ax.set_yticklabels(feature_names[sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        ax.set_title("Feature Importances")
        st.pyplot(fig)


def prediction_evaluation(data, model, train_test_dict):
    y_true = train_test_dict["y_test"]
    y_pred = model.predict(train_test_dict["X_test"])
    
    # Classification Eval
    if type(model) == RandomForestClassifier:
        st.subheader("Classification report: ")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).T)

        st.markdown("---")
        st.subheader("Confusion Matrix: ")
        st.write(pd.DataFrame(confusion_matrix(y_true, y_pred)))

        st.markdown("---")
        st.subheader("Feature importances: ")
        plot_feature_importance(data, model)

    # Regression Eval
    elif type(model) == LinearRegression:
        st.subheader("Plot of the real against the predicted values: ")
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred)
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--') 
        ax.set_xlabel('Real Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Real Values vs Predicted Values')
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Regression metrics: ")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**R2 score**:", r2_score(y_true, y_pred))
        with col2:
            st.write("**RMSE**:", np.sqrt(mean_squared_error(y_true, y_pred)))
        with col3:
            st.write("**Model's Intercept**:", model.intercept_)
        
        st.write("**Model's Coefficients**: ", pd.DataFrame(model.coef_).T)
    
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
    model = st.session_state["model"] 
    last_model_name = model.__class__.__name__ 

    st.subheader("Evaluation of the last trained model")
    st.markdown(f"**Model type:** {last_model_name}")
    st.markdown("**Dataset Preview**: ")
    st.write(df.head())

    if type(model) == KMeans or type(model) == DBSCAN:
        clustering_evaluation(df, model)
    elif type(model) == RandomForestClassifier or type(model) == LinearRegression:
        st.write("**Target variable**: ", st.session_state["target"])
        st.markdown("---")
        prediction_evaluation(df, model, st.session_state["train_test_dict"])
    else:
        st.markdown("""**:warning: No model was trained yet on the "Clustering or Prediction" page!**""")

# Sidebar
st.sidebar.write("""
    Jean-Baptiste Capella - Yohan FAJERMAN - BIA1
""")
st.sidebar.write("Data Mining Project")