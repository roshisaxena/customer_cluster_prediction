# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model
kmeans = joblib.load("cluster_model.pkl")

# Load dataset for plotting
df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_array = X.values

# Streamlit UI
st.set_page_config(page_title="Mall Customer Clustering", layout="centered")
st.title("🛍️ Mall Customer Cluster Predictor")
st.write("Enter the customer's **Annual Income** and **Spending Score** to predict the cluster.")

# Inputs
income = st.number_input("Annual Income (in k$)", min_value=0, max_value=300, value=50)
score = st.slider("Spending Score (1–100)", 1, 100, 50)

# Predict button
if st.button("Predict Cluster"):
    input_data = np.array([[income, score]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f"🎯 Predicted Cluster: {cluster}")

    # Plotting the clusters
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    colors = ["green", "red", "blue", "yellow", "pink"]

    for i in range(5):
        ax.scatter(X_array[kmeans.labels_ == i, 0], X_array[kmeans.labels_ == i, 1],
                   s=100, c=colors[i], label=f"Cluster {i}")
    
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               s=300, c="cyan", label="Centroids", edgecolors='black')
    
    ax.scatter(income, score, s=300, c="black", label="Your Input", marker="X")

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segments")
    ax.legend()
    st.pyplot(fig)
