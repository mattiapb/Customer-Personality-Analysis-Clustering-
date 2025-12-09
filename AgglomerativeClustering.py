"""
Project: Customer Personality Analysis
Module: Hierarchical Clustering (Agglomerative)
Date: Spring 2024

Description:
    This script implements Agglomerative Clustering, a hierarchical method that 
    merges data points based on distance. It serves as a validation model to 
    compare against K-Means results.
    
    Methodology:
    1. Standardization (StandardScaler)
    2. Dimensionality Reduction (PCA to 2 components)
    3. Clustering (Agglomerative with Ward linkage)
    4. Evaluation (Silhouette Coefficient)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- Visualization Settings ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_agglomerative_clustering(data_filepath):
    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(data_filepath, sep='\t')
    except FileNotFoundError:
        print("Error: File not found. Please check the filepath.")
        return

    # 1. Feature Selection
    # Selecting numerical features relevant for segmentation as defined in the report
    # We exclude ID and non-numeric fields
    features = [
        'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    
    # Cleaning: Dropping missing values to ensure scaler works correctly
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]

    # 2. Standardization
    # Normalizing features to mean=0 and std=1 (Crucial for distance-based clustering)
    print("--- Standardizing Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Dimensionality Reduction
    # Applying PCA to reduce to 2 components for 2D Visualization
    print("--- Applying PCA ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Modeling: Agglomerative Clustering
    # We use n_clusters=3 based on the optimal k found in previous K-Means analysis
    print("--- Training Agglomerative Clustering Model ---")
    agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = agg_clustering.fit_predict(X_scaled)

    # Adding results to the dataframe
    df_clean['Cluster_Agg'] = labels
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]

    # 5. Evaluation
    # Calculating Silhouette Score to measure cluster cohesion/separation
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score (Agglomerative): {score:.4f}")
    # Note: Report indicates a score around 0.5754 for this model

    # --- Visualization ---
    print("--- Plotting Results ---")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster_Agg', 
        data=df_clean, 
        palette='viridis', 
        s=70, alpha=0.8
    )
    plt.title('Customer Segments: Agglomerative Clustering (PCA Projection)', fontsize=15)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster ID')
    plt.tight_layout()
    plt.show()

    print("--- Process Completed ---")

if __name__ == "__main__":
    # Ensure this points to the dataset file
    DATA_PATH = 'marketing_campaign.csv'
    run_agglomerative_clustering(DATA_PATH)
