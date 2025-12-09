"""
Project: Customer Personality Analysis
Module: Experimental Clustering - Spectral Clustering
Authors: Mattia Pinilla, Jorge Córdoba, Ignacio López, Carlos Flores, Francisco Santibáñez
Date: Spring 2024

Description:
    This script implements Spectral Clustering. This algorithm applies clustering 
    to a projection of the normalized Laplacian. It is particularly useful when 
    the structure of the individual clusters is highly non-convex.

    NOTE: As detailed in the final report, this model was part of the exploratory 
    phase but was excluded from the final solution due to performance issues 
    and inconsistent cluster separation compared to K-Means.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Visualization Settings ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_spectral_clustering(data_filepath):
    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(data_filepath, sep='\t')
    except FileNotFoundError:
        print("Error: File not found. Please check the filepath.")
        return

    # 1. Feature Selection
    # Consistent feature set used across the project
    features = [
        'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    
    # Handling missing values
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]

    # 2. Standardization
    # Essential for Spectral Clustering as it builds an affinity matrix based on distance/similarity
    print("--- Standardizing Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Modeling: Spectral Clustering
    # We use 'nearest_neighbors' for affinity to construct the connectivity graph
    # assign_labels='kmeans' is the standard strategy for the final step
    print("--- Training Spectral Clustering Model ---")
    
    # Note: Spectral Clustering can be computationally expensive on large datasets
    spectral = SpectralClustering(
        n_clusters=3, 
        affinity='nearest_neighbors', 
        assign_labels='kmeans', 
        random_state=42,
        n_jobs=-1
    )
    labels = spectral.fit_predict(X_scaled)

    # Adding labels to dataframe
    df_clean['Cluster_Spectral'] = labels

    # 4. Dimensionality Reduction for Visualization
    print("--- Applying PCA for Visualization ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]

    # 5. Visualization
    print("--- Plotting Results ---")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster_Spectral', 
        data=df_clean, 
        palette='plasma', 
        s=60, alpha=0.7,
        legend='full'
    )
    plt.title('Spectral Clustering (PCA Projection)', fontsize=15)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster ID')
    plt.tight_layout()
    plt.show()

    print("--- Analysis Complete ---")
    print("Note: This model is included for comparative purposes only.")

if __name__ == "__main__":
    DATA_PATH = 'marketing_campaign.csv'
    run_spectral_clustering(DATA_PATH)
