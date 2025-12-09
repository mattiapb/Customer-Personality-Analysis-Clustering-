"""
Project: Customer Personality Analysis
Module: Experimental Clustering - DBSCAN
Date: Spring 2024

Description:
    This script implements Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    
    NOTE: As detailed in the final report, this model was part of the exploratory 
    phase. It was ultimately excluded from the final analysis because it failed 
    to produce distinct, actionable customer segments compared to K-Means.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Visualization Settings ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_dbscan(data_filepath):
    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(data_filepath, sep='\t')
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 1. Feature Selection
    # Selecting relevant numerical features for segmentation
    features = [
        'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    
    # Removing rows with missing values in selected features
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]

    # 2. Standardization
    # Critical for DBSCAN as it relies on epsilon distance neighborhoods
    print("--- Standardizing Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Modeling: DBSCAN
    # Parameters:
    #   eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    #   min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    print("--- Running DBSCAN ---")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    # Adding labels to dataframe
    df_clean['Cluster_DBSCAN'] = labels

    # 4. Analysis of Results
    # DBSCAN assigns -1 to noise points (outliers)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")
    print(f"Noise percentage: {100 * n_noise / len(labels):.2f}%")

    # 5. Visualization via PCA
    print("--- Generating Visualization ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster_DBSCAN', 
        data=df_clean, 
        palette='bright', 
        legend='full',
        s=60, alpha=0.7
    )
    plt.title(f'DBSCAN Clustering (PCA Projection)\nClusters: {n_clusters} | Noise Points: {n_noise}', fontsize=15)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster ID (-1 is Noise)')
    plt.tight_layout()
    plt.show()

    print("--- Analysis Complete ---")
    print("Note: If the plot shows mostly noise (-1) or a single cluster, parameters 'eps' and 'min_samples' may need adjustment.")

if __name__ == "__main__":
    DATA_PATH = 'marketing_campaign.csv'
    run_dbscan(DATA_PATH)
