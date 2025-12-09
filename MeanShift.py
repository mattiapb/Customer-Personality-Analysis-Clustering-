"""
Project: Customer Personality Analysis
Module: Experimental Clustering - MeanShift
Date: Spring 2024

Description:
    This script implements the MeanShift clustering algorithm. 
    MeanShift is a centroid-based algorithm which works by updating candidates 
    for centroids to be the mean of the points within a given region.

    NOTE: As detailed in the final report, this model was explored but excluded 
    from the final solution due to suboptimal segmentation results compared 
    to K-Means.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Visualization Settings ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_meanshift_clustering(data_filepath):
    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(data_filepath, sep='\t')
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 1. Feature Selection
    # Consistent feature set used across all experimental models
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
    # MeanShift calculations are distance-based, making scaling essential
    print("--- Standardizing Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Bandwidth Estimation
    # MeanShift requires a bandwidth parameter (radius of the area). 
    # We estimate it automatically based on the data distribution.
    print("--- Estimating Bandwidth ---")
    # quantile=0.2 is a standard starting point for this dataset size
    bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)
    print(f"Estimated Bandwidth: {bandwidth:.4f}")

    # 4. Modeling: MeanShift
    print("--- Training MeanShift Model ---")
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X_scaled)

    # Adding labels to dataframe
    df_clean['Cluster_MeanShift'] = labels
    
    # Analyzing the number of clusters found
    n_clusters = len(np.unique(labels))
    print(f"Number of estimated clusters: {n_clusters}")

    # 5. Visualization via PCA
    print("--- Generating Visualization ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster_MeanShift', 
        data=df_clean, 
        palette='tab20', # High contrast palette for potentially many clusters
        s=60, alpha=0.7,
        legend='full'
    )
    plt.title(f'MeanShift Clustering (PCA Projection)\nClusters Found: {n_clusters}', fontsize=15)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster ID', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()

    print("--- Analysis Complete ---")
    print("Note: This model is included for comparative purposes only.")

if __name__ == "__main__":
    DATA_PATH = 'marketing_campaign.csv'
    run_meanshift_clustering(DATA_PATH)
