"""
Project: Customer Personality Analysis
Module: Experimental Clustering - Affinity Propagation
Date: Spring 2024

Description:
    This script implements the Affinity Propagation clustering algorithm.
    
    NOTE: As detailed in the final report, this model was part of the 
    exploratory phase. It was ultimately discarded because it generated 
    an excessive number of clusters and lacked consistent segmentation 
    compared to K-Means.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- Visualization Settings ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_affinity_propagation(data_filepath):
    print("--- Loading and Preprocessing Data ---")
    # Load the dataset (Assuming data is already cleaned from the previous module)
    try:
        df = pd.read_csv(data_filepath, sep='\t')
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Selecting numerical features for clustering
    # We exclude non-numeric or pre-processed ID columns
    features = [
        'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth', 'Customer_Age', 'Total_Spent'
    ]
    
    # Ensure all selected features exist in the dataframe
    selected_features = [col for col in features if col in df.columns]
    X = df[selected_features].dropna()

    # Standardization
    # Scaling data to mean=0 and std=1 is crucial for distance-based algorithms
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("--- Training Affinity Propagation Model ---")
    # affinity: 'euclidean' is standard. 
    # damping: Controls the extent to which the current value is maintained relative to incoming values.
    model = AffinityPropagation(damping=0.9, random_state=42)
    labels = model.fit_predict(X_scaled)

    # Adding cluster labels to the original dataframe
    df_result = df.loc[X.index].copy()
    df_result['Cluster'] = labels

    n_clusters = len(set(labels))
    print(f"Model converged. Number of clusters found: {n_clusters}")

    # Calculating Silhouette Score
    # A low score here confirms the poor performance mentioned in the report
    try:
        sil_score = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {sil_score:.4f}")
    except ValueError:
        print("Error calculating Silhouette Score (possibly only one cluster found).")

    # --- Dimensionality Reduction for Visualization ---
    print("--- Generating PCA Visualization ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_result['PCA1'] = X_pca[:, 0]
    df_result['PCA2'] = X_pca[:, 1]

    # --- Plotting Results ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster', 
        data=df_result, 
        palette='tab20', # Using a large palette as AP tends to create many clusters
        legend='full',
        s=60, alpha=0.7
    )
    plt.title(f'Affinity Propagation Clustering (PCA Projection)\nDetected Clusters: {n_clusters}', fontsize=15)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Handling legend for many clusters
    if n_clusters > 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cluster ID', ncol=2)
    else:
        plt.legend(title='Cluster ID')
        
    plt.tight_layout()
    plt.show()

    print("--- Analysis Complete ---")
    print("Note: This model is included for comparative purposes only.")

if __name__ == "__main__":
    # Ensure this points to your CLEANED dataset
    DATA_PATH = 'marketing_campaign.csv' 
    run_affinity_propagation(DATA_PATH)
