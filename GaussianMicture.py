"""
Project: Customer Personality Analysis
Module: Experimental Clustering - Gaussian Mixture Models (GMM)
Authors: Mattia Pinilla, Jorge Córdoba, Ignacio López, Carlos Flores, Francisco Santibáñez
Date: Spring 2024

Description:
    This script implements Gaussian Mixture Models (GMM) for clustering.
    Unlike K-Means, GMM assumes that data points are generated from a mixture 
    of a finite number of Gaussian distributions with unknown parameters.

    NOTE: As detailed in the final report, this model was excluded from the 
    final solution. The resulting segmentation was found to be inconsistent 
    or overly fragmented compared to the K-Means approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Visualization Settings ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_gmm_clustering(data_filepath):
    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(data_filepath, sep='\t')
    except FileNotFoundError:
        print("Error: File not found. Please check the filepath.")
        return

    # 1. Feature Selection
    # Using the standard feature set defined in the project scope
    features = [
        'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    
    # Dropping missing values for the selected features
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]

    # 2. Standardization
    # Essential for GMM to properly estimate covariance matrices
    print("--- Standardizing Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Modeling: Gaussian Mixture
    # We test with n_components=4 based on experimental observations
    # covariance_type='full' allows each component to have its own general covariance matrix
    print("--- Training Gaussian Mixture Model ---")
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X_scaled)

    # Adding labels to dataframe
    df_clean['Cluster_GMM'] = labels
    
    # Probabilistic soft clustering (optional analysis)
    # GMM allows us to see the probability of belonging to each cluster
    probs = gmm.predict_proba(X_scaled)
    print(f"GMM Converged: {gmm.converged_}")
    print(f"Number of iterations: {gmm.n_iter_}")

    # 4. Dimensionality Reduction for Visualization
    print("--- Applying PCA for Visualization ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]

    # 5. Visualization
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster_GMM', 
        data=df_clean, 
        palette='tab10', 
        s=60, alpha=0.7,
        legend='full'
    )
    plt.title('Gaussian Mixture Models (PCA Projection)', fontsize=15)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster ID')
    plt.tight_layout()
    plt.show()

    print("--- Analysis Complete ---")
    print("Note: This model is included for comparative purposes only.")

if __name__ == "__main__":
    DATA_PATH = 'marketing_campaign.csv'
    run_gmm_clustering(DATA_PATH)
