"""
Project: Customer Personality Analysis
Module: Final Clustering Model - K-Means with GridSearchCV
Authors: Mattia Pinilla, Jorge Córdoba, Ignacio López, Carlos Flores, Francisco Santibáñez
Date: Spring 2024

Description:
    This script implements the primary clustering solution using K-Means.
    It includes a complete pipeline:
    1. Data Cleaning & Feature Engineering
    2. Standardization (StandardScaler)
    3. Dimensionality Reduction (PCA)
    4. Hyperparameter Optimization (GridSearchCV)
    5. Evaluation (Silhouette Score & Elbow Method)
    6. Visualization of the final clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

# --- Visualization Settings ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_preprocess_data(filepath):
    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(filepath, sep='\t')
    except FileNotFoundError:
        print("Error: File not found.")
        return None, None

    # 1. Cleaning & Imputation
    # Imputing missing Income values
    df['Income'] = df['Income'].fillna(df['Income'].mean())
    
    # Removing outliers (Birth Year < 1930, Income > 200k)
    df = df[(df['Year_Birth'] >= 1930) & (df['Income'] < 200000)]
    
    # 2. Feature Selection
    # Selecting columns relevant to customer behavior and demographics
    selected_features = [
        'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    
    # Creating a copy for processing
    X = df[selected_features].copy()
    
    # 3. Standardization
    # Transforming data to mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled

def run_elbow_method(X_scaled):
    """
    Plots the Elbow Curve to help visualize the optimal number of clusters.
    """
    print("--- Running Elbow Method ---")
    inertia = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        
    plt.figure()
    plt.plot(K_range, inertia, 'bo-', markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def run_kmeans_optimization(df, X_scaled):
    print("--- Hyperparameter Tuning with GridSearchCV ---")
    
    # 1. Dimensionality Reduction (PCA)
    # Reducing to 2 components for visualization and noise reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 2. Grid Search Setup
    # We define a custom wrapper or use a loop since KMeans isn't a supervised classifier
    # However, for this script, we'll simulate the GridSearch logic found in the report
    # focusing on 'n_clusters', 'init', and 'n_init'
    
    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'init': ['k-means++', 'random'],
        'n_init': [10, 15, 20]
    }
    
    print(f"Testing parameters: {param_grid}")
    
    best_score = -1
    best_params = {}
    best_labels = None
    
    # Iterating through parameters to maximize Silhouette Score
    for k in param_grid['n_clusters']:
        for init_method in param_grid['init']:
            for n_init_val in param_grid['n_init']:
                kmeans = KMeans(
                    n_clusters=k, 
                    init=init_method, 
                    n_init=n_init_val, 
                    random_state=42
                )
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                
                if score > best_score:
                    best_score = score
                    best_params = {'n_clusters': k, 'init': init_method, 'n_init': n_init_val}
                    best_labels = labels
                    best_model = kmeans

    print(f"Best Silhouette Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

    # 3. Final Model Application
    # Assigning the best cluster labels to the dataframe
    df['Cluster'] = best_labels
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    return df, best_params

def visualize_results(df):
    print("--- Visualizing Final Clusters ---")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster', 
        data=df, 
        palette='viridis', 
        s=80, alpha=0.8
    )
    plt.title('Final Customer Segments (K-Means)', fontsize=16)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster ID')
    plt.tight_layout()
    plt.show()

    # Optional: Boxplots for cluster profiling (e.g., Income vs Cluster)
    plt.figure()
    sns.boxplot(x='Cluster', y='Income', data=df, palette='viridis')
    plt.title('Income Distribution by Cluster')
    plt.show()

if __name__ == "__main__":
    DATA_PATH = 'marketing_campaign.csv'
    
    # Pipeline Execution
    df_raw, X_scaled = load_and_preprocess_data(DATA_PATH)
    
    if df_raw is not None:
        # Step 1: Visualize optimal k
        run_elbow_method(X_scaled)
        
        # Step 2: Optimize and Fit
        df_final, best_params = run_kmeans_optimization(df_raw, X_scaled)
        
        # Step 3: Visualize Results
        visualize_results(df_final)
        
        # Output cluster statistics
        print("\n--- Cluster Statistics ---")
        print(df_final.groupby('Cluster')[['Income', 'Total_Spent', 'Recency']].mean())
