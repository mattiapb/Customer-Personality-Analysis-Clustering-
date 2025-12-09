"""
Project: Customer Personality Analysis
Module: Experimental Clustering - Self-Organizing Maps (SOM)
Date: Spring 2024

Description:
    This script implements a Self-Organizing Map (SOM), a type of Artificial 
    Neural Network trained using unsupervised learning to produce a 
    low-dimensional (typically two-dimensional) representation of the input 
    space of the training samples, called a map.
    
    Dependencies:
        - minisom (Requires installation: pip install minisom)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Note: Ensure minisom is installed in your environment
try:
    from minisom import MiniSom
except ImportError:
    print("Error: 'minisom' library not found. Please install it using 'pip install minisom'")

# --- Visualization Settings ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

def run_som_clustering(data_filepath):
    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv(data_filepath, sep='\t')
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 1. Feature Selection
    # Using the same feature set as the main models for consistency
    features = [
        'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    
    # Handling missing values
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features].values

    # 2. Scaling
    # SOM performance is highly sensitive to scaling. 
    # MinMaxScaler (0-1) is often preferred for SOMs over StandardScaler, 
    # though both can be used. We use MinMaxScaler here for neural network stability.
    print("--- Scaling Features (MinMax) ---")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Modeling: Self-Organizing Map
    # Parameters:
    #   x, y: Dimension of the grid (e.g., 10x10 neurons)
    #   input_len: Number of features
    #   sigma: Radius of the different neighbors in the SOM
    #   learning_rate: How much weights are adjusted during iteration
    print("--- Training SOM Network ---")
    
    som_grid_rows = 10
    som_grid_columns = 10
    input_len = X_scaled.shape[1]
    
    som = MiniSom(x=som_grid_rows, y=som_grid_columns, input_len=input_len, sigma=1.0, learning_rate=0.5)
    
    # Initialization and Training
    som.random_weights_init(X_scaled)
    print("Training initiated...")
    som.train_random(data=X_scaled, num_iteration=1000) # 1000 iterations as a baseline
    print("Training complete.")

    # 4. Visualization: Distance Map (U-Matrix)
    # The U-matrix shows the distance between neighboring neurons.
    # Darker areas represent clusters (low distance), light areas represent borders (high distance).
    print("--- Visualizing Distance Map (U-Matrix) ---")
    
    plt.figure(figsize=(10, 10))
    # distance_map() returns the matrix of distances
    plt.pcolor(som.distance_map().T, cmap='bone_r') 
    plt.colorbar(label='Mean Inter-neuron Distance')
    
    plt.title('Self-Organizing Map (U-Matrix)', fontsize=16)
    plt.xlabel('Neuron Row')
    plt.ylabel('Neuron Column')
    plt.show()

    print("--- Analysis Complete ---")
    print("Note: Darker regions indicate potential clusters. Lighter regions indicate boundaries.")

if __name__ == "__main__":
    DATA_PATH = 'marketing_campaign.csv'
    run_som_clustering(DATA_PATH)
