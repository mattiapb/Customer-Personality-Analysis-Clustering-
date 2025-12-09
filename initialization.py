"""
Project: Customer Personality Analysis
Module: Data Cleaning & Exploratory Data Analysis (EDA)
Date: Spring 2024

Description:
    This script performs the initial data loading, preprocessing, feature engineering,
    and exploratory visualization. It prepares the dataset for the clustering 
    pipeline by handling missing values, removing outliers, and creating 
    business-relevant features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Visualization Configuration ---
# Setting the aesthetic style of the plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    # 1. Data Loading
    print("--- Loading Data ---")
    # The dataset uses tab separation
    try:
        df = pd.read_csv('marketing_campaign.csv', sep='\t')
        print(f"Data loaded successfully. Original Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'marketing_campaign.csv' not found. Please check the file path.")
        return

    # 2. Handling Missing Values
    # Imputing null values in 'Income' with the mean to preserve data distribution
    if df['Income'].isnull().sum() > 0:
        print(f"Imputing {df['Income'].isnull().sum()} missing values in 'Income' with the mean.")
        df['Income'] = df['Income'].fillna(df['Income'].mean())

    # 3. Feature Engineering
    # Creating 'Customer_Age': Transforming 'Year_Birth' into an age variable (Reference Year: 2024)
    df['Customer_Age'] = 2024 - df['Year_Birth']

    # Creating 'Total_Spent': Aggregating spending across all product categories
    df['Total_Spent'] = (
        df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + 
        df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    )

    # Creating 'Tenure': Calculating the number of days since the customer enrolled
    # Parsing 'Dt_Customer' to datetime objects
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    df['Tenure'] = (datetime.now() - df['Dt_Customer']).dt.days

    # 4. Outlier Removal
    # Filtering based on EDA findings: Removing unlikely birth years and extreme income values
    print("--- Removing Outliers ---")
    initial_rows = len(df)
    
    # Filter: Born after 1930 and Income < $200,000 (to reduce noise)
    df = df[(df['Year_Birth'] >= 1930) & (df['Income'] < 200000)]
    
    rows_removed = initial_rows - len(df)
    print(f"Rows removed: {rows_removed}")
    print(f"Cleaned Dataset Shape: {df.shape}")

    # 5. Dropping Irrelevant Features
    # Removing ID and internal codes that do not contribute to customer segmentation
    cols_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # --- Exploratory Data Analysis (EDA) Plots ---
    print("--- Generating Visualizations ---")

    # Plot 1: Age Distribution
    plt.figure()
    sns.histplot(df['Customer_Age'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Customer Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    # Plot 2: Income Distribution
    plt.figure()
    sns.histplot(df['Income'], bins=30, kde=True, color='gold')
    plt.title('Distribution of Annual Income')
    plt.xlabel('Income ($)')
    plt.ylabel('Frequency')
    plt.show()

    # Plot 3: Spending by Category (Subplots)
    products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    titles = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']
    colors = ['purple', 'orange', 'red', 'blue', 'pink', 'yellow']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Spending Distribution by Product Category', fontsize=16)

    for ax, col, title, color in zip(axes.flatten(), products, titles, colors):
        sns.histplot(df[col], bins=30, kde=False, color=color, ax=ax)
        ax.set_title(f'Spending on {title}')
        ax.set_xlabel('Amount ($)')
        ax.set_ylabel('Count')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot 4: Recency Distribution
    plt.figure()
    sns.histplot(df['Recency'], bins=30, kde=False, color='teal')
    plt.title('Distribution of Recency (Days since last purchase)')
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.show()

    # Plot 5: Campaign Acceptance
    campaigns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    campaign_counts = df[campaigns].sum()

    plt.figure()
    sns.barplot(x=campaign_counts.index, y=campaign_counts.values, palette='magenta')
    plt.title('Total Acceptances by Marketing Campaign')
    plt.xlabel('Campaign')
    plt.ylabel('Total Acceptances')
    plt.xticks(rotation=45)
    plt.show()

    print("--- Process Completed ---")

if __name__ == "__main__":
    main()
