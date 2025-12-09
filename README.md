# Customer Personality Analysis: A Clustering Approach

## 📌 Project Overview
Customer personality analysis is a detailed analysis of a company's ideal customers based on transactional and behavioral data. By understanding customer habits, businesses can modify products and marketing campaigns to specific segments.

This project implements **Unsupervised Machine Learning** techniques to segment a customer base into distinct clusters. By analyzing these groups, we derived actionable business insights to optimize marketing strategies and maximize revenue.

**Original Dataset:** [Kaggle: Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

## 👥 Team Members
* **Mattia Pinilla**
* Jorge Córdoba
* Ignacio López
* Carlos Flores
* Francisco Santibáñez

*University of Chile - Faculty of Physical and Mathematical Sciences (FCFM)*

## 🛠️ Methodology & Workflow

### 1. Data Preprocessing & Feature Engineering
To ensure high-quality clusters, the raw dataset (29 variables) underwent rigorous cleaning:
* **Imputation:** Missing values were filled using the mean strategy.
* **Outlier Removal:** Records with birth years prior to 1930 or annual incomes exceeding $200k were removed to eliminate noise.
* **Feature Selection:** Irrelevant IDs and non-discriminatory variables were dropped to reduce dimensionality.
* **Standardization:** Data was scaled using `StandardScaler` (mean=0, std=1).
* **Dimensionality Reduction:** Principal Component Analysis (PCA) was applied to reduce the feature space to 2 components for visualization and computational efficiency.

### 2. Modeling Strategy
We experimented with multiple clustering algorithms to find the optimal segmentation:
* **K-Means (Selected Model):** Optimized using the Elbow Method and GridSearch for hyperparameter tuning.
* **Agglomerative Clustering:** Hierarchical approach used for comparison.
* **Ensemble Method:** A "Majority Voting" system combining K-Means and Agglomerative Clustering.
* *Discarded Models:* DBSCAN, Affinity Propagation, GMM, MeanShift, and Spectral Clustering (due to poor segmentation performance).

## 📊 Results & Evaluation

We evaluated the models using the **Silhouette Score** to measure cluster cohesion and separation.

| Clustering Model | Silhouette Score | Performance |
| :--- | :--- | :--- |
| **K-Means** | **0.5814** | **Best Fit** |
| Ensemble (Voting) | 0.5761 | High |
| Agglomerative Clustering | 0.5754 | High |

### Cluster Profiling (K-Means)
The analysis identified **3 distinct customer personalities**:

#### 🟢 Cluster 0: "The Average Customer"
* **Profile:** Moderate income, balanced spending across all channels (Web, Store, Catalog).
* **Behavior:** Low initial campaign acceptance but responsive to final offers.
* **Strategy:** Multichannel marketing campaigns emphasizing convenience and variety.

#### 🟠 Cluster 1: "The Elite / High-Net-Worth"
* **Profile:** Highest income bracket, high spending on luxury goods (Wine, Gold, Meat).
* **Behavior:** Very high response rate to marketing campaigns (approx. 71%).
* **Strategy:** Target with premium/luxury products and exclusive VIP offers.

#### 🔵 Cluster 2: "The Budget / Family-Oriented"
* **Profile:** Lowest income, typically families with more children/dependents.
* **Behavior:** Low discretionary spending, primarily purchases necessities. Very low campaign engagement.
* **Strategy:** Focus on cost-effective promotions, discounts on essential goods (e.g., Fruits), and family bundles.

## 🚀 Key Business Insights
1.  **Luxury Targeting:** Marketing efforts for high-margin products (Gold, Wine) should be strictly funnelled to **Cluster 1**, as they show the highest propensity to buy and respond to promos.
2.  **Efficiency:** Avoid targeting **Cluster 2** with high-cost acquisition campaigns; instead, use organic reach or massive discounts on low-margin staples.
3.  **Omnichannel Approach:** **Cluster 0** represents the core user base that utilizes all shopping interfaces; maintaining a consistent user experience across the App, Web, and Store is crucial for this segment.

## 💻 Tech Stack
* **Python**
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.
* **Techniques:** PCA, K-Means, Hierarchical Clustering, GridSearch.

---
*Project developed for the Electrical Engineering Laboratory course, Spring 2024.*
