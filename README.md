# Customer Personality Analysis

## Project Overview
Customer personality analysis is a detailed analysis of a company's ideal customers based on transactional and behavioral data. By understanding customer habits, businesses can modify products and marketing campaigns to specific segments.

This project implements **Unsupervised Machine Learning** techniques to segment a customer base into distinct clusters. The process began with a deep Exploratory Data Analysis (EDA) to understand distributions and correlations, followed by rigorous Feature Engineering, and finally the implementation of clustering algorithms (K-Means, Agglomerative Clustering, and Ensemble methods) to derive actionable business insights.

**Original Dataset:** [Kaggle: Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

*University of Chile - Faculty of Physical and Mathematical Sciences (FCFM)*

---

## 1. Exploratory Data Analysis (EDA)
Before modeling, we conducted a statistical analysis of the 29 original variables to identify trends, outliers, and potential correlations.

### Data Demographics
We analyzed the distribution of key demographic variables such as **Year of Birth** and **Income**.
* **Age Structure:** Identified the predominant age groups by calculating `Customer_Age` (2024 - Year_Birth).
* **Income Distribution:** Analyzed mean, median, and standard deviation to understand customer purchasing power. We filtered out outliers (e.g., birth years before 1930 or income > $200,000) to avoid skewing the model.

![Distribution of User Income](path/to/income_distribution_image.png)
*Figure 1: Distribution of User Income showing the concentration of purchasing power.*

### Purchasing Behavior
We examined spending habits across different categories:
* **Products:** Wines, Fruits, Meat, Fish, Sweets, and Gold.
* **Recency:** Evaluated the days since the last purchase to measure customer retention and engagement.

![Distribution of Wine Purchases](path/to/wine_purchases_image.png)
*Figure 2: Distribution of spending on Wine products.*

### Channel Preference & Promotion Sensitivity
We analyzed how customers interact with sales channels (Web, Catalog, Store) and their response history to previous marketing campaigns (`AcceptedCmp1` through `AcceptedCmp5`).

---

## 2. Feature Engineering & Preprocessing
To improve the quality of the clustering, we transformed the raw data through several steps.

### New Feature Creation
Based on the EDA, we engineered new variables to capture complex behaviors:
* **Customer_Age:** Calculated from `Year_Birth` to categorize users by age group.
* **Total Spent:** Sum of all product categories (`MntWines` + `MntFruits` + ...).
* **Cmp_Acc_Rate:** The proportion of marketing campaigns accepted by the client.
* **Purchase_Channel_Preference:** Derived to identify the primary shopping channel for each user.

### Data Cleaning
* **Imputation:** Missing values were filled using the mean strategy.
* **Dimensionality Reduction:** We applied **PCA (Principal Component Analysis)** to reduce the feature space to 2 principal components, optimizing visualization and computational efficiency.
* **Feature Selection:** Non-informative variables (e.g., `ID`, `Z_CostContact`) and redundant demographic variables were removed prior to model training.

---

## 3. Modeling Strategy
We implemented and compared multiple clustering algorithms to find the optimal segmentation.

### Algorithms
1.  **K-Means:** The primary model. We optimized the number of clusters (*k*) using the Elbow Method and tuned hyperparameters via GridSearch.
2.  **Agglomerative Clustering:** A hierarchical approach used to validate the stability of the clusters found by K-Means.
3.  **Ensemble Method:** A "Majority Voting" system that combined predictions from K-Means and Agglomerative Clustering to assign the most robust cluster label.

*Note: Other models such as DBSCAN, Affinity Propagation, and Spectral Clustering were tested but discarded due to inconsistent segmentation results.*

![Clustering Visualization with PCA](path/to/kmeans_pca_image.png)
*Figure 3: Visualization of the 3 final clusters using PCA components.*

---

## 4. Results & Evaluation
We used the **Silhouette Score** as the primary metric to evaluate cluster cohesion and separation.

| Clustering Model | Silhouette Score | Performance |
| :--- | :--- | :--- |
| **K-Means** | **0.5814** | **Best Fit** |
| Ensemble (Voting) | 0.5761 | High |
| Agglomerative Clustering | 0.5754 | High |

### Cluster Profiles
The analysis resulted in **3 distinct customer personalities**:

#### Cluster 0: "The Average Customer"
* **Profile:** Moderate income and balanced spending habits across all channels (Web, Store, Catalog).
* **Behavior:** Generally low acceptance of initial campaigns but shows average responsiveness to final offers.
* **Strategy:** Best targeted via multichannel campaigns emphasizing convenience.

#### Cluster 1: "The Elite / High-Net-Worth"
* **Profile:** Highest income bracket. Significant spending on high-margin luxury goods (Wine, Gold, Meat).
* **Behavior:** Very high response rate to marketing campaigns (approx. 71% acceptance).
* **Strategy:** Ideal targets for premium product launches and exclusive VIP offers.

#### Cluster 2: "The Budget / Family-Oriented"
* **Profile:** Lowest income, often associated with larger families (more children/dependents).
* **Behavior:** Spending is focused on necessities; very low discretionary spending and campaign engagement.
* **Strategy:** Responsive to cost-effective promotions and discounts on essential goods.

---

## 5. Conclusions & Business Insights
1.  **Targeted Marketing:** High-cost campaigns for luxury items must be strictly funnelled to **Cluster 1** to maximize ROI, as they exhibit the highest propensity to purchase.
2.  **Cost Efficiency:** Marketing to **Cluster 2** should be minimized or focused solely on high-discount essential items to avoid wasted ad spend.
3.  **Omnichannel Engagement:** **Cluster 0** represents the core user base that utilizes the entire ecosystem; maintaining a consistent user experience across the App, Web, and Store is crucial for retaining this segment.

## 6. Future Improvements
While the current analysis provides strong insights, incorporating the following data points could further refine the segmentation:
* **Geolocation:** Analyzing customer region/location to account for cultural or logistical preferences.
* **Online Behavior:** Detailed web logs (time on site, clicked items) to infer interest beyond just transaction history.
* **Social Media Interaction:** Data on brand engagement across social platforms to measure brand loyalty and sentiment.

---
*Project developed for the Electrical Engineering Laboratory course, Spring 2024.*
