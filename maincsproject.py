#maincsproject
# %% [markdown]
# # PROJECT TITLE

# %% [markdown]
# ### CUSTOMER SEGMENTATION AND PRODUCT RECOMENDATION IN ECOMMERCE.

# %% [markdown]
# ## PROBLEM STATEMENT

# %% [markdown]
# The gobal e-commerce industry generates vast amounts of transaction data daily, offering valuable insights into customer purchasing behaviors. Analyzing this data is essential for identifying meaningful customer segments and recommending relevant products to enhance customer experience and drive business growth. This project aims to examine transaction data from an online retail business to uncover patterns in customer purchase behavior, segment customers based on Recency, Frequency, and Monetary (RFM) analysis, and develop a product recommendation system using collaborative filtering techniques.
# 

# %% [markdown]
# importing necessary libraries for ETL process

# %%
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
#matplotlib inline 

# %%
#for machine learning and clusteering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# %%
#Date and time 
import datetime as dt
import warnings

#ignore warnings 
warnings.filterwarnings('ignore')


# %% [markdown]
# Loading Dataset
# importing csv file

# %%
order = pd.read_csv('online_retail.csv')


# %%
print(order, order.shape)


# %%
order.head()
order.info()

# %%
order.describe()

# %%


# %%
duplicate_count = order.duplicated()
print(f"Number of duplicate rows: {duplicate_count}")

# %%
missing_values = order.isnull().sum()
print(f"Number of missing values in each column:\n{missing_values}")

# %% [markdown]
# Visualisation of DATA

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(order.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Heatmap of Missing Values")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()


# %% [markdown]
# ### UNDERSTANDING VARIABLES

# %%
print(order.columns.tolist())


# %% [markdown]
# ## DATA WRANGLING

# %% [markdown]
# ## Step 1: Remove rows with missing CustomerID
# 

# %%
order_cleaned = order.drop_duplicates()

# %% [markdown]
# ## Step 2 Remove cancelled transaction

# %%
order_cleaned['InvoiceNo'] = order_cleaned['InvoiceNo'].astype(str)
order_cleaned['InvoiceNo'] = order_cleaned['InvoiceNo'].str.startswith('C').astype(int)

# %%
(order_cleaned).info()

# %% [markdown]
# ## Covert InvoiceDate into datetime format

# %%
order_cleaned['InvoiceDate'] = pd.to_datetime(order_cleaned['InvoiceDate'])

# %% [markdown]
# ## Convert Customer id into String type

# %%
order_cleaned = order_cleaned.dropna(subset=['CustomerID']).copy()
order_cleaned['CustomerID'] = order_cleaned['CustomerID'].astype(int).astype(str)

# %%

order_cleaned.reset_index(drop=True, inplace=True)


# %% [markdown]
# # VISUALISATION OF DATA

# %% [markdown]
# CHART 1 : TOP 10 PRODUCTS SOLD

# %%
top_products = order_cleaned.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# %%
plt.figure(figsize=(12, 6))
sns.barplot(x=top_products.values, y=top_products.index, palette='rocket')
plt.title('Top 10 Most Sold Products')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product Description')
plt.show()

# %% [markdown]
# ## Product sold in top 10 Location

# %%
top_locations = order_cleaned.groupby('Country')['Quantity'].sum().sort_values(ascending=False).head(10)

# %%
plt.figure(figsize=(10, 5))
sns.barplot(x=top_locations.values, y=top_locations.index, palette='rocket')
plt.title('Top 10 Most Active Locations')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Country')
plt.show()

# %% [markdown]
# ## Changing in Purchace Trend over time.

# %%
order_cleaned['InvoiceMonth'] = order_cleaned['InvoiceDate'].dt.to_period('M')
monthly_trend = order_cleaned.groupby('InvoiceMonth')['Quantity'].sum().reset_index()
monthly_trend['InvoiceMonth'] = monthly_trend['InvoiceMonth'].dt.to_timestamp()

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_trend, x='InvoiceMonth', y='Quantity', marker='o')
plt.title('Purchase Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Monetry distribution between transaction and customer

# %%
valid_transactions = (
    order.drop_duplicates()
    .dropna(subset=['CustomerID'])
    .copy()
)

valid_transactions['InvoiceNo'] = valid_transactions['InvoiceNo'].astype(str)
valid_transactions = valid_transactions[~valid_transactions['InvoiceNo'].str.startswith('C', na=False)].copy()
valid_transactions['InvoiceDate'] = pd.to_datetime(valid_transactions['InvoiceDate'])
valid_transactions['CustomerID'] = valid_transactions['CustomerID'].astype(int).astype(str)
valid_transactions['Revenue'] = valid_transactions['Quantity'] * valid_transactions['UnitPrice']
valid_transactions = valid_transactions[valid_transactions['Quantity'] > 0].copy()

transaction_revenue = valid_transactions.groupby('InvoiceNo')['Revenue'].sum()
customer_revenue = valid_transactions.groupby('CustomerID')['Revenue'].sum()

print("Transaction revenue summary:")
print(transaction_revenue.describe())
print("\nCustomer revenue summary:")
print(customer_revenue.describe())

top_transactions = transaction_revenue.sort_values(ascending=False).head(5)
top_customers = customer_revenue.sort_values(ascending=False).head(5)

print("\nTop 5 transactions by revenue:")
print(top_transactions)
print("\nTop 5 customers by revenue:")
print(top_customers)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(transaction_revenue, bins=50, kde=True)
plt.title('Monetary Distribution per Transaction')
plt.xlabel('Transaction Revenue')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.histplot(customer_revenue, bins=50, kde=True)
plt.title('Monetary Distribution per Customer')
plt.xlabel('Customer Revenue')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### RFM distributions
# 

# %%
# RFM distributions
# Assumes valid_transactions, pd, plt, sns, np are available in the notebook

# ensure InvoiceDate is datetime
valid_transactions['InvoiceDate'] = pd.to_datetime(valid_transactions['InvoiceDate'])

snapshot_date = valid_transactions['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = valid_transactions.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('Revenue', 'sum')
)

print(rfm.describe().T)

# Plot distributions
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(rfm['Recency'], bins=50, kde=True, color='tab:blue')
plt.title('Recency (days)')
plt.xlabel('Days since last purchase')

plt.subplot(1, 3, 2)
sns.histplot(rfm['Frequency'], bins=50, kde=True, color='tab:green')
plt.title('Frequency (unique invoices)')
plt.xlabel('Number of transactions')

plt.subplot(1, 3, 3)
sns.histplot(rfm['Monetary'], bins=50, kde=True, color='tab:orange')
plt.title('Monetary (total revenue)')
plt.xlabel('Revenue')
plt.xscale('log')  # monetary typically skewed; log scale helps visualization

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Customer cluster profiles
# 

# %%
# Customer cluster profiles (uses existing `rfm`, `KMeans`, `StandardScaler`, `silhouette_score`, `np`, `plt`, `sns`)

rfm_clust = rfm.copy()

# stabilize skewed features
rfm_clust['Monetary_log'] = np.log1p(rfm_clust['Monetary'])
rfm_clust['Frequency_log'] = np.log1p(rfm_clust['Frequency'])

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(rfm_clust[['Recency', 'Frequency_log', 'Monetary_log']])

# choose best k by silhouette (k=2..6)
scores = []
models = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    scores.append(silhouette_score(X, labels))
    models[k] = km

best_k = list(range(2, 7))[int(np.argmax(scores))]
best_k, scores

# fit best model and assign clusters
best_km = models[best_k]
rfm_clust['Cluster'] = best_km.labels_.astype(int)

# cluster profiles
profiles = rfm_clust.groupby('Cluster').agg(
    NumCustomers=('Cluster', 'count'),
    Recency_mean=('Recency', 'mean'),
    Frequency_mean=('Frequency', 'mean'),
    Monetary_mean=('Monetary', 'mean')
).sort_index()

print("Chosen k:", best_k)
print(profiles)

# normalized cluster means for visualization
cluster_means = profiles[['Recency_mean', 'Frequency_mean', 'Monetary_mean']]
cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

plt.figure(figsize=(10, 5))
cluster_norm.plot(kind='bar')
plt.title('Normalized RFM Means per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Normalized value')
plt.legend(title='RFM')
plt.tight_layout()
plt.show()

# customer counts per cluster
plt.figure(figsize=(6,4))
sns.barplot(x=profiles.index, y=profiles['NumCustomers'], palette='viridis')
plt.title('Number of Customers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Num Customers')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Elbow curve for cluster selection
# 

# %%
# Elbow curve for cluster selection
inertias = []
k_range = range(2, 7)

for k in k_range:
    inertias.append(models[k].inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve for Optimal Cluster Selection')
plt.grid(True, alpha=0.3)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Display all clusters with their characteristics
print("=" * 80)
print("CUSTOMER SEGMENTATION CLUSTERS")
print("=" * 80)

for cluster_id in sorted(rfm_clust['Cluster'].unique()):
    cluster_data = rfm_clust[rfm_clust['Cluster'] == cluster_id]
    print(f"\n{'─' * 80}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'─' * 80}")
    print(f"Number of Customers: {len(cluster_data)}")
    print(f"\nRFM Metrics:")
    print(f"  Recency (days since last purchase):")
    print(f"    Mean: {cluster_data['Recency'].mean():.2f}, Median: {cluster_data['Recency'].median():.2f}")
    print(f"  Frequency (number of transactions):")
    print(f"    Mean: {cluster_data['Frequency'].mean():.2f}, Median: {cluster_data['Frequency'].median():.2f}")
    print(f"  Monetary (total revenue):")
    print(f"    Mean: ${cluster_data['Monetary'].mean():.2f}, Median: ${cluster_data['Monetary'].median():.2f}")
    print(f"\nCluster Profile Summary:")
    print(f"  {profiles.loc[cluster_id].to_string()}")

print(f"\n{'=' * 80}")
print("CLUSTER SUMMARY TABLE")
print(f"{'=' * 80}")
print(profiles)

# %% [markdown]
# ## Product recommendation heatmap / similarity matrix
# 

# %%
# Product recommendation using collaborative filtering
# Create user-item interaction matrix
user_item_matrix = valid_transactions.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Calculate cosine similarity between products
product_similarity = cosine_similarity(user_item_matrix.T)
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# Visualize similarity matrix for top products
top_stock_codes = valid_transactions.groupby('StockCode')['Quantity'].sum().nlargest(10).index
similarity_subset = product_similarity_df.loc[top_stock_codes, top_stock_codes]

plt.figure(figsize=(12, 10))
sns.heatmap(similarity_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Cosine Similarity'}, square=True)
plt.title('Product Similarity Matrix (Top 10 Products by Quantity)')
plt.xlabel('Stock Code')
plt.ylabel('Stock Code')
plt.tight_layout()
plt.show()

# %% [markdown]
# # CLUSTER METHODOLGY

# %%
# Calculate Recency (days since last purchase) per customer
last_purchase = valid_transactions.groupby('CustomerID')['InvoiceDate'].max()
recency_days = (snapshot_date - pd.to_datetime(last_purchase)).dt.days

# update or create rfm DataFrame with Recency
if 'rfm' in globals():
    rfm['Recency'] = recency_days
else:
    rfm = pd.DataFrame({'Recency': recency_days})

# quick check
print(rfm['Recency'].head())
print(rfm['Recency'].describe())

# %% [markdown]
# .

# %%
# Compute number of unique transactions (Frequency) per customer and update rfm
frequency = valid_transactions.groupby('CustomerID')['InvoiceNo'].nunique()

if 'rfm' in globals():
    rfm['Frequency'] = frequency
else:
    rfm = pd.DataFrame({'Frequency': frequency})

print(rfm['Frequency'].head())
print(rfm['Frequency'].describe())

# %%
# Calculate Monetary (total revenue per customer) and update rfm
monetary = valid_transactions.groupby('CustomerID')['Revenue'].sum()

if 'rfm' in globals():
    rfm['Monetary'] = monetary
else:
    rfm = pd.DataFrame({'Monetary': monetary})

print(rfm['Monetary'].head())
print(rfm['Monetary'].describe())

# %%
# Standardize/Normalize the RFM values
rfm_normalized = rfm.copy()

# Apply log transformation to handle skewness
rfm_normalized['Recency_log'] = np.log1p(rfm_normalized['Recency'])
rfm_normalized['Frequency_log'] = np.log1p(rfm_normalized['Frequency'])
rfm_normalized['Monetary_log'] = np.log1p(rfm_normalized['Monetary'])

# Normalize using StandardScaler
scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm_normalized[['Recency_log', 'Frequency_log', 'Monetary_log']])

# Create a dataframe with scaled values
rfm_normalized_scaled = pd.DataFrame(
    rfm_scaled,
    columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'],
    index=rfm_normalized.index
)

print("Normalized RFM values (first 10 rows):")
print(rfm_normalized_scaled.head(10))
print("\nNormalized RFM statistics:")
print(rfm_normalized_scaled.describe())

# Visualization of normalized distributions
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(rfm_normalized_scaled['Recency_scaled'], bins=50, kde=True, color='tab:blue')
plt.title('Normalized Recency')
plt.xlabel('Scaled Recency')

plt.subplot(1, 3, 2)
sns.histplot(rfm_normalized_scaled['Frequency_scaled'], bins=50, kde=True, color='tab:green')
plt.title('Normalized Frequency')
plt.xlabel('Scaled Frequency')

plt.subplot(1, 3, 3)
sns.histplot(rfm_normalized_scaled['Monetary_scaled'], bins=50, kde=True, color='tab:orange')
plt.title('Normalized Monetary')
plt.xlabel('Scaled Monetary')

plt.tight_layout()
plt.show()

# %%
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Choose and compare clustering algorithms on the existing RFM scaled data
# Uses: rfm_scaled (np.ndarray), rfm (pd.DataFrame)

X = rfm_scaled  # precomputed scaled RFM (shape: n_customers x 3)

results = []

# KMeans (re-evaluate to compare)
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    labels = km.labels_
    try:
        score = silhouette_score(X, labels)
    except:
        score = np.nan
    results.append(("KMeans", {"n_clusters": k}, score, labels))

# Agglomerative (Hierarchical)
for k in range(2, 7):
    ag = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X)
    labels = ag.labels_
    try:
        score = silhouette_score(X, labels)
    except:
        score = np.nan
    results.append(("Agglomerative", {"n_clusters": k, "linkage": "ward"}, score, labels))

# Gaussian Mixture (soft clustering -> hard labels via predict)
for k in range(2, 7):
    gm = GaussianMixture(n_components=k, random_state=42).fit(X)
    labels = gm.predict(X)
    try:
        score = silhouette_score(X, labels)
    except:
        score = np.nan
    results.append(("GaussianMixture", {"n_components": k}, score, labels))

# DBSCAN (density-based) - grid search for eps/min_samples
eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
min_samples_values = [3, 5, 10]
for eps in eps_values:
    for ms in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=ms).fit(X)
        labels = db.labels_
        # count clusters ignoring noise (-1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            try:
                score = silhouette_score(X, labels)
            except:
                score = np.nan
        else:
            score = np.nan
        results.append(("DBSCAN", {"eps": eps, "min_samples": ms, "n_clusters": n_clusters}, score, labels))

# Summarize and pick best valid result by silhouette_score
res_df = pd.DataFrame([{"algo": a, "params": p, "silhouette": s} for a, p, s, _ in results])
res_df = res_df.sort_values("silhouette", ascending=False).reset_index(drop=True)

print("Top clustering results (by silhouette):")
print(res_df.head(10))

# Attach best clustering labels to rfm DataFrame (if available)
best_idx = res_df['silhouette'].first_valid_index()
if best_idx is not None and not np.isnan(res_df.loc[best_idx, 'silhouette']):
    best_entry = results[best_idx]
    best_algo, best_params, best_score, best_labels = best_entry
    # Ensure labels length matches rfm index
    if len(best_labels) == len(rfm):
        rfm['Cluster_chosen_algo'] = f"{best_algo}_{best_params}"
        rfm['Cluster_chosen'] = best_labels.astype(int)
        print(f"\nSelected best: {best_algo} with params {best_params}, silhouette={best_score:.4f}")
    else:
        print("\nBest labels length does not match rfm; labels not assigned.")
else:
    print("\nNo valid clustering found (silhouette not computed).")

# %%
# Run Clustering - Extract best clustering result and create final customer segments

# Get the best clustering result from res_df
best_idx = res_df['silhouette'].idxmax()
best_algo, best_params, best_score, best_labels = results[best_idx]

print("=" * 80)
print("BEST CLUSTERING RESULT")
print("=" * 80)
print(f"Algorithm: {best_algo}")
print(f"Parameters: {best_params}")
print(f"Silhouette Score: {best_score:.4f}")
print(f"=" * 80)

# Assign best clusters to rfm dataframe
rfm['Cluster_Final'] = best_labels.astype(int)

# Create comprehensive cluster profiles
final_profiles = rfm.groupby('Cluster_Final').agg(
    NumCustomers=('Cluster_Final', 'count'),
    Recency_mean=('Recency', 'mean'),
    Recency_median=('Recency', 'median'),
    Frequency_mean=('Frequency', 'mean'),
    Frequency_median=('Frequency', 'median'),
    Monetary_mean=('Monetary', 'mean'),
    Monetary_median=('Monetary', 'median'),
    Monetary_std=('Monetary', 'std')
).sort_index()

print("\nFINAL CLUSTER PROFILES:")
print(final_profiles)

# Visualize final clustering results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Customer count per cluster
final_profiles['NumCustomers'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Number of Customers per Cluster')
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=0)

# Plot 2: Recency distribution by cluster
rfm.boxplot(column='Recency', by='Cluster_Final', ax=axes[0, 1])
axes[0, 1].set_title('Recency Distribution by Cluster')
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Days Since Last Purchase')

# Plot 3: Frequency distribution by cluster
rfm.boxplot(column='Frequency', by='Cluster_Final', ax=axes[1, 0])
axes[1, 0].set_title('Frequency Distribution by Cluster')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Number of Transactions')

# Plot 4: Monetary distribution by cluster
rfm.boxplot(column='Monetary', by='Cluster_Final', ax=axes[1, 1])
axes[1, 1].set_title('Monetary Distribution by Cluster')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Total Revenue ($)')

plt.tight_layout()
plt.show()

# Detailed cluster interpretation
print("\n" + "=" * 80)
print("DETAILED CLUSTER INTERPRETATION")
print("=" * 80)

for cluster_id in sorted(rfm['Cluster_Final'].unique()):
    cluster_data = rfm[rfm['Cluster_Final'] == cluster_id]
    print(f"\n{'─' * 80}")
    print(f"CLUSTER {cluster_id} - {len(cluster_data)} customers")
    print(f"{'─' * 80}")
    print(f"Recency: {cluster_data['Recency'].mean():.1f} days (median: {cluster_data['Recency'].median():.1f})")
    print(f"Frequency: {cluster_data['Frequency'].mean():.2f} transactions (median: {cluster_data['Frequency'].median():.1f})")
    print(f"Monetary: ${cluster_data['Monetary'].mean():.2f} (median: ${cluster_data['Monetary'].median():.2f})")
    print(f"Revenue Share: {(cluster_data['Monetary'].sum() / rfm['Monetary'].sum() * 100):.1f}%")

# %%
# Label clusters based on RFM interpretation
cluster_labels = {}

for cluster_id in sorted(rfm['Cluster_Final'].unique()):
    cluster_data = rfm[rfm['Cluster_Final'] == cluster_id]
    
    recency_mean = cluster_data['Recency'].mean()
    frequency_mean = cluster_data['Frequency'].mean()
    monetary_mean = cluster_data['Monetary'].mean()
    
    # Overall RFM statistics for comparison
    recency_overall_mean = rfm['Recency'].mean()
    frequency_overall_mean = rfm['Frequency'].mean()
    monetary_overall_mean = rfm['Monetary'].mean()
    
    # Determine label based on RFM values
    recency_status = "Recent" if recency_mean < recency_overall_mean else "Inactive"
    frequency_status = "Frequent" if frequency_mean > frequency_overall_mean else "Infrequent"
    monetary_status = "High-Value" if monetary_mean > monetary_overall_mean else "Low-Value"
    
    cluster_labels[cluster_id] = f"{recency_status} {frequency_status} {monetary_status}"

# Add labels to rfm dataframe
rfm['Cluster_Label'] = rfm['Cluster_Final'].map(cluster_labels)

print("=" * 80)
print("CLUSTER LABELS AND INTERPRETATION")
print("=" * 80)
for cluster_id, label in sorted(cluster_labels.items()):
    print(f"Cluster {cluster_id}: {label}")

print("\n" + "=" * 80)
print("LABELED CLUSTER PROFILES")
print("=" * 80)
labeled_profiles = final_profiles.copy()
labeled_profiles['Label'] = [cluster_labels[i] for i in labeled_profiles.index]
print(labeled_profiles)

# Visualization with labels
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (cluster_id, label) in enumerate(sorted(cluster_labels.items())):
    cluster_data = rfm[rfm['Cluster_Final'] == cluster_id]
    ax.scatter(cluster_data['Recency'], cluster_data['Monetary'], 
               s=100, alpha=0.6, label=label, color=colors[i % len(colors)])

ax.set_xlabel('Recency (days since last purchase)')
ax.set_ylabel('Monetary (total revenue)')
ax.set_title('Customer Segments by RFM Analysis')
ax.legend(title='Cluster Label')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
from mpl_toolkits.mplot3d import Axes3D

# Visualize RFM clusters in 2D and 3D

# choose cluster column (fallbacks)
cluster_col = 'Cluster_Final' if 'Cluster_Final' in rfm.columns else ('Cluster' if 'Cluster' in rfm.columns else 'Cluster_chosen')

plot_df = rfm_normalized_scaled.join(rfm[cluster_col])

clusters = sorted(plot_df[cluster_col].unique())
palette = sns.color_palette('tab10', n_colors=max(3, len(clusters)))

fig = plt.figure(figsize=(14, 6))

# 2D scatter: Recency vs Monetary
ax1 = fig.add_subplot(1, 2, 1)
for i, c in enumerate(clusters):
    sub = plot_df[plot_df[cluster_col] == c]
    ax1.scatter(sub['Recency_scaled'], sub['Monetary_scaled'],
                s=30, alpha=0.6, label=f'Cluster {c}', color=palette[i % len(palette)])
ax1.set_xlabel('Recency (scaled)')
ax1.set_ylabel('Monetary (scaled)')
ax1.set_title('RFM clusters: Recency vs Monetary')
ax1.legend(title='Cluster')
ax1.grid(alpha=0.3)

# 3D scatter: Recency, Frequency, Monetary
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for i, c in enumerate(clusters):
    sub = plot_df[plot_df[cluster_col] == c]
    ax2.scatter(sub['Recency_scaled'], sub['Frequency_scaled'], sub['Monetary_scaled'],
                s=20, alpha=0.6, label=f'Cluster {c}', color=palette[i % len(palette)])
ax2.set_xlabel('Recency (scaled)')
ax2.set_ylabel('Frequency (scaled)')
ax2.set_zlabel('Monetary (scaled)')
ax2.view_init(elev=25, azim=45)
ax2.set_title('3D RFM cluster scatter')
ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %% [markdown]
# # RECOMENDED ENGINE FUNCTION

# %%

def item_based_recommendations(customer_id, top_n=10, normalize=True):
    if customer_id not in user_item_matrix.index:
        raise ValueError(f"CustomerID {customer_id} not found in user_item_matrix")

    customer_profile = user_item_matrix.loc[customer_id]
    purchased = customer_profile[customer_profile > 0]
    
    if purchased.empty:
        return pd.DataFrame(columns=['Score', 'Description'])

    # Vectorized dot product to calculate raw item recommendation scores
    scores = product_similarity_df.loc[purchased.index].T.dot(purchased)
    
    # Normalize scores to prevent popular items from dominating recommendations
    if normalize:
        norm = product_similarity_df.loc[purchased.index].sum(axis=0).replace(0, np.nan)
        scores = scores / norm

    # Drop items the user has already bought, sort, and slice top N
    scores = scores.drop(index=purchased.index, errors='ignore').sort_values(ascending=False).head(top_n)
    
    # Format output DataFrame
    recommendations = scores.rename('Score').to_frame()
    recommendations['Description'] = recommendations.index.map(product_description)
    return recommendations

def top_similar_items(stock_code, top_n=10):
    stock_code = str(stock_code)
    if stock_code not in product_similarity_df.index:
        raise ValueError(f"StockCode {stock_code} not found in product_similarity_df")
        
    similar = product_similarity_df[stock_code].drop(index=stock_code).sort_values(ascending=False).head(top_n)
    
    df = similar.rename('Similarity').to_frame()
    df['Description'] = df.index.map(product_description)
    return df

# Generate recommendations
sample_customer = top_customers.index[0]
# Create a stock-code -> description lookup for recommender functions
product_description = (
    valid_transactions[['StockCode', 'Description']]
    .drop_duplicates(subset=['StockCode'])
    .assign(StockCode=lambda x: x['StockCode'].astype(str))
    .set_index('StockCode')['Description']
)

def item_based_recommendations(customer_id, top_n=10, normalize=True):
    if customer_id not in user_item_matrix.index:
        raise ValueError(f"CustomerID {customer_id} not found in user_item_matrix")

    customer_profile = user_item_matrix.loc[customer_id]
    purchased = customer_profile[customer_profile > 0].copy()
    purchased.index = purchased.index.astype(str)

    if purchased.empty:
        return pd.DataFrame(columns=['Score', 'Description'])

    # Vectorized dot product to calculate raw item recommendation scores
    scores = product_similarity_df.loc[purchased.index].T.dot(purchased)

    # Normalize scores to prevent popular items from dominating recommendations
    if normalize:
        norm = product_similarity_df.loc[purchased.index].sum(axis=0).replace(0, np.nan)
        scores = scores / norm

    # Drop items the user has already bought, sort, and slice top N
    scores = scores.drop(index=purchased.index, errors='ignore').sort_values(ascending=False).head(top_n)

    # Format output DataFrame
    recommendations = scores.rename('Score').to_frame()
    recommendations['Description'] = recommendations.index.astype(str).map(product_description)
    return recommendations

def top_similar_items(stock_code, top_n=10):
    stock_code = str(stock_code)
    if stock_code not in product_similarity_df.index:
        raise ValueError(f"StockCode {stock_code} not found in product_similarity_df")

    similar = product_similarity_df[stock_code].drop(index=stock_code).sort_values(ascending=False).head(top_n)

    df = similar.rename('Similarity').to_frame()
    df['Description'] = df.index.astype(str).map(product_description)
    return df

# Generate recommendations
sample_customer = top_customers.index[0]
print(f"Top recommendations for customer {sample_customer}:")
print(item_based_recommendations(sample_customer, top_n=10))
print(item_based_recommendations(sample_customer, top_n=10))


# %%
# Build CustomerID x Description matrix and compute product-product cosine similarity
# Uses: valid_transactions, top_products, cosine_similarity, pd, np

# pivot to customer x product (Description) matrix (aggregate by quantity)
cust_desc = valid_transactions.pivot_table(
    index='CustomerID',
    columns='Description',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Optional: binarize to capture co-purchase patterns rather than volume
cust_desc_bin = (cust_desc > 0).astype(int)

# compute cosine similarity between products (columns)
prod_sim = cosine_similarity(cust_desc_bin.T)  # shape: n_products x n_products
product_similarity_desc_df = pd.DataFrame(prod_sim, index=cust_desc_bin.columns, columns=cust_desc_bin.columns)

# show similarity for top 10 products by quantity (if available)
top_desc = top_products.index if 'top_products' in globals() else product_similarity_desc_df.sum().nlargest(10).index
similarity_subset_desc = product_similarity_desc_df.loc[top_desc, top_desc]

print("Computed product-product cosine similarity (based on CustomerID–Description). Top-10 subset:")
st.dataframe(similarity_subset_desc)

# helper to get top-N similar products for a given description
def top_similar_descriptions(description, top_n=10):
    if description not in product_similarity_desc_df.index:
        raise ValueError(f"Description '{description}' not found in product similarity matrix")
    sims = product_similarity_desc_df[description].drop(index=description).sort_values(ascending=False).head(top_n)
    return sims.to_frame(name='CosineSimilarity')

# %%
from difflib import get_close_matches

def top_5_similar_products(description, top_n=5):
    """
    Return top N similar product descriptions to the given product description
    using precomputed product_similarity_desc_df (cosine similarity on Description).
    """
    description = description.strip()
    if 'product_similarity_desc_df' not in globals():
        raise RuntimeError("product_similarity_desc_df not found in the notebook environment.")
    sims_df = product_similarity_desc_df
    if description in sims_df.index:
        sims = sims_df[description].drop(index=description).sort_values(ascending=False).head(top_n)
        return sims.rename('Similarity').to_frame().reset_index().rename(columns={'index':'Description'})
    # try fuzzy match suggestions
    candidates = get_close_matches(description, sims_df.index, n=3, cutoff=0.6)
    if candidates:
        suggestion = candidates[0]
        raise ValueError(f"Description '{description}' not found. Did you mean: '{suggestion}'? "
                         f"Use that exact string or check available descriptions.")
    raise ValueError(f"Description '{description}' not found and no close matches available.")

# Example usage:
print(top_5_similar_products("WHITE HANGING HEART T-LIGHT HOLDER", top_n=5))

# %% [markdown]
# ### AT LAST 

# %%
import sys, subprocess
try:
    import ipywidgets as widgets
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ipywidgets"])
    import ipywidgets as widgets

from IPython.display import display, HTML
from sklearn.cluster import DBSCAN
from IPython.display import display, HTML

# Create input widgets for RFM values
recency_input = widgets.IntSlider(
    value=50,
    min=0,
    max=int(rfm['Recency'].max()),
    step=1,
    description='Recency (days):',
    style={'description_width': '150px'}
)

frequency_input = widgets.IntSlider(
    value=5,
    min=1,
    max=int(rfm['Frequency'].max()),
    step=1,
    description='Frequency (transactions):',
    style={'description_width': '150px'}
)

monetary_input = widgets.FloatSlider(
    value=1000,
    min=0,
    max=rfm['Monetary'].max(),
    step=100,
    description='Monetary ($):',
    style={'description_width': '150px'}
)

predict_button = widgets.Button(
    description='Predict Cluster',
    button_style='info',
    tooltip='Click to predict customer segment'
)

output = widgets.Output()

def on_predict_click(b):
    output.clear_output()
    
    with output:
        # Prepare input data
        input_data = pd.DataFrame({
            'Recency': [recency_input.value],
            'Frequency': [frequency_input.value],
            'Monetary': [monetary_input.value]
        })
        
        # Apply log transformation
        input_log = input_data.copy()
        input_log['Recency_log'] = np.log1p(input_log['Recency'])
        input_log['Frequency_log'] = np.log1p(input_log['Frequency'])
        input_log['Monetary_log'] = np.log1p(input_log['Monetary'])
        
        # Scale using the existing scaler
        input_scaled = scaler_rfm.transform(input_log[['Recency_log', 'Frequency_log', 'Monetary_log']])
        
        # Predict cluster using best model
        best_idx = res_df['silhouette'].idxmax()
        best_algo, best_params, _, best_labels = results[best_idx]
        
        if best_algo == 'KMeans':
            predicted_cluster = models[best_params['n_clusters']].predict(input_scaled)[0]
        elif best_algo == 'Agglomerative':
            predicted_cluster = AgglomerativeClustering(
                n_clusters=best_params['n_clusters'],
                linkage=best_params['linkage']
            ).fit_predict(input_scaled)[0]
        elif best_algo == 'GaussianMixture':
            gm = GaussianMixture(n_components=best_params['n_components'], random_state=42)
            gm.fit(rfm_scaled)
            predicted_cluster = gm.predict(input_scaled)[0]
        
        # Get cluster label
        cluster_label = cluster_labels.get(predicted_cluster, f"Cluster {predicted_cluster}")
        
        # Display results
        display(HTML(f"<h3>Prediction Results</h3>"))
        display(HTML(f"<p><b>Predicted Cluster:</b> {predicted_cluster}</p>"))
        display(HTML(f"<p><b>Segment Label:</b> <span style='color: green; font-size: 18px;'>{cluster_label}</span></p>"))
        
        # Show cluster statistics
        cluster_stats = rfm[rfm['Cluster_Final'] == predicted_cluster]
        display(HTML(f"<h4>Cluster {predicted_cluster} Statistics:</h4>"))
        display(HTML(f"<p>Number of Customers: {len(cluster_stats)}</p>"))
        display(HTML(f"<p>Avg Recency: {cluster_stats['Recency'].mean():.1f} days</p>"))
        display(HTML(f"<p>Avg Frequency: {cluster_stats['Frequency'].mean():.2f} transactions</p>"))
        display(HTML(f"<p>Avg Monetary: ${cluster_stats['Monetary'].mean():.2f}</p>"))

predict_button.on_click(on_predict_click)

# Display the interface
display(HTML("<h2>🎯 Customer Segmentation Prediction Module</h2>"))
display(recency_input)
display(frequency_input)
display(monetary_input)
display(predict_button)
display(output)

# %%
import streamlit as st
import pandas as pd 

st.title("Customer Segmentation Prediction Module")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from difflib import get_close_matches
import datetime as dt

# Set page layout to wide for professional aesthetic
st.set_page_config(page_title="E-Commerce Customer Analytics Dashboard", layout="wide")

# ==========================================
# 1. CACHED DATA PROCESSING PIPELINE (ETL)
# ==========================================
@st.cache_data
def load_and_clean_data(file_path):
    """Loads, cleans, and engineers RFM + Recommendation metrics from transactional data."""
    # Load dataset
    order = pd.read_csv(file_path)
    
    # Data Cleaning
    order_cleaned = order.drop_duplicates()
    order_cleaned['InvoiceNo'] = order_cleaned['InvoiceNo'].astype(str)
    
    # Filter out cancelled transactions & keep positive quantities
    valid_transactions = order_cleaned[
        (~order_cleaned['InvoiceNo'].str.startswith('C', na=False)) & 
        (order_cleaned['Quantity'] > 0)
    ].copy()
    
    # Structural Type Casting
    valid_transactions['InvoiceDate'] = pd.to_datetime(valid_transactions['InvoiceDate'])
    valid_transactions = valid_transactions.dropna(subset=['CustomerID']).copy()
    valid_transactions['CustomerID'] = valid_transactions['CustomerID'].astype(int).astype(str)
    
    # Feature Engineering
    valid_transactions['Revenue'] = valid_transactions['Quantity'] * valid_transactions['UnitPrice']
    
    return valid_transactions

@st.cache_data
def compute_rfm_and_clusters(valid_transactions):
    """Computes RFM features, normalizes them, and trains the optimal clustering model."""
    snapshot_date = valid_transactions['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Compute RFM metrics
    rfm = valid_transactions.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Revenue', 'sum')
    )
    
    # Stabilize variance using Log-Transformations
    rfm_normalized = rfm.copy()
    rfm_normalized['Recency_log'] = np.log1p(rfm_normalized['Recency'])
    rfm_normalized['Frequency_log'] = np.log1p(rfm_normalized['Frequency'])
    rfm_normalized['Monetary_log'] = np.log1p(rfm_normalized['Monetary'])
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_normalized[['Recency_log', 'Frequency_log', 'Monetary_log']])
    
    # Fit the best-performing model from our exploration phase (e.g., Gaussian Mixture Model / KMeans)
    # Using GMM for soft-boundary cluster distributions as evaluated in R&D
    gmm = GaussianMixture(n_components=3, random_state=42)
    rfm['Cluster_Final'] = gmm.fit_predict(X_scaled)
    
    # Define descriptive business segment rules mapped to data properties
    recency_mean = rfm['Recency'].mean()
    frequency_mean = rfm['Frequency'].mean()
    monetary_mean = rfm['Monetary'].mean()
    
    cluster_labels = {}
    for cluster_id in sorted(rfm['Cluster_Final'].unique()):
        cluster_data = rfm[rfm['Cluster_Final'] == cluster_id]
        r_status = "Recent" if cluster_data['Recency'].mean() < recency_mean else "Inactive"
        f_status = "Frequent" if cluster_data['Frequency'].mean() > frequency_mean else "Infrequent"
        m_status = "High-Value" if cluster_data['Monetary'].mean() > monetary_mean else "Low-Value"
        cluster_labels[cluster_id] = f"{r_status} {f_status} {m_status}"
        
    rfm['Cluster_Label'] = rfm['Cluster_Final'].map(cluster_labels)
    
    return rfm, scaler, gmm, cluster_labels

@st.cache_data
def build_recommendation_matrices(valid_transactions):
    """Generates collaborative filtering lookups based on transactional history."""
    # User-Item interaction frame
    user_item_matrix = valid_transactions.pivot_table(
        index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum', fill_value=0
    )
    
    # Description lookups mapping matrix keys back to clean readable text
    product_description = (
        valid_transactions[['StockCode', 'Description']]
        .drop_duplicates(subset=['StockCode'])
        .assign(StockCode=lambda x: x['StockCode'].astype(str))
        .set_index('StockCode')['Description']
    )
    
    # Product Cosine Similarity (Item-Based Collaborative Filtering)
    product_similarity = cosine_similarity(user_item_matrix.T)
    product_similarity_df = pd.DataFrame(
        product_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns
    )
    
    # Pivot matrix targeting clean Item Descriptions for alternate lookup paradigms
    cust_desc = valid_transactions.pivot_table(
        index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0
    )
    cust_desc_bin = (cust_desc > 0).astype(int)
    product_similarity_desc = pd.DataFrame(
        cosine_similarity(cust_desc_bin.T), index=cust_desc_bin.columns, columns=cust_desc_bin.columns
    )
    
    return user_item_matrix, product_similarity_df, product_description, product_similarity_desc

# ==========================================
# 2. STREAMLIT APP APPLICATION INTERFACE
# ==========================================
st.title("🎯 E-Commerce Core Intelligence Engine")
st.caption("Enterprise Analytics Platform: Customer RFM Segmentation & Collaborative Filtering Recommendation Engine")

# File Upload block
uploaded_file = st.sidebar.file_uploader("Upload 'online_retail.csv' to initialize pipeline", type=["csv"])

if uploaded_file is not None:
    # Trigger the processing backend
    with st.spinner("Processing transaction metrics and generating matrices..."):
        df_valid = load_and_clean_data(uploaded_file)
        rfm_df, rfm_scaler, cluster_model, cluster_labels = compute_rfm_and_clusters(df_valid)
        user_item, prod_sim_df, prod_desc, prod_sim_desc = build_recommendation_matrices(df_valid)
    
    st.sidebar.success("Pipeline Online")
    
    # Dashboard Tabs structured for Business Stakeholder reviews
    tab1, tab2, tab3 = st.tabs(["📊 Executive Performance Insights", "👥 Customer Segmentation Portal", "🛍️ Intelligent Recommender System"])
    
    # ------------------------------------------
    # TAB 1: EXECUTIVE INSIGHTS (EDA DASHBOARD)
    # ------------------------------------------
    with tab1:
        st.header("Executive Data Overview")
        
        # High-level KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Cleaned Transactions", f"{df_valid['InvoiceNo'].nunique():,}")
        kpi2.metric("Active Customer Base", f"{df_valid['CustomerID'].nunique():,}")
        kpi3.metric("Gross Platform Revenue", f"${df_valid['Revenue'].sum():,.2f}")
        kpi4.metric("Catalog Size", f"{df_valid['StockCode'].nunique():,}")
        
        st.markdown("---")
        
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            st.subheader("Top 10 Most Active Markets (by Quantity)")
            top_locations = df_valid.groupby('Country')['Quantity'].sum().nlargest(10)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(x=top_locations.values, y=top_locations.index, palette='rocket', ax=ax)
            ax.set_xlabel("Total Units Sold")
            st.pyplot(fig)
            
        with col_graph2:
            st.subheader("Aggregate Transaction Trend Analysis")
            df_valid['InvoiceMonth'] = df_valid['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
            monthly_trend = df_valid.groupby('InvoiceMonth')['Quantity'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.lineplot(data=monthly_trend, x='InvoiceMonth', y='Quantity', marker='o', color='#45B7D1', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # ------------------------------------------
    # TAB 2: CUSTOMER SEGMENTATION PORTAL
    # ------------------------------------------
    with tab2:
        st.header("ML-Powered RFM Cohort Analysis")
        
        # Interactive Inference Section
        st.subheader("🔮 Live Segment Inference Engine")
        st.markdown("Input prospective customer parameters below to classify their behavioral cluster profile instantly.")
        
        inf_col1, inf_col2, inf_col3 = st.columns(3)
        with inf_col1:
            rec_in = st.slider("Recency (Days since last interaction)", 0, int(rfm_df['Recency'].max()), 45)
        with inf_col2:
            freq_in = st.slider("Frequency (Historical Lifetime Order Count)", 1, int(rfm_df['Frequency'].max()), 5)
        with inf_col3:
            mon_in = st.number_input("Monetary Value ($ Net Revenue Value)", min_value=0.0, max_value=float(rfm_df['Monetary'].max()), value=250.0)
            
        # Conduct localized mathematical transformation on input vectors
        input_vector = pd.DataFrame({'Recency': [rec_in], 'Frequency': [freq_in], 'Monetary': [mon_in]})
        input_vector['Recency_log'] = np.log1p(input_vector['Recency'])
        input_vector['Frequency_log'] = np.log1p(input_vector['Frequency'])
        input_vector['Monetary_log'] = np.log1p(input_vector['Monetary'])
        
        scaled_input = rfm_scaler.transform(input_vector[['Recency_log', 'Frequency_log', 'Monetary_log']])
        predicted_id = cluster_model.predict(scaled_input)[0]
        assigned_label = cluster_labels.get(predicted_id, "Unknown Segment")
        
        st.info(f"Target Vector classified under **Cluster ID: {predicted_id}** — **Business Cohort Category: {assigned_label}**")
        
        st.markdown("---")
        st.subheader("Statistical Profile Distributions")
        
        # Comprehensive Cohort Aggregations Display
        profiles = rfm_df.groupby('Cluster_Final').agg(
            Size=('Cluster_Final', 'count'),
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean')
        ).round(2)
        profiles['Cohort_Tag'] = profiles.index.map(cluster_labels)
        st.dataframe(profiles, use_container_width=True)

    # ------------------------------------------
    # TAB 3: INTELLIGENT RECOMMENDER SYSTEM
    # ------------------------------------------
    with tab3:
        st.header("Collaborative Filtering Recommendation Engines")
        
        # Display the Matrix Subset INSIDE the tab for supervisors to review R&D metrics
        st.subheader("Product Similarity Matrix (R&D Baseline)")
        st.markdown("Below is the precomputed product-to-product cosine similarity snapshot utilized by our affinity tracking matrix algorithms:")
        st.dataframe(prod_sim_desc.head(10)) # Renders safely inside the Tab context
        
        st.markdown("---")
        
        rec_type = st.radio("Choose Recommendation Strategy Paradigm:", [
            "User-Targeted Recommendations (Personalized Cross-Sell)", 
            "Item-to-Item Similarity Analysis (Affinity Mapping)"
        ])
        
        if rec_type == "User-Targeted Recommendations (Personalized Cross-Sell)":
            st.subheader("Generate Targeted User Offers")
            user_id_selection = st.selectbox("Select Target Customer Account ID:", options=user_item.index.unique())
            
            if user_id_selection:
                customer_profile = user_item.loc[user_id_selection]
                purchased_items = customer_profile[customer_profile > 0]
                
                st.markdown("**Account Purchase Foundations:**")
                history_list = [f"{code} ({prod_desc.get(code, 'Unknown SKU')})" for code in purchased_items.index]
                st.caption(", ".join(history_list[:12]) + ("..." if len(history_list) > 12 else ""))
                
                # Perform vectorized dot-product scoring
                scores = prod_sim_df.loc[purchased_items.index].T.dot(purchased_items)
                norm = prod_sim_df.loc[purchased_items.index].sum(axis=0).replace(0, np.nan)
                normalized_scores = (scores / norm).drop(index=purchased_items.index, errors='ignore').sort_values(ascending=False).head(5)
                
                rec_frame = normalized_scores.to_frame(name="Algorithmic Match Score")
                rec_frame['Product Description Lookup'] = rec_frame.index.map(prod_desc)
                
                st.markdown("**Top 5 Strategic Recommended SKUs:**")
                st.table(rec_frame)
                
        else:
            st.subheader("Item Contextual Affinity Mapping")
            item_search = st.text_input("Enter Reference Product Description:", value="WHITE HANGING HEART T-LIGHT HOLDER")
            
            if item_search:
                clean_search = item_search.strip()
                if clean_search in prod_sim_desc.index:
                    affinity_scores = prod_sim_desc[clean_search].drop(index=clean_search).sort_values(ascending=False).head(5)
                    st.success(f"Affinity profiles extracted for item matches: '{clean_search}'")
                    st.table(affinity_scores.rename("Cosine Proximity Index").to_frame())
                else:
                    fuzzy_options = get_close_matches(clean_search, prod_sim_desc.index, n=3, cutoff=0.5)
                    st.warning(f"SKU identity exact literal match for '{clean_search}' not resolved within product array mappings.")
                    if fuzzy_options:
                        st.info(f"Did you mean one of these indexed catalog strings? \n\n {fuzzy_options}")
    
    
    