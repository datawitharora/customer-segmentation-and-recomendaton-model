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
    