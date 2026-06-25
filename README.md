Shopper Spectrum – Customer Segmentation & Recommendation System
A machine learning project that analyzes customer behavior and recommends products using unsupervised learning, NLP, and collaborative filtering — all wrapped in an interactive Streamlit app.

📊 Project Overview
This project solves two key business problems for e-commerce:

Customer Segmentation using RFM + KMeans clustering
Product Recommendation using:
Content-Based Filtering (TF-IDF + Cosine Similarity)
User-Based Collaborative Filtering
🧠 Models Used
Model	Purpose	Algorithm
Model 1	Segment customers	KMeans Clustering
Model 2	Recommend similar products	TF-IDF + Cosine Similarity
Model 3	Personalized recommendations	User-Based Collaborative Filtering
📁 Dataset
Due to GitHub file size restrictions, please download the dataset from Dropbox:

👉 Download online_retail.csv
💾 Required Model Files
Please download the following .joblib files and place them in the root project folder before running app.py:

🔗 user_item_matrix.joblib
🔗 user_similarity_matrix.joblib
🚀 How to Run This Project
Clone this repository
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py
📊 Project Highlights
✅ Customer Segmentation using RFM + KMeans
✅ Product Recommendation via Content & Collaborative Filtering
✅ Final deployment via interactive Streamlit interface
