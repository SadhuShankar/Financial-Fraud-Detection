import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ==========================
# Load Data
# ==========================

@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

df = load_data()

# ==========================
# Title
# ==========================

st.title("💳 Financial Fraud Detection Dashboard")
st.markdown("### Fraud Analytics on 284K+ Transactions")

# ==========================
# KPI Cards
# ==========================

total_transactions = len(df)
fraud_transactions = df["Class"].sum()
genuine_transactions = total_transactions - fraud_transactions
fraud_rate = (fraud_transactions / total_transactions) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

with col3:
    st.metric(
        "Genuine Transactions",
        f"{genuine_transactions:,}"
    )

with col4:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.4f}%"
    )

st.divider()

# ==========================
# Dataset Preview
# ==========================

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ==========================
# Fraud Distribution
# ==========================

st.subheader("Fraud vs Genuine Transactions")

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    x="Class",
    data=df,
    ax=ax
)

ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.set_title("Fraud Distribution")

st.pyplot(fig)

# ==========================
# Transaction Amount
# ==========================

st.subheader("Transaction Amount Distribution")

fig2, ax2 = plt.subplots(figsize=(8,4))

sns.histplot(
    df["Amount"],
    bins=50,
    ax=ax2
)

ax2.set_title("Transaction Amount Distribution")

st.pyplot(fig2)

# ==========================
# Fraud Amount Analysis
# ==========================

st.subheader("Fraud Transaction Amounts")

fraud_df = df[df["Class"] == 1]

st.write(
    fraud_df["Amount"].describe()
)

# ==========================
# Sidebar Filters
# ==========================

st.sidebar.header("Filters")

sample_size = st.sidebar.slider(
    "Rows to View",
    5,
    100,
    10
)

st.sidebar.write("Sample Data")

st.sidebar.dataframe(
    df.head(sample_size)
)

# ==========================
# Model Performance
# ==========================

st.subheader("Model Performance")

st.success(
"""
XGBoost + SMOTE Results

• Precision Recall AUC = 0.8614

• Fraud Recall = 88.8%

• Detected 87 of 98 Fraud Cases

• Missed Only 11 Fraud Transactions
"""
)

# ==========================
# Footer
# ==========================

st.markdown("---")

st.markdown(
    "Developed by Shankar Sadhu | Financial Fraud Detection Project"
)