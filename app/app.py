import streamlit as st
import pandas as pd
import pickle
import shap
import plotly.express as px
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.main{
    background:#0E1117;
}

section[data-testid="stSidebar"]{
    background:#161A28;
}

div[data-testid="metric-container"]{
    background:#1E1E2F;
    border:1px solid #4CAF50;
    padding:15px;
    border-radius:12px;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    background:#4CAF50;
    color:white;
    font-weight:bold;
}

h1,h2,h3{
    color:#4CAF50;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# KPI CARD
# ============================================

def kpi_card(title, value, color):

    st.markdown(
        f"""
        <div style="
        background:{color};
        padding:18px;
        border-radius:12px;
        color:white;
        text-align:center;
        box-shadow:0px 4px 12px rgba(0,0,0,.3);
        ">
        <h4>{title}</h4>
        <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_data():

    return pd.read_csv("data/creditcard.csv")

df = load_data()

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    with open("models/fraud_model.pkl","rb") as file:

        model = pickle.load(file)

    return model

model = load_model()

# ============================================
# SHAP
# ============================================

explainer = shap.TreeExplainer(model)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("💳 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dataset Analysis",
        "🤖 Predict Fraud",
        "🧠 SHAP Explainability",
        "📈 Model Performance",
        "ℹ️ About"
    ]
)

# ============================================
# HOME PAGE
# ============================================

if page == "🏠 Home":

    st.markdown("""
# 💳 AI Powered Financial Fraud Detection

### 🚀 Detect fraudulent credit card transactions using Machine Learning + Explainable AI (SHAP)

✔ **284,807 Transactions**

✔ **XGBoost Classifier**

✔ **SHAP Explainability**

✔ **Interactive Streamlit Dashboard**
""")

    total_transactions = len(df)

    fraud_transactions = int(df["Class"].sum())

    genuine_transactions = total_transactions - fraud_transactions

    fraud_rate = fraud_transactions / total_transactions * 100

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        kpi_card(
            "💳 Total Transactions",
            f"{total_transactions:,}",
            "#2563eb"
        )

    with c2:
        kpi_card(
            "🚨 Fraud Transactions",
            fraud_transactions,
            "#dc2626"
        )

    with c3:
        kpi_card(
            "✅ Genuine",
            genuine_transactions,
            "#16a34a"
        )

    with c4:
        kpi_card(
            "📈 Fraud Rate",
            f"{fraud_rate:.4f}%",
            "#7c3aed"
        )

    st.divider()

    st.markdown("## 📊 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.success("""
✔ XGBoost Classifier

✔ SMOTE

✔ SHAP Explainability

✔ Streamlit Dashboard
""")
    
# ============================================
# DATASET ANALYSIS
# ============================================

elif page == "📊 Dataset Analysis":

    st.title("📊 Dataset Analysis")

    st.markdown(
        """
        Explore the credit card fraud dataset with
        interactive visualizations.
        """
    )

    st.divider()

    # Dataset Shape
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", df.shape[1])

    st.divider()

    # Dataset Preview
    st.markdown("## 📄 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # Fraud Distribution
    st.markdown("## 📈 Fraud Distribution")

    fraud_counts = (
        df["Class"]
        .replace({0: "Genuine", 1: "Fraud"})
        .value_counts()
        .reset_index()
    )

    fraud_counts.columns = ["Transaction", "Count"]

    fig = px.bar(
        fraud_counts,
        x="Transaction",
        y="Count",
        color="Transaction",
        text="Count",
        title="Fraud vs Genuine Transactions"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # Transaction Amount Distribution
    st.markdown("## 💰 Transaction Amount Distribution")

    fig2 = px.histogram(
        df,
        x="Amount",
        nbins=60,
        title="Distribution of Transaction Amount",
    )

    fig2.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    # Fraud Statistics

    st.markdown("## 🚨 Fraud Transaction Statistics")

    fraud_df = df[df["Class"] == 1]

    st.dataframe(
        fraud_df["Amount"].describe().to_frame(),
        use_container_width=True
    )

    st.divider()

    st.markdown("## 🔍 Missing Values")

    missing = df.isnull().sum()

    st.dataframe(
        missing.to_frame("Missing Values"),
        use_container_width=True
    )

    st.success("✅ Dataset analysis completed successfully.")    
# ============================================
# PREDICT FRAUD
# ============================================

elif page == "🤖 Predict Fraud":

    st.title("🤖 Fraud Prediction")

    st.markdown("""
    Select a real transaction from the dataset and let the trained
    **XGBoost** model predict whether it is **Fraud** or **Genuine**.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button("🟢 Load Genuine Transaction"):

            sample = df[df["Class"] == 0].sample(
                1,
                random_state=42
            )

            st.session_state["sample"] = sample

    with col2:

        if st.button("🔴 Load Fraud Transaction"):

            sample = df[df["Class"] == 1].sample(
                1,
                random_state=42
            )

            st.session_state["sample"] = sample

    # =======================================
    # Prediction
    # =======================================

    if "sample" in st.session_state:

        sample = st.session_state["sample"]

        st.markdown("## 📄 Selected Transaction")

        st.dataframe(
            sample,
            use_container_width=True
        )

        X_sample = sample.drop("Class", axis=1)

        prediction = model.predict(X_sample)[0]

        probability = model.predict_proba(X_sample)[0]

        fraud_prob = probability[1] * 100
        genuine_prob = probability[0] * 100

        st.divider()

        st.markdown("## 🤖 Prediction")

        if prediction == 1:

            st.error("🚨 Fraudulent Transaction")

        else:

            st.success("✅ Genuine Transaction")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Fraud Probability",
                f"{fraud_prob:.2f}%"
            )

        with c2:

            st.metric(
                "Genuine Probability",
                f"{genuine_prob:.2f}%"
            )

        confidence = float(max(probability))
        st.progress(confidence)
        st.info(f"Prediction Confidence : {confidence*100:.2f}%")

        st.info(
            f"Prediction Confidence : {confidence:.2f}%"
        )

        st.divider()

        st.markdown("## 📊 Transaction Summary")

        summary = pd.DataFrame({

            "Feature": X_sample.columns,

            "Value": X_sample.iloc[0].values

        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        st.success(
            "✅ Transaction saved successfully. You can now open the 🧠 SHAP Explainability page."
        )    
# ============================================
# SHAP EXPLAINABILITY
# ============================================

elif page == "🧠 SHAP Explainability":

    st.title("🧠 SHAP Explainability")

    st.markdown("""
    Understand **why** the model predicted a transaction as
    **Fraud** or **Genuine** using SHAP (Explainable AI).
    """)

    st.divider()

    # Check if a sample exists
    if "sample" not in st.session_state:

        st.warning(
            "⚠️ Please visit the **🤖 Predict Fraud** page first and load a transaction."
        )

    else:

        sample = st.session_state["sample"]

        X_sample = sample.drop("Class", axis=1)

        # Generate SHAP explanation
        explanation = explainer(X_sample)

        st.success("✅ SHAP explanation generated successfully!")

        # -----------------------------
        # Selected Transaction
        # -----------------------------

        st.markdown("## 📄 Selected Transaction")

        st.dataframe(
            X_sample,
            use_container_width=True
        )

        st.divider()

        # -----------------------------
        # Top Influential Features
        # -----------------------------

        shap_values = explanation.values[0]

        importance = pd.DataFrame({

            "Feature": X_sample.columns,

            "SHAP Value": shap_values

        })

        importance["Importance"] = importance["SHAP Value"].abs()

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        st.markdown("## 🧠 Top 10 Influential Features")

        st.dataframe(
            importance[["Feature","SHAP Value"]].head(10),
            use_container_width=True
        )

        st.divider()

        # -----------------------------
        # Waterfall Plot
        # -----------------------------

        st.markdown("## 🌊 SHAP Waterfall Plot")

        fig = plt.figure(figsize=(10,6))

        shap.plots.waterfall(
            explanation[0],
            max_display=10,
            show=False
        )

        st.pyplot(fig)

        plt.close()

        st.divider()

        # -----------------------------
        # Human Explanation
        # -----------------------------

        st.markdown("## 💡 Prediction Explanation")

        top_feature = importance.iloc[0]["Feature"]

        top_value = importance.iloc[0]["SHAP Value"]

        if top_value > 0:

            direction = "Fraud"

        else:

            direction = "Genuine"

        st.info(
            f"""
### Model Interpretation

The prediction was mainly influenced by **{top_feature}**.

• Positive SHAP values push predictions towards **Fraud**.

• Negative SHAP values push predictions towards **Genuine**.

• The strongest contributing feature in this prediction is **{top_feature}**.

Overall, the SHAP explanation indicates that this transaction was pushed towards **{direction}**.
"""
        )

        st.divider()

        st.markdown("## 📊 Complete SHAP Values")

        st.dataframe(
            importance,
            use_container_width=True
        ) 
# ============================================
# MODEL PERFORMANCE
# ============================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance Dashboard")

    st.markdown("""
    Performance metrics of the trained **XGBoost** model after applying **SMOTE**.
    """)

    st.divider()

    # ===========================
    # KPI Cards
    # ===========================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Precision-Recall AUC",
            "0.8614"
        )

    with c2:
        st.metric(
            "Fraud Recall",
            "88.8%"
        )

    with c3:
        st.metric(
            "Detected Frauds",
            "87 / 98"
        )

    with c4:
        st.metric(
            "Missed Frauds",
            "11"
        )

    st.divider()

    # ===========================
    # Model Information
    # ===========================

    st.markdown("## 🤖 Model Information")

    model_info = pd.DataFrame({

        "Property":[

            "Algorithm",

            "Dataset",

            "Training Samples",

            "Testing Samples",

            "Fraud Rate",

            "Explainability"

        ],

        "Value":[

            "XGBoost",

            "Credit Card Fraud Detection",

            "227,845",

            "56,962",

            "0.1727%",

            "SHAP"

        ]

    })

    st.dataframe(
        model_info,
        use_container_width=True
    )

    st.divider()

    # ===========================
    # Performance Summary
    # ===========================

    st.markdown("## 📊 Model Summary")

    st.success("""

✅ XGBoost Classifier

✅ SMOTE for Imbalanced Data

✅ Precision Recall AUC : 0.8614

✅ Fraud Recall : 88.8%

✅ Explainable AI using SHAP

✅ Interactive Streamlit Dashboard

""")

    st.divider()

    # ===========================
    # Confusion Matrix
    # ===========================

    st.markdown("## 🔥 Confusion Matrix")

    cm = pd.DataFrame(

        [[56786,78],

         [11,87]],

        columns=["Predicted Genuine","Predicted Fraud"],

        index=["Actual Genuine","Actual Fraud"]

    )

    st.dataframe(
        cm,
        use_container_width=True
    )

    st.divider()

    # ===========================
    # Final Evaluation
    # ===========================

    st.markdown("## 🏆 Final Evaluation")

    st.info("""

The model successfully detects fraudulent transactions with a high recall while maintaining a strong Precision-Recall AUC.

SHAP improves transparency by explaining individual predictions, making the model more suitable for real-world financial applications.

""")            
# ============================================
# ABOUT PAGE
# ============================================

elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
# 💳 Financial Fraud Detection using Explainable AI (SHAP)

This project uses **Machine Learning** and **Explainable AI (SHAP)** to detect fraudulent credit card transactions and explain every prediction in an interpretable way.

The model is trained using **XGBoost** with **SMOTE** to handle class imbalance and deployed using **Streamlit** for an interactive user experience.
""")

    st.divider()

    # ============================================
    # Technologies Used
    # ============================================

    st.markdown("## 🛠️ Technologies Used")

    tech = pd.DataFrame({

        "Category":[

            "Programming",

            "Data Processing",

            "Machine Learning",

            "Explainable AI",

            "Visualization",

            "Deployment"

        ],

        "Technology":[

            "Python",

            "Pandas, NumPy",

            "Scikit-learn, XGBoost",

            "SHAP",

            "Matplotlib, Plotly",

            "Streamlit"

        ]

    })

    st.dataframe(
        tech,
        use_container_width=True
    )

    st.divider()

    # ============================================
    # Dataset Information
    # ============================================

    st.markdown("## 📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", "284,807")

    with col2:
        st.metric("Fraud Transactions", "492")

    with col3:
        st.metric("Fraud Rate", "0.1727%")

    st.divider()

    # ============================================
    # Project Workflow
    # ============================================

    st.markdown("## 🏗️ Project Workflow")

    st.code("""
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Train-Test Split
    │
    ▼
SMOTE
    │
    ▼
XGBoost Model
    │
    ▼
Fraud Prediction
    │
    ▼
SHAP Explainability
    │
    ▼
Streamlit Dashboard
""")

    st.divider()

    # ============================================
    # Project Features
    # ============================================

    st.markdown("## 🚀 Features")

    st.success("""
✅ Interactive Dashboard

✅ Dataset Analysis

✅ Fraud Prediction

✅ SHAP Explainability

✅ Waterfall Plot

✅ Model Performance Dashboard

✅ Explainable AI

✅ XGBoost Classifier
""")

    st.divider()

    # ============================================
    # Developer
    # ============================================

    st.markdown("## 👨‍💻 Developer")

    st.info("""
**Shankar Sadhu**

🎓 B.Tech – Electronics & Communication Engineering

🏫 National Institute of Technology Andhra Pradesh

💻 Aspiring Data Scientist | Data Analyst | AI Engineer
""")

    st.divider()

    # ============================================
    # Footer
    # ============================================

    st.markdown("---")

    st.markdown(
        """
<div style='text-align:center;'>

### 💳 Financial Fraud Detection using Explainable AI

👨‍💻 Developed by **Shankar Sadhu**

🎓 National Institute of Technology Andhra Pradesh

💻 Python • XGBoost • SHAP • Streamlit

© 2026 All Rights Reserved

</div>
""",
        unsafe_allow_html=True
    )      