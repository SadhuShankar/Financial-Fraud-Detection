# Financial Fraud Detection

## Overview

This project detects fraudulent credit card transactions using Machine Learning techniques.

## Dataset

* Total Transactions: 284,807
* Fraud Transactions: 492
* Fraud Rate: 0.1727%

## Technologies Used

* Python
* Pandas
* Scikit-learn
* SMOTE
* XGBoost
* Streamlit

## Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Train-Test Split
4. Logistic Regression Baseline
5. SMOTE Oversampling
6. XGBoost Training
7. Model Evaluation
8. Streamlit Dashboard

## Results

* Precision Recall AUC: 0.8614
* Fraud Recall: 88.8%
* Detected 87 out of 98 fraud cases

## Run Project

```bash
python src/fraud_detection.py
```

```bash
python -m streamlit run app/app.py
```
