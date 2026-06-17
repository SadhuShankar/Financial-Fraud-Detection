import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

from xgboost import (
    XGBClassifier,
    plot_importance
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/creditcard.csv")

# ==========================
# Dataset Information
# ==========================

print("Dataset Shape:")
print(df.shape)

print("\nFraud Distribution:")
print(df["Class"].value_counts())

fraud_rate = df["Class"].mean() * 100

print(f"\nFraud Rate: {fraud_rate:.4f}%")

# ==========================
# Features and Target
# ==========================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================
# Logistic Regression Baseline
# ==========================

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("\nLogistic Regression Results")
print("=" * 50)

print(classification_report(
    y_test,
    y_pred_lr
))

# ==========================
# Apply SMOTE
# ==========================

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE")
print("=" * 50)

print(y_train_smote.value_counts())

# ==========================
# XGBoost Model
# ==========================

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train_smote,
    y_train_smote
)

# ==========================
# Predictions
# ==========================

y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Results")
print("=" * 50)

print(classification_report(
    y_test,
    y_pred_xgb
))

# ==========================
# Precision Recall AUC
# ==========================

y_prob = xgb_model.predict_proba(X_test)[:, 1]

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\nPrecision Recall AUC")
print("=" * 50)

print(pr_auc)

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred_xgb
)

print("\nConfusion Matrix")
print("=" * 50)

print(cm)

# ==========================
# Feature Importance Plot
# ==========================

plt.figure(figsize=(10, 8))

plot_importance(
    xgb_model,
    max_num_features=10
)

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.show()

# ==========================
# Save Model
# ==========================

with open("models/fraud_model.pkl", "wb") as file:
    pickle.dump(xgb_model, file)

print("\nModel Saved Successfully")
