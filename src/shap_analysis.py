import pickle
import shap
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================================
# Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Load Model
# ==========================================
with open("models/fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

print("✅ Model Loaded Successfully")

# ==========================================
# SHAP Explainer
# ==========================================
explainer = shap.TreeExplainer(model)

# ==========================================
# SHAP Values
# ==========================================
shap_values = explainer.shap_values(X_test)

print("✅ SHAP Values Generated")

# ==========================================
# Force Plot
# ==========================================
shap.initjs()

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)

shap.save_html(
    "reports/force_plot.html",
    force_plot
)

print("✅ Force Plot saved successfully!")