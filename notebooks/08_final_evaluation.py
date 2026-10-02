import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance


# ==========================================
# Configuration
# ==========================================

DATA_PATH = "dataset/clean_customer_churn.csv"
MODEL_PATH = "model/churn_model.pkl"
FEATURE_PATH = "model/feature_names.pkl"
IMPORTANCE_PATH = "model/feature_importance.csv"

THRESHOLD = 0.35


print("\n========== FINAL MODEL EVALUATION ==========\n")


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["churn"])
y = df["churn"]


# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# Final HistGradientBoosting Model
# ==========================================

model = HistGradientBoostingClassifier(
    min_samples_leaf=20,
    max_leaf_nodes=15,
    max_iter=400,
    max_depth=10,
    learning_rate=0.03,
    l2_regularization=1.0,
    random_state=42
)


print("Training final model...")

model.fit(X_train, y_train)


# ==========================================
# Prediction Probability
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ==========================================
# Evaluation Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n========== FINAL RESULTS ==========\n")

print(f"Threshold : {THRESHOLD:.2f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ==========================================
# Classification Report
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Stayed",
            "Churned"
        ]
    )
)


# ==========================================
# Confusion Matrix
# ==========================================

print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)

print("\nTN:", cm[0][0])
print("FP:", cm[0][1])
print("FN:", cm[1][0])
print("TP:", cm[1][1])


# ==========================================
# Permutation Feature Importance
# ==========================================

print("\n========== FEATURE IMPORTANCE ==========\n")

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std
})

feature_importance = feature_importance.sort_values(
    by="importance_mean",
    ascending=False
)

print(feature_importance.to_string(index=False))


# ==========================================
# Save Feature Importance
# ==========================================

feature_importance.to_csv(
    IMPORTANCE_PATH,
    index=False
)


# ==========================================
# Save Model
# ==========================================

joblib.dump(
    model,
    MODEL_PATH
)


# ==========================================
# Save Feature Names
# ==========================================

joblib.dump(
    list(X.columns),
    FEATURE_PATH
)


print("\n========== MODEL ARTIFACTS ==========\n")

print("Model saved:")
print(MODEL_PATH)

print("\nFeature names saved:")
print(FEATURE_PATH)

print("\nFeature importance saved:")
print(IMPORTANCE_PATH)

print("\nFinal evaluation completed successfully.")