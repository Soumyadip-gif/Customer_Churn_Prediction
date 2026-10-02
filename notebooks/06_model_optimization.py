import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/clean_customer_churn.csv")

print("\n========== MODEL OPTIMIZATION ==========\n")

# ==========================================
# Features & Target
# ==========================================

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
# Random Forest Optimization
# ==========================================

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

rf_params = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=20,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Optimizing Random Forest...")

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("\nBest Random Forest Parameters:")
print(rf_search.best_params_)

# ==========================================
# HistGradientBoosting Optimization
# ==========================================

hgb = HistGradientBoostingClassifier(
    random_state=42
)

hgb_params = {
    "max_iter": [100, 200, 300, 400],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [10, 20, 30],
    "l2_regularization": [0, 0.1, 1.0]
}

hgb_search = RandomizedSearchCV(
    estimator=hgb,
    param_distributions=hgb_params,
    n_iter=20,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nOptimizing HistGradientBoosting...")

hgb_search.fit(X_train, y_train)

best_hgb = hgb_search.best_estimator_

print("\nBest HistGradientBoosting Parameters:")
print(hgb_search.best_params_)

# ==========================================
# Evaluation Function
# ==========================================

def evaluate_model(name, model):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"\n========== {name} ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ==========================================
# Evaluate Optimized Models
# ==========================================

rf_result = evaluate_model(
    "Optimized Random Forest",
    best_rf
)

hgb_result = evaluate_model(
    "Optimized HistGradientBoosting",
    best_hgb
)

# ==========================================
# Save Results
# ==========================================

results = pd.DataFrame([
    rf_result,
    hgb_result
])

results.to_csv(
    "model/optimized_model_comparison.csv",
    index=False
)

print("\nOptimized model results saved successfully.")