import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/clean_customer_churn.csv")

print("\n========== THRESHOLD ANALYSIS ==========\n")

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
# Optimized Random Forest
# ==========================================

rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features="log2",
    max_depth=20,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

rf_probabilities = rf.predict_proba(X_test)[:, 1]

# ==========================================
# Optimized HistGradientBoosting
# ==========================================

hgb = HistGradientBoostingClassifier(
    min_samples_leaf=20,
    max_leaf_nodes=15,
    max_iter=400,
    max_depth=10,
    learning_rate=0.03,
    l2_regularization=1.0,
    random_state=42
)

hgb.fit(X_train, y_train)

hgb_probabilities = hgb.predict_proba(X_test)[:, 1]


# ==========================================
# Threshold Evaluation Function
# ==========================================

def analyze_thresholds(model_name, probabilities):

    results = []

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70
    ]

    print(f"\n========== {model_name} ==========\n")

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(
            f"Threshold: {threshold:.2f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

    return results


# ==========================================
# Analyze Both Models
# ==========================================

rf_results = analyze_thresholds(
    "Optimized Random Forest",
    rf_probabilities
)

hgb_results = analyze_thresholds(
    "Optimized HistGradientBoosting",
    hgb_probabilities
)

# ==========================================
# Save Results
# ==========================================

all_results = pd.DataFrame(
    rf_results + hgb_results
)

all_results.to_csv(
    "model/threshold_analysis.csv",
    index=False
)

print("\nThreshold analysis saved successfully.")