import pandas as pd
import joblib


# ==========================================
# Load Model
# ==========================================

MODEL_PATH = "model/churn_model.pkl"
FEATURE_PATH = "model/feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)


# ==========================================
# Customer Input
# ==========================================

customer = {
    "credit_score": 650,
    "age": 45,
    "tenure": 5,
    "balance": 80000,
    "products_number": 2,
    "credit_card": 1,
    "active_member": 0,
    "estimated_salary": 75000,
    "country_Germany": 1,
    "country_Spain": 0,
    "gender_Male": 1
}


# ==========================================
# Create DataFrame
# ==========================================

input_data = pd.DataFrame(
    [customer],
    columns=feature_names
)


# ==========================================
# Prediction
# ==========================================

probability = model.predict_proba(
    input_data
)[0][1]


threshold = 0.35

prediction = int(
    probability >= threshold
)


# ==========================================
# Result
# ==========================================

print("\n========== CUSTOMER CHURN PREDICTION ==========\n")

print(f"Churn Probability : {probability * 100:.2f}%")

if prediction == 1:
    print("Prediction         : HIGH CHURN RISK")
else:
    print("Prediction         : LOW CHURN RISK")

print("\nPrediction completed successfully.")