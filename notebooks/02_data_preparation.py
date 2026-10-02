import pandas as pd

# ==============================
# Load Dataset
# ==============================

file_path = "dataset/Bank Customer Churn Prediction.csv"

df = pd.read_csv(file_path)

print("\n========== DATA PREPARATION ==========\n")

# ==============================
# Remove Customer ID
# ==============================

df = df.drop(columns=["customer_id"])

# ==============================
# Check categorical values
# ==============================

print("Countries:")
print(df["country"].value_counts())

print("\nGender:")
print(df["gender"].value_counts())

# ==============================
# Convert Categorical Features
# ==============================

df = pd.get_dummies(
    df,
    columns=["country", "gender"],
    drop_first=True,
    dtype=int
)

# ==============================
# Separate Features & Target
# ==============================

X = df.drop(columns=["churn"])
y = df["churn"]

# ==============================
# Display Results
# ==============================

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())

print("\nPrepared Dataset:")
print(df.head())

# ==============================
# Save Prepared Dataset
# ==============================

df.to_csv(
    "dataset/clean_customer_churn.csv",
    index=False
)

print("\nClean dataset saved successfully.")