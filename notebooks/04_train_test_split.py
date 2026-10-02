import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================
# Load Clean Dataset
# ==============================

df = pd.read_csv("dataset/clean_customer_churn.csv")

print("\n========== TRAIN / TEST SPLIT ==========\n")

# ==============================
# Separate Features & Target
# ==============================

X = df.drop(columns=["churn"])
y = df["churn"]

# ==============================
# Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==============================
# Display Shapes
# ==============================

print("Original Dataset:")
print("X:", X.shape)
print("y:", y.shape)

print("\nTraining Dataset:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Dataset:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ==============================
# Check Churn Distribution
# ==============================

print("\nTraining Churn Distribution:")
print(y_train.value_counts())

print("\nTesting Churn Distribution:")
print(y_test.value_counts())

print("\nTrain/Test split completed successfully.")