import pandas as pd

# ==============================
# Load Dataset
# ==============================

file_path = "dataset/Bank Customer Churn Prediction.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET INSPECTION ==========\n")

# Dataset shape
print("Dataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# First 5 records
print("\nFirst 5 Records:")
print(df.head())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Target distribution
print("\nChurn Distribution:")
print(df["churn"].value_counts())

# Target percentage
print("\nChurn Percentage:")
print(df["churn"].value_counts(normalize=True) * 100)

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())