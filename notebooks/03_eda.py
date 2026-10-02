import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Load Clean Dataset
# ==============================

df = pd.read_csv("dataset/clean_customer_churn.csv")

print("\n========== EXPLORATORY DATA ANALYSIS ==========\n")

# ==============================
# Churn Distribution
# ==============================

print("Churn Distribution:")
print(df["churn"].value_counts())

plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# ==============================
# Churn by Age
# ==============================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="churn", y="age")
plt.title("Age vs Churn")
plt.xlabel("Churn")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# ==============================
# Churn by Credit Score
# ==============================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="churn", y="credit_score")
plt.title("Credit Score vs Churn")
plt.xlabel("Churn")
plt.ylabel("Credit Score")
plt.tight_layout()
plt.show()

# ==============================
# Churn by Balance
# ==============================

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="churn", y="balance")
plt.title("Balance vs Churn")
plt.xlabel("Churn")
plt.ylabel("Balance")
plt.tight_layout()
plt.show()

# ==============================
# Churn by Number of Products
# ==============================

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="products_number", hue="churn")
plt.title("Products Number vs Churn")
plt.xlabel("Number of Products")
plt.ylabel("Customers")
plt.tight_layout()
plt.show()

# ==============================
# Churn by Active Member
# ==============================

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="active_member", hue="churn")
plt.title("Active Member vs Churn")
plt.xlabel("Active Member")
plt.ylabel("Customers")
plt.tight_layout()
plt.show()

# ==============================
# Correlation Matrix
# ==============================

plt.figure(figsize=(12, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

print("\nEDA completed successfully.")