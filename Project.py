import pandas as pd

print("Step 1: Loading dataset...")

df = pd.read_excel("churn.xlsx")

print("Dataset loaded successfully\n")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:\n", df.columns)
print("\n--- Step 2: Data Cleaning ---")

# 1. Drop ID column
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# 2. Fix TotalCharges (very important for this dataset)
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Check missing values
print("\nMissing values BEFORE cleaning:\n", df.isnull().sum())

# 4. Remove missing rows
df.dropna(inplace=True)

# 5. Verify
print("\nMissing values AFTER cleaning:\n", df.isnull().sum())
print("\nCleaned Shape:", df.shape)
print("\n--- Step 3: Exploratory Data Analysis ---")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 1. Churn Distribution
plt.figure()
sns.countplot(x=df['Churn'])
plt.title("Customer Churn Distribution")
plt.savefig("outputs/churn_distribution.png")
plt.show()

# 2. Monthly Charges vs Churn
plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("outputs/monthly_charges_vs_churn.png")
plt.show()

# 3. Contract Type vs Churn (important business insight)
plt.figure()
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=30)
plt.savefig("outputs/contract_vs_churn.png")
plt.show()

print("\nInsights:")
print("- Customers with higher monthly charges tend to churn more.")
print("- Month-to-month contract customers have the highest churn rate.")
print("\n--- Step 4: Feature Engineering & Modeling ---")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Convert categorical variables → numeric
df_model = pd.get_dummies(df, drop_first=True)

# 2. Define features and target
X = df_model.drop('Churn_Yes', axis=1)
y = df_model['Churn_Yes']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Model 1: Logistic Regression
# -------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# -------------------------
# Model 2: Random Forest
# -------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
