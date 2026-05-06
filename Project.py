import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Step 1: Loading dataset...")

df = pd.read_excel("churn.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully\n")
print(df.head())

print("\nShape:", df.shape)
print("\nColumns:\n", df.columns)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

print("\nStep 2: Data Cleaning")

# Drop ID column
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# Fix TotalCharges column
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

# Check missing values
print("\nMissing values BEFORE cleaning:\n")
print(df.isnull().sum())

# Remove missing rows
df.dropna(inplace=True)

# Verify cleaning
print("\nMissing values AFTER cleaning:\n")
print(df.isnull().sum())

print("\nCleaned Shape:", df.shape)

print("\nStep 3: Exploratory Data Analysis")

sns.set_style("whitegrid")

# Detect target column safely
target_col = None

possible_targets = ['Churn', 'churn', 'Exited', 'Attrition']

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("Target column not found.")

# Churn Distribution
plt.figure(figsize=(6,4))

sns.countplot(
    x=df[target_col],
    palette="Set2"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "outputs/churn_distribution.png",
    dpi=300
)

plt.show()

# Monthly Charges vs Churn
if 'MonthlyCharges' in df.columns:

    plt.figure(figsize=(7,5))

    sns.boxplot(
        x=target_col,
        y='MonthlyCharges',
        data=df,
        palette="coolwarm"
    )

    plt.title("Monthly Charges vs Churn")

    plt.tight_layout()

    plt.savefig(
        "outputs/monthly_charges_vs_churn.png",
        dpi=300
    )

    plt.show()

# Contract Type vs Churn
if 'Contract' in df.columns:

    plt.figure(figsize=(8,5))

    sns.countplot(
        x='Contract',
        hue=target_col,
        data=df,
        palette="viridis"
    )

    plt.title("Contract Type vs Churn")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "outputs/contract_vs_churn.png",
        dpi=300
    )

    plt.show()

print("\nInsights:")
print("- Customers with higher monthly charges tend to churn more.")
print("- Month-to-month contract customers have the highest churn rate.")

print("\nStep 4: Feature Engineering & Modeling")

# Convert categorical variables → numeric
df_model = pd.get_dummies(df, drop_first=True)

# Detect encoded target column
target_encoded = None

for col in df_model.columns:
    if col.startswith(target_col + "_"):
        target_encoded = col
        break

if target_encoded is None:
    raise ValueError("Encoded target column not found.")

# Define features and target
X = df_model.drop(target_encoded, axis=1)
y = df_model[target_encoded]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Logistic Regression
print("\nLogistic Regression Results")

lr = LogisticRegression(max_iter=2000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nAccuracy:",
      round(accuracy_score(y_test, lr_pred), 2))

print("\nClassification Report:\n")
print(classification_report(y_test, lr_pred))

# Random Forest
print("\nRandom Forest Results")

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nAccuracy:",
      round(accuracy_score(y_test, rf_pred), 2))

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

# Confusion Matrix
plt.figure(figsize=(6,5))

cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nStep 5: Feature Importance Analysis")

# Feature importance
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

top_features = (
    importance
    .sort_values(ascending=False)
    .head(10)
)

# Plot feature importance
plt.figure(figsize=(10,6))

sns.barplot(
    x=top_features.values,
    y=top_features.index,
    palette="viridis"
)

plt.title("Top 10 Features Affecting Customer Churn")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.show()

print("\nTop Features Influencing Churn:\n")
print(top_features)

print("\nInsights:")
print("- Contract type, tenure, and monthly charges strongly influence churn.")
print("- Customers with shorter contracts are more likely to leave.")

# Correlation Heatmap
plt.figure(figsize=(12,8))

sns.heatmap(
    df_model.corr(),
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300
)

plt.show()

# Save cleaned dataset
df.to_csv(
    "clean_churn_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")
print("Project Completed Successfully.")