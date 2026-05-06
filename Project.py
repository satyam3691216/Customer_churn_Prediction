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

# Professional Graph Styling
sns.set_theme(
    style="whitegrid",
    palette="deep",
    context="talk"
)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Step 1: Load Dataset

print("Loading dataset...")

df = pd.read_excel("churn.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

print("\nDataset Loaded Successfully")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Columns:")
print(list(df.columns))

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Step 2: Data Cleaning

print("\nData Cleaning Started")

# Remove unnecessary ID columns
possible_id_cols = [
    'customerID',
    'CustomerID',
    'Customer ID'
]

for col in possible_id_cols:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Convert Total Charges column if available
if 'Total Charges' in df.columns:
    df['Total Charges'] = pd.to_numeric(
        df['Total Charges'],
        errors='coerce'
    )

# Check missing values
print("\nMissing Values Before Cleaning:\n")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Verify cleaning
print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

# Step 3: Exploratory Data Analysis

print("\nExploratory Data Analysis")

# Target column
target_col = 'Churn Value'

# Churn Distribution
plt.figure(figsize=(7,5))

colors = ["#00B894", "#D63031"]

sns.countplot(
    x=df[target_col],
    palette=colors
)

plt.title(
    "Customer Churn Distribution",
    fontsize=20,
    weight='bold'
)

plt.xlabel("Churn Value")
plt.ylabel("Customer Count")

plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

plt.savefig(
    "outputs/churn_distribution.png",
    dpi=300
)

plt.show()

# Monthly Charges vs Churn
if 'Monthly Charges' in df.columns:

    plt.figure(figsize=(8,6))

    sns.boxplot(
        x=target_col,
        y='Monthly Charges',
        data=df,
        palette=["#0984E3", "#E17055"],
        linewidth=2
    )

    plt.title(
        "Monthly Charges vs Customer Churn",
        fontsize=20,
        weight='bold'
    )

    plt.xlabel("Churn Value")
    plt.ylabel("Monthly Charges")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "outputs/monthly_charges_vs_churn.png",
        dpi=300
    )

    plt.show()

# Contract Type vs Churn
if 'Contract' in df.columns:

    plt.figure(figsize=(10,6))

    sns.countplot(
        x='Contract',
        hue=target_col,
        data=df,
        palette="magma"
    )

    plt.title(
        "Contract Type vs Customer Churn",
        fontsize=20,
        weight='bold'
    )

    plt.xlabel("Contract Type")
    plt.ylabel("Customer Count")

    plt.xticks(rotation=15)

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        "outputs/contract_vs_churn.png",
        dpi=300
    )

    plt.show()

print("\nInsights:")
print("- Customers with shorter contracts churn more.")
print("- Higher monthly charges are associated with higher churn.")

# Step 4: Feature Engineering

print("\nFeature Engineering")

# Convert categorical variables into numeric
df_model = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_model.drop('Churn Value', axis=1)
y = df_model['Churn Value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Step 5: Logistic Regression

print("\nLogistic Regression Model")

lr = LogisticRegression(max_iter=2000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(round(accuracy_score(y_test, lr_pred), 2))

print("\nClassification Report:\n")
print(classification_report(y_test, lr_pred))

# Step 6: Random Forest

print("\nRandom Forest Model")

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(round(accuracy_score(y_test, rf_pred), 2))

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

# Step 7: Confusion Matrix

print("\nGenerating Confusion Matrix")

plt.figure(figsize=(7,6))

cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    linewidths=1,
    linecolor='white'
)

plt.title(
    "Random Forest Confusion Matrix",
    fontsize=18,
    weight='bold'
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

# Step 8: Feature Importance

print("\nFeature Importance Analysis")

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

top_features = (
    importance
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(11,6))

sns.barplot(
    x=top_features.values,
    y=top_features.index,
    palette="rocket"
)

plt.title(
    "Top 10 Features Affecting Customer Churn",
    fontsize=20,
    weight='bold'
)

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.show()

print("\nTop Features Influencing Churn:\n")
print(top_features)

print("\nInsights:")
print("- Contract type, tenure, and monthly charges strongly affect churn.")
print("- Customers with shorter contracts are more likely to leave.")

# Step 9: Correlation Heatmap

print("\nGenerating Correlation Heatmap")

plt.figure(figsize=(14,10))

sns.heatmap(
    df_model.corr(),
    cmap="Spectral",
    center=0,
    linewidths=0.3
)

plt.title(
    "Feature Correlation Heatmap",
    fontsize=20,
    weight='bold'
)

plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300
)

plt.show()

# Step 10: Save Clean Dataset

df.to_csv(
    "clean_churn_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")
print("\nProject Completed Successfully.")