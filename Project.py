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
