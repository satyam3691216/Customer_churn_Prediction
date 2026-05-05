import pandas as pd

print("Step 1: Loading dataset...")

df = pd.read_excel("churn.xlsx")

print("Dataset loaded successfully\n")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:\n", df.columns)
