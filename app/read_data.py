import pandas as pd

# Dataset path
file_path = "data/credit_card_fraud_10k.csv"

# Load dataset
df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nFraud counts:")
print(df["is_fraud"].value_counts())