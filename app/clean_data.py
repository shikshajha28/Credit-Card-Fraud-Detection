import pandas as pd

# Dataset load karo
df = pd.read_csv("data/credit_card_fraud_10k.csv")

print("Original dataset shape:")
print(df.shape)

# Missing values check karo
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows check karo
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Duplicate rows remove karo
df = df.drop_duplicates()

# Missing values remove karo
df = df.dropna()

print("\nAfter cleaning:")
print(df.shape)

# Fraud distribution
print("\nFraud counts:")
print(df["is_fraud"].value_counts())

# Clean dataset save karo
df.to_csv("data/cleaned_fraud_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
