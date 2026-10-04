import pandas as pd
import joblib

# Model load karo
model = joblib.load("model/fraud_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

# Clean dataset load karo
df = pd.read_csv("data/cleaned_fraud_data.csv")

# Target alag karo
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Data transform karo
X_processed = preprocessor.transform(X)

# Prediction
predictions = model.predict(X_processed)

print("Predictions:")
print(predictions[:20])

print("\nActual values:")
print(y.head(20).values)
