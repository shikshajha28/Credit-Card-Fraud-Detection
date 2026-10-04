import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# 1. Cleaned dataset load karo
df = pd.read_csv("data/cleaned_fraud_data.csv")

print("Dataset loaded!")
print("Shape:", df.shape)


# 2. Target column alag karo
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


# 3. Transaction ID ko remove karo
X = X.drop("transaction_id", axis=1)


# 4. Categorical column
categorical_columns = ["merchant_category"]


# 5. Numerical columns
numerical_columns = [
    "amount",
    "transaction_hour",
    "foreign_transaction",
    "cardholder_age",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h"
]


# 6. Categorical data ko numbers me convert karo
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)


# 7. Data transform karo
X_processed = preprocessor.fit_transform(X)


# 8. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 9. Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# 10. Model train karo
print("\nTraining model...")
model.fit(X_train, y_train)

print("Model trained successfully!")


# 11. Prediction
y_pred = model.predict(X_test)


# 12. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# 13. Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 14. Model save karo
joblib.dump(model, "model/fraud_model.pkl")

# Preprocessor bhi save karo
joblib.dump(preprocessor, "model/preprocessor.pkl")

print("\nModel saved successfully!")
print("fraud_model.pkl")
print("preprocessor.pkl")
