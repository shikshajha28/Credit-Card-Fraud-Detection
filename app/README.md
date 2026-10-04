# Credit Card Fraud Detection & Risk Prediction System

## Project Overview

This project is a Machine Learning based Credit Card Fraud Detection system that predicts whether a transaction is legitimate or potentially fraudulent.

The system also calculates fraud probability and classifies the transaction into Low, Medium, or High risk.

## Objectives

- Detect potentially fraudulent credit card transactions.
- Calculate fraud probability using Machine Learning.
- Classify transactions based on risk level.
- Provide an easy-to-use web interface using Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Machine Learning

The project uses a Machine Learning classification model trained on credit card transaction data.

The trained model and preprocessor are saved using Joblib.

## Input Features

The application uses the following transaction details:

- Transaction Amount
- Transaction Hour
- Merchant Category
- Foreign Transaction
- Cardholder Age
- Location Mismatch
- Device Trust Score
- Transactions in Last 24 Hours

## Risk Classification

- LOW: Fraud probability below 40%
- MEDIUM: Fraud probability between 40% and 69.99%
- HIGH: Fraud probability 70% or above

## Application

The project provides a Streamlit web application where users can enter transaction details and receive:

- Fraud/Legitimate prediction
- Fraud probability
- Risk level

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app/
│   ├── read_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   ├── predict.py
│   └── streamlit_app.py
│
├── data/
│   ├── credit_card_fraud_10k.csv
│   └── cleaned_fraud_data.csv
│
├── model/
│   ├── fraud_model.pkl
│   └── preprocessor.pkl
│
├── venv/
│
└── README.md