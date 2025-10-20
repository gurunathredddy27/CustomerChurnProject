# Telco Customer Churn — End-to-End
📌 Overview

- The Customer Churn Prediction Project is a complete end-to-end machine learning solution designed to predict whether a telecom customer will churn (leave the service) or stay subscribed.
- It covers every stage of the ML lifecycle — from data preprocessing, model training, and evaluation to deployment using FastAPI and Docker — making it production-ready and easy to extend.

🎯 Objective

Telecommunication companies often struggle to retain customers.
This project aims to:
- Predict the likelihood of customer churn.
- Identify key factors driving churn.
- Help businesses take data-driven retention actions.

Key Features

- ✅ Data Cleaning and Preprocessing using Pandas
- ✅ Model Training using Random Forest Classifier
- ✅ Model Evaluation using AUC, Accuracy, Precision, Recall, and F1-Score
- ✅ API Deployment using FastAPI + Uvicorn
- ✅ Containerization using Docker
- ✅ Ready for integration with web apps or dashboards
```
### Project Structure
Customer-Churn-Project/
│
├── data/                         # Raw dataset
│   └── Telco-Customer-Churn.csv
│
├── models/                       # Saved trained models
│   └── best_rf.pkl
│
├── src/
│   ├── api/                      # FastAPI app folder
│   │   ├── app.py                # Main API file
│   │   ├── predict_client.py     # Client script for testing API
│   │   └── __init__.py
│   │
│   ├── data_processing.py        # Data preprocessing functions
│   ├── features.py               # Feature selection & transformation
│   ├── model.py                  # Model creation logic
│   ├── train.py                  # Model training and evaluation
│   ├── predict.py                # Local prediction script
│   ├── evaluate.py               # Model performance metrics
│   ├── explain.py                # Model interpretation (optional)
│   └── __init__.py
│
├── notebooks/                    # Jupyter notebooks for exploration
│
├── Dockerfile                    # For containerizing the API
├── docker-compose.yml            # Optional Docker orchestration
├── requirements.txt              # All dependencies
├── README.md                     # Project documentation
└── .dockerignore / .gitignore
```

## Quick start (manual dataset)
1. Download `Telco-Customer-Churn.csv` 
2. Place the CSV at `data/raw/Telco-Customer-Churn.csv`.
3. Create a Python venv and install requirements:
``` bash
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### How It Works
1️⃣ Data Preprocessing (data_processing.py)

Handles missing values, encodes categorical columns, and ensures data consistency.

Converts Yes/No features to binary (1/0).

One-hot encodes service types, contract, and payment methods.

2️⃣ Feature Engineering (features.py)

Splits data into features (X) and labels (y).

Selects key predictors for the churn model.

3️⃣ Model Training (train.py)

Loads and preprocesses the data.

Trains a Random Forest Classifier.

Evaluates it using AUC and classification metrics.

Saves the model as best_rf.pkl.

4️⃣ Model Prediction (predict.py)

Loads the saved model and predicts churn on new data samples.

5️⃣ Model Serving via API (api/app.py)

A FastAPI endpoint exposes /predict to accept customer data in JSON format and return churn probability.

Example Request:
``` bash
{
    "tenure": 5,
    "MonthlyCharges": 70.3,
    "TotalCharges": 350.5,
    "Contract_Two year": 0,
    "InternetService_Fiber optic": 1
}
```

Example Response:
``` bash
{
    "Churn_Probability": 0.78,
    "Churn_Prediction": "Yes"
}
```
6️⃣ Containerization (Docker)

The Dockerfile builds the entire app with dependencies.

Run the following commands:
```
# Build Docker image
docker build -t churn-prediction-app .

# Run container
docker run -d -p 8501:8501 churn-prediction-app
```

Once running, the API will be accessible at:
👉 http://localhost:8501/docs

## Business Recommendations
- Based on model insights:

- Offer Loyalty Programs to customers with short tenure (< 6 months).
- Reduce Monthly Charges or introduce bundled offers for high-charge users.
- Improve Support for users lacking TechSupport or OnlineSecurity.
- Target Fiber-Optic Customers who show higher churn risk.
- Encourage Long-Term Contracts, as two-year customers churn less.