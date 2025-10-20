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

✅ Data Cleaning and Preprocessing using Pandas
✅ Model Training using Random Forest Classifier
✅ Model Evaluation using AUC, Accuracy, Precision, Recall, and F1-Score
✅ API Deployment using FastAPI + Uvicorn
✅ Containerization using Docker
✅ Ready for integration with web apps or dashboards
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

## Notes & Next steps 
Place the CSV in data/raw/ before running src/train.py.

The src/features.py preprocessor expects the full set of raw CSV columns; the Streamlit example sends a minimal set and uses defaults — for production, create a contract and validate fields.

For production readiness: wrap preprocessing and model into a single sklearn Pipeline (already done in src/train.py) and add input validation.

If you want, I can now:

run training here and show model met 

## Business Recommendations
Based on model insights:

Offer Loyalty Programs to customers with short tenure (< 6 months).

Reduce Monthly Charges or introduce bundled offers for high-charge users.

Improve Support for users lacking TechSupport or OnlineSecurity.

Target Fiber-Optic Customers who show higher churn risk.

Encourage Long-Term Contracts, as two-year customers churn less.