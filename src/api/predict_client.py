import requests


API_URL = 'http://127.0.0.1:5000/predict'


sample_payload = {
# include a minimal set of fields; ideally use full CSV columns
'gender': 'Female',
'SeniorCitizen': 0,
'Partner': 'No',
'Dependents': 'No',
'tenure': 1,
'PhoneService': 'No',
'MultipleLines': 'No phone service',
'InternetService': 'DSL',
'OnlineSecurity': 'No',
'OnlineBackup': 'Yes',
'DeviceProtection': 'No',
'TechSupport': 'No',
'StreamingTV': 'No',
'StreamingMovies': 'No',
'Contract': 'Month-to-month',
'PaperlessBilling': 'Yes',
'PaymentMethod': 'Electronic check',
'MonthlyCharges': 29.85,
'TotalCharges': 29.85
}


r = requests.post(API_URL, json=sample_payload)
print(r.json())