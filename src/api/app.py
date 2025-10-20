# src/api/app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)
MODEL_PATH = 'models/best_rf.pkl'

# Load model on startup
model = joblib.load(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.json
        # Expect payload to be a dict of raw fields (same as CSV columns minus target)
        df = pd.DataFrame([payload])
        # Ensure same columns exist (customerID optional)
        preds = model.predict_proba(df)[:, 1]
        return jsonify({'churn_prob': float(preds[0])})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)