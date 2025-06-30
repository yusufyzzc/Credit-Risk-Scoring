from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['age'], data['income'], data['debt'], data['openLoans'], data['latePayments']]])
    risk = model.predict_proba(features)[0][1] * 100
    return jsonify({'risk_percent': risk})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'credit-risk-scoring'})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

