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

@app.route('/predict.html')
def predict_html():
    return send_from_directory('.', 'predict.html')

@app.route('/png/<path:filename>')
def serve_png(filename):
    return send_from_directory('png', filename)

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

