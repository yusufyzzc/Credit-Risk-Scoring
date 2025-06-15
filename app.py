from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['age'], data['income'], data['debt'], data['openLoans'], data['latePayments']]])
    risk = model.predict_proba(features)[0][1] * 100
    return jsonify({'risk_percent': risk})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
