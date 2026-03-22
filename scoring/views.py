import os
import json
import numpy as np
import joblib

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# Base directory of the project (where model.pkl and html files live)
BASE_DIR = settings.BASE_DIR

# Load model once at startup to avoid reloading on every request
_MODEL = joblib.load(BASE_DIR / 'model.pkl')



def home(request):
    """Serve the index.html landing page."""
    return render(request, 'scoring/index.html')


def predict_html(request):
    """Serve the predict.html page."""
    return render(request, 'scoring/predict.html')


@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    """
    Accept a JSON payload with age, income, debt, openLoans, latePayments.
    Load model.pkl, run prediction, and return {'risk_percent': float}.
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON body'}, status=400)

    required_fields = ['age', 'income', 'debt', 'openLoans', 'latePayments']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return JsonResponse({'error': f'Missing fields: {missing}'}, status=400)

    try:
        features = np.array([[
            data['age'],
            data['income'],
            data['debt'],
            data['openLoans'],
            data['latePayments'],
        ]])
        risk = _MODEL.predict_proba(features)[0][1] * 100
        return JsonResponse({'risk_percent': risk})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def health(request):
    """Simple health check endpoint."""
    return JsonResponse({'status': 'healthy', 'service': 'credit-risk-scoring'})
