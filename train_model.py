import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Create sample data for testing
np.random.seed(42)
n_samples = 1000

# Generate sample data
age = np.random.randint(18, 80, n_samples)
income = np.random.normal(5000, 2000, n_samples)
income = np.clip(income, 1000, 20000) 
debt_ratio = np.random.beta(2, 5, n_samples)  
open_loans = np.random.poisson(3, n_samples)  
late_payments = np.random.poisson(0.5, n_samples)  


risk_score = (age * 0.01 + 
              (1 / (income / 1000)) * 0.3 + 
              debt_ratio * 2 + 
              open_loans * 0.1 + 
              late_payments * 0.5 + 
              np.random.normal(0, 0.2, n_samples))

y = (risk_score > np.percentile(risk_score, 80)).astype(int)

X = pd.DataFrame({
    'age': age,
    'MonthlyIncome': income,
    'DebtRatio': debt_ratio,
    'NumberOfOpenCreditLinesAndLoans': open_loans,
    'NumberOfTimes90DaysLate': late_payments
})

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear'))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'model.pkl')