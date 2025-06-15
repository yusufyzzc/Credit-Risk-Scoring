import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Veriyi yükle ve temizle
df = pd.read_csv("cs-training.csv", index_col=0)
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
df = df[df['age'] > 0]
df['age'] = df['age'].clip(18, 95)

# Özellikler
X = df[['age', 'MonthlyIncome', 'DebtRatio', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate']]
y = df['SeriousDlqin2yrs']

# Eğitim seti
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear'))
])

pipeline.fit(X_train, y_train)

# ✅ Modeli yeniden kaydet
joblib.dump(pipeline, 'model.pkl')
