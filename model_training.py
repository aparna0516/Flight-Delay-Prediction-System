# Script to train models on sample data and save them to models/
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

ROOT = os.path.dirname(__file__)
data_path = os.path.join(ROOT, 'sample_data', 'flight_sample.csv')
df = pd.read_csv(data_path)

# Basic feature engineering
# Features used: dep_delay, distance, weather_delay, hour
df['hour'] = df['scheduled_dep_time'].apply(lambda x: int(str(int(x)).zfill(4)[:2]) if not pd.isna(x) else 12)
X = df[['dep_delay', 'distance', 'weather_delay', 'hour']].fillna(0)
y = df['delayed'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'rf': RandomForestClassifier(n_estimators=100),
    'dt': DecisionTreeClassifier()
}

os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)

for name, model in models.items():
    print(f'Training {name} ...')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'{name} accuracy: {acc:.4f}')
    print(classification_report(y_test, preds))
    joblib.dump(model, os.path.join(ROOT, 'models', f'{name}_model.joblib'))

print('Models saved to models/ directory.')
