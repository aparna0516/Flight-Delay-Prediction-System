import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev_secret_key')

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_models():
    models = {}
    for name in ['xgb', 'rf', 'dt']:
        path = os.path.join(MODELS_DIR, f'{name}_model.joblib')
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # NOTE: For demo only. Replace with real auth.
        if username == 'admin' and password == 'password':
            return redirect(url_for('dashboard'))
        flash('Invalid credentials. Use admin / password for demo.')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    models = load_models()
    if request.method == 'POST':
        try:
            dep_delay = float(request.form.get('dep_delay', 0))
            distance = float(request.form.get('distance', 0))
            weather_delay = float(request.form.get('weather_delay', 0))
            hour = float(request.form.get('hour', 12))
            features = np.array([[dep_delay, distance, weather_delay, hour]])
            preds = {}
            for k, m in models.items():
                preds[k] = int(m.predict(features)[0])
            # simple ensemble: majority vote if multiple models exist
            votes = list(preds.values())
            ensemble = int(round(sum(votes)/len(votes))) if votes else 0
            return render_template('predict_result.html', preds=preds, ensemble=ensemble)
        except Exception as e:
            flash(f'Prediction error: {e}')
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    models = load_models()
    features = [data.get('dep_delay',0), data.get('distance',0), data.get('weather_delay',0), data.get('hour',12)]
    import numpy as np
    arr = np.array([features])
    result = {}
    for k, m in models.items():
        result[k] = int(m.predict(arr)[0])
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
