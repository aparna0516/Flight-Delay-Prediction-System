# Flight Delay Prediction System (Full-featured)

## Overview
A full-featured Flask web application that predicts flight delays using machine learning models (XGBoost, Random Forest, Decision Tree).

## Features
- Train models using `model_training.py` (sample dataset provided).
- Flask app with routes for login (demo), dashboard, prediction form, and an API endpoint `/api/predict`.
- Models saved in `models/` (created when training).
- Simple ensemble prediction in the UI.

## Quickstart (local)
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux / macOS)
   venv\Scripts\activate    # (Windows)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train models:
   ```bash
   python model_training.py
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open http://localhost:5000 in your browser. Use `admin` / `password` to login (demo).

## Project structure
- `app.py` - Flask application
- `model_training.py` - Training and model save script
- `models/` - Trained model files (joblib)
- `sample_data/flight_sample.csv` - Sample CSV you can expand with real data
- `templates/`, `static/` - Web UI assets

## Notes & Next steps
- Replace demo login with proper authentication (Flask-Login or OAuth).
- Add real dataset and more features (airport codes, carrier, weather APIs).
- Deploy to Heroku / Railway / AWS Elastic Beanstalk.
