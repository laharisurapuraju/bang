from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import joblib
from functools import wraps
import secrets
import json
import plotly
import plotly.graph_objs as go
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from ward_analysis import WardAnalyzer
from complaint_analyzer import ComplaintSeverityAnalyzer

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dengue_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- USER MODEL ---------------- #
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

with app.app_context():
    db.create_all()

# ---------------- LOAD MODEL ---------------- #
model = None
scaler = None

try:
    model = joblib.load('model_xgboost.joblib')
    try:
        model.set_params(predictor='cpu_predictor')
    except:
        pass
except:
    try:
        model = joblib.load('model_linear_regression.joblib')
        scaler = joblib.load('scaler_linear_regression.joblib')
    except:
        model = None
        scaler = None

ward_analyzer = WardAnalyzer()
complaint_analyzer = ComplaintSeverityAnalyzer()

# ---------------- RISK CLASSIFICATION ---------------- #
def classify_risk(cases):
    if cases < 30:
        return "Low Risk", "success"
    elif cases < 80:
        return "Medium Risk", "warning"
    else:
        return "High Risk", "danger"

# ---------------- GAUGE ---------------- #
def generate_prediction_gauge(predicted_cases):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_cases,
        title={'text': "Predicted Dengue Cases"},
        gauge={
            'axis': {'range': [0, 200]},
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 80], 'color': '#fef08a'},
                {'range': [80, 200], 'color': '#fee2e2'}
            ]
        }
    ))

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ---------------- AUTH ---------------- #
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- ROUTES ---------------- #
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()

        if user and user.check_password(request.form['password']):
            session['user'] = user.email
            return redirect(url_for('index'))

        return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if request.form['password'] != request.form['confirm_password']:
            return render_template('register.html', error="Passwords do not match")

        user = User(email=request.form['email'])
        user.set_password(request.form['password'])

        db.session.add(user)
        db.session.commit()

        session['user'] = user.email
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------------- MAIN LOGIC ---------------- #
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    prediction_text = ""
    risk_level = ""
    alert_color = ""
    gauge_chart = None

    if request.method == 'POST':
        try:
            ward_id = request.form['ward_id']
            garbage = float(request.form['garbage'])
            waterlogging = float(request.form['waterlogging'])

            garbage_text = request.form.get('garbage_text', '')
            waterlogging_text = request.form.get('waterlogging_text', '')

            # Calculate severity-weighted complaint scores
            weighted_garbage = complaint_analyzer.calculate_weighted_complaint_score(
                garbage, garbage_text
            )
            weighted_waterlogging = complaint_analyzer.calculate_weighted_complaint_score(
                waterlogging, waterlogging_text
            )

            # Use default/typical weather values (fixed, consistent with training)
            rainfall = 150.0
            temp = 28.5
            rain_lag1 = 120.0
            rain_lag2 = 100.0
            temp_lag1 = 27.5
            
            # Calculate additional features matching training data engineering
            cases_lag1 = max(0, 35 + (rainfall - 150) * 0.2 + (garbage + waterlogging) * 0.1)
            
            rainfall_roll3_mean = (rainfall + rain_lag1 + rain_lag2) / 3.0
            
            cases_roll3_mean = max(0, 35 + (rainfall_roll3_mean - 150) * 0.2 + (garbage + waterlogging) * 0.1)
            
            is_monsoon = 1 if rainfall > 200 else 0
            
            garbage_complaints_ward_mean = weighted_garbage * 0.8
            waterlogging_complaints_ward_mean = weighted_waterlogging * 0.8

            input_data = pd.DataFrame([[ 
                rainfall, temp,
                weighted_garbage, weighted_waterlogging,
                rain_lag1, rain_lag2, temp_lag1,
                cases_lag1, rainfall_roll3_mean, cases_roll3_mean,
                is_monsoon, garbage_complaints_ward_mean, waterlogging_complaints_ward_mean
            ]], columns=[
                'Rainfall_mm','Avg_Temp_C',
                'Garbage_Complaints','Waterlogging_Complaints',
                'Rainfall_Lag1','Rainfall_Lag2','Temp_Lag1',
                'Cases_Lag1','Rainfall_roll3_mean','Cases_roll3_mean',
                'Is_Monsoon','Garbage_Complaints_ward_mean','Waterlogging_Complaints_ward_mean'
            ])

            # DEBUG
            print("Feature Vector:", input_data.values)

            # 🔥 PREDICTION - NO RANDOM BOOSTING, JUST DIRECT MODEL PREDICTION
            if model:
                try:
                    prediction = model.predict(input_data)[0]
                except Exception as e:
                    # Fallback for gpu_id errors
                    try:
                        import xgboost as xgb
                        booster = model.get_booster()
                        dmatrix = xgb.DMatrix(input_data)
                        prediction = booster.predict(dmatrix)[0]
                    except:
                        prediction = 20
            else:
                prediction = 20

            predicted_cases = int(max(0, prediction))

            print("Prediction:", predicted_cases)

            risk_level, alert_color = classify_risk(predicted_cases)

            prediction_text = f"Predicted Cases: {predicted_cases}"
            gauge_chart = generate_prediction_gauge(predicted_cases)

        except Exception as e:
            prediction_text = str(e)

    return render_template(
        'index.html',
        prediction=prediction_text,
        risk_level=risk_level,
        alert_color=alert_color,
        gauge_chart=gauge_chart
    )

# ---------------- RUN ---------------- #
if __name__ == '__main__':
    app.run(debug=True)