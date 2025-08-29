import os
import pickle
import pandas as pd
import json

from flask import Flask, render_template, request, flash, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from preprocessing.data_preprocessor import DataPreprocessor

# ======================
# App Initialization
# ======================
app = Flask(__name__)
app.secret_key = 'your_professional_secret_key'

# ======================
# File Paths
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_model.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "training", "credit_dataset.csv")
PROD_DATA_PATH = os.path.join(BASE_DIR, "data", "production", "neural_credit_data.csv")

# ======================
# Model & Data Functions
# ======================
def train_model():
    try:
        df_raw = pd.read_csv(TRAIN_DATA_PATH)
        feature_columns = df_raw.columns.drop(['Label', 'Username'], errors='ignore')
        preprocessor = DataPreprocessor(TRAIN_DATA_PATH, drop_columns=[])
        X, y = preprocessor.preprocess()
        
        if X.shape[1] != len(feature_columns):
            raise ValueError(f"Mismatch between processed data columns ({X.shape[1]}) and feature names ({len(feature_columns)}).")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump((clf, feature_columns), f)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        return clf, acc, X_test, y_test, feature_columns
        
    except Exception as e:
        return None, str(e), None, None, None

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model, columns = pickle.load(f)
            return model, columns
        except ValueError:
            print("--- Outdated model file found. Please retrain. ---")
            return None, None
    return None, None

# ======================
# Plotting Functions (ALL EDITED FOR LIGHT THEME)
# ======================
def create_feature_importance_plot(model, columns):
    importances = model.feature_importances_
    df = pd.DataFrame({'feature': columns, 'importance': importances}).sort_values('importance', ascending=True)
    # Use color_discrete_sequence to set the bar color to our theme's red
    fig = px.bar(df, x='importance', y='feature', orientation='h', title="Feature Importance",
                 color_discrete_sequence=['#DC2626'])
    # Set template to white and font color to a dark gray
    fig.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Poppins, sans-serif", color="#1F2937"))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_confusion_matrix_plot(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_text = [[str(y) for y in x] for x in cm]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=['High Risk (Pred)', 'Low Risk (Pred)'], y=['High Risk (True)', 'Low Risk (True)'],
        colorscale='Reds', showscale=False)) # Use Reds colorscale
    # Set template to white and font color to dark gray
    fig.update_layout(title_text='Confusion Matrix', template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Poppins, sans-serif", color="#1F2937"))
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            fig.add_annotation(x=j, y=i, text=cm_text[i][j], showarrow=False, font_size=16)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
def create_user_comparison_plot(user_data, username):
    prod_df = pd.read_csv(PROD_DATA_PATH)
    features_to_compare = [col for col in user_data.columns if col not in ['Username', 'Label']]
    
    avg_low_risk = prod_df[prod_df['Label'] == 1][features_to_compare].mean()
    avg_high_risk = prod_df[prod_df['Label'] == 0][features_to_compare].mean()
    user_values = user_data[features_to_compare].iloc[0]

    fig = go.Figure()
    # Updated bar colors for the light theme
    fig.add_trace(go.Bar(x=features_to_compare, y=avg_high_risk, name='Avg. High Risk', marker_color='#DC2626'))
    fig.add_trace(go.Bar(x=features_to_compare, y=avg_low_risk, name='Avg. Low Risk', marker_color='#059669'))
    fig.add_trace(go.Bar(x=features_to_compare, y=user_values, name=username, marker_color='#991B1B'))
    
    # Set template to white and font color to dark gray
    fig.update_layout(barmode='group', title=f'Feature Comparison for {username}', template='plotly_white',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title='Value',
                      font=dict(family="Poppins, sans-serif", color="#1F2937"))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ======================
# Flask Routes (No changes needed here)
# ======================
@app.route("/")
def overview():
    model, _ = load_model()
    accuracy = "Model Not Trained Yet"
    if model:
        accuracy = "Model Loaded"
    return render_template("index.html", accuracy=accuracy)

@app.route("/train", methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        model, acc, X_test, y_test, columns = train_model()
        if model:
            flash(f"Model trained with accuracy: {acc:.2f}", "success")
            y_pred = model.predict(X_test)
            fig_importance_json = create_feature_importance_plot(model, columns)
            fig_cm_json = create_confusion_matrix_plot(y_test, y_pred)
            return render_template("train.html", acc=acc, fig_importance_json=fig_importance_json, fig_cm_json=fig_cm_json)
        else:
            flash(f"Training failed: {acc}", "error")
    return render_template("train.html", acc=None)

@app.route("/predict-user", methods=['GET', 'POST'])
def predict_user_route():
    prod_df = pd.read_csv(PROD_DATA_PATH)
    users = prod_df["Username"].unique()
    
    if request.method == 'POST':
        model, columns = load_model()
        if not model:
            flash("Model not trained yet. Please train a model first.", "error")
            return redirect(url_for('train'))

        username = request.form.get("username")
        user_data = prod_df[prod_df["Username"] == username]
        
        if not user_data.empty:
            user_features = user_data[columns]
            prediction = model.predict(user_features)[0]
            probability = model.predict_proba(user_features)[0]
            
            risk_prob = probability[1] if prediction == 1 else probability[0]
            pred_text = "Low Risk" if prediction == 1 else "High Risk"
            flash_cat = "low-risk" if prediction == 1 else "high-risk"
            
            fig_user_comp_json = create_user_comparison_plot(user_data, username)
            
            return render_template("predict_user.html", users=users, prediction_made=True,
                                   username=username, pred_text=pred_text, risk_prob=f"{risk_prob:.2%}",
                                   flash_cat=flash_cat, fig_user_comp_json=fig_user_comp_json)
        else:
            flash(f"User '{username}' not found.", "error")
            
    return render_template("predict_user.html", users=users, prediction_made=False)

@app.route("/predict-all")
def predict_all_route():
    model, columns = load_model()
    if not model:
        flash("Model not trained yet. Please train a model first.", "error")
        return redirect(url_for('train'))
        
    df = pd.read_csv(PROD_DATA_PATH)
    df_features = df[columns]
    predictions = model.predict(df_features)
    df['Prediction'] = ["Low Risk" if p == 1 else "High Risk" for p in predictions]
    
    data_for_template = df.to_dict(orient='records')
    
    return render_template("predict_all.html", table_data=data_for_template, headers=df.columns)

@app.route("/add-user", methods=['GET', 'POST'])
def add_user_route():
    if request.method == 'POST':
        try:
            user_data = {
                "Username": request.form["username"], "Avg Weekly Calls": float(request.form["avg_calls"]),
                "SMS/Call Ratio": float(request.form["sms_ratio"]), "Finance App Hours": float(request.form["finance_hours"]),
                "Recharge Freq": int(request.form["recharge_freq"]), "On-time Payments": int(request.form["ontime_payments"]),
                "UPI Txn Count": int(request.form["upi_txn"]), "E-comm Spending": float(request.form["ecommerce_spending"]),
                "Behavioral Score": float(request.form["beh_score"]), "Label": int(request.form["label"]),
            }
            df = pd.read_csv(PROD_DATA_PATH)
            new_df = pd.concat([df, pd.DataFrame([user_data])], ignore_index=True)
            new_df.to_csv(PROD_DATA_PATH, index=False)
            flash(f"User '{user_data['Username']}' added successfully!", "success")
        except Exception as e:
            flash(f"Error adding user: {e}", "error")
        return redirect(url_for('add_user_route'))
    return render_template("add_user.html")

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    app.run(debug=True)
