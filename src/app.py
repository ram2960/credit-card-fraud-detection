import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# ---- Paths resolved relative to this file, so `python app.py` works
#      no matter which folder you run it from -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'final_fraud_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'demo_samples.csv')

# ---- Load model pipeline + demo data once at startup ------------------
pipeline = joblib.load(MODEL_PATH)
model = pipeline['model']
threshold = pipeline['threshold']
feature_names = pipeline['feature_names']

sample_data = pd.read_csv(DATA_PATH)

# ---- Precompute top feature importances (once, at startup) ------------
def get_top_features(n=6):
    if not hasattr(model, 'feature_importances_'):
        return []
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(n)
    max_val = top.max()
    return [
        {'name': name, 'pct': round(float(val / max_val) * 100, 1)}
        for name, val in top.items()
    ]

TOP_FEATURES = get_top_features()

# ---- In-memory session history (resets on server restart) -------------
recent_predictions = []


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    selected_id = None

    if request.method == 'POST':
        selected_id = int(request.form['transaction_id'])
        row = sample_data[sample_data['id'] == selected_id]
        features = row.drop(['id', 'actual_label', 'amount_raw'], axis=1)
        proba = float(model.predict_proba(features)[0][1])
        pred = 1 if proba >= threshold else 0
        probability_pct = round(proba * 100, 2)

        # Confidence = distance from the threshold, framed as a plain label
        if probability_pct >= 85 or probability_pct <= 15:
            confidence = "High"
        elif probability_pct >= 65 or probability_pct <= 35:
            confidence = "Medium"
        else:
            confidence = "Low"

        amount = float(row['Amount_scaled'].values[0]) if 'Amount_scaled' in row.columns else 0.0
        time_val = float(row['Time_scaled'].values[0]) if 'Time_scaled' in row.columns else 0.0

        result = {
            'id': selected_id,
            'amount': row['amount_raw'].values[0] if 'amount_raw' in row.columns else abs(amount) * 250,
            'time': int(abs(time_val) * 10000),
            'prediction': 'FRAUD' if pred == 1 else 'Legitimate',
            'probability': probability_pct,
            'confidence': confidence,
            'threshold': round(threshold, 3),
            'actual': 'FRAUD' if row['actual_label'].values[0] == 1 else 'Legitimate'
        }

        recent_predictions.insert(0, {
            'id': result['id'],
            'amount': result['amount'],
            'probability': result['probability'],
            'prediction': result['prediction'],
        })
        del recent_predictions[6:]  # keep last 6 only

    # Build sample dropdown list with amounts for display
    samples_display = []
    for _, r in sample_data.iterrows():
        amt = r['amount_raw'] if 'amount_raw' in sample_data.columns else abs(r.get('Amount_scaled', 0)) * 250
        samples_display.append({'id': int(r['id']), 'amount': float(amt)})

    return render_template(
        'index.html',
        samples=samples_display,
        result=result,
        selected_id=selected_id,
        top_features=TOP_FEATURES,
        recent=recent_predictions,
    )


if __name__ == '__main__':
    app.run(debug=True)