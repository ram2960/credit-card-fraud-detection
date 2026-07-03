# Sentinel — Credit Card Fraud Detection

An end-to-end machine learning system that detects fraudulent credit card transactions in a dataset where fraud accounts for just **0.17%** of all records. The project covers the full pipeline — EDA, preprocessing, model comparison, an unsupervised anomaly detection baseline, threshold tuning, and a live Flask dashboard for interactive screening.

**[Live demo screenshot / GIF here]**

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Approach](#approach)
- [Results](#results)
- [Key Concepts Used](#key-concepts-used)
- [How to Run This Locally](#how-to-run-this-locally)
- [How to Use the Dashboard](#how-to-use-the-dashboard)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Credit card fraud detection is a classic **imbalanced classification** problem. In this dataset, only 492 of 284,807 transactions are fraudulent — roughly **1 in 578**. This creates a trap: a model that predicts "not fraud" on every single transaction would score **99.83% accuracy** while catching zero fraud.

This project is built around solving that trap properly — using the right evaluation metrics, comparing multiple modeling strategies, and being explicit about trade-offs rather than chasing a single misleading number.

## Dataset

- **Source:** [Credit Card Fraud Detection (Kaggle, ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 492 confirmed fraud cases (0.173%)
- **Features:** `Time`, `Amount`, and `V1`–`V28` — the latter are PCA-transformed components of the original transaction features, anonymized for confidentiality. `Class` is the target (1 = fraud, 0 = legitimate).

> **Note:** `creditcard.csv` (~150MB) is not included in this repo due to GitHub's file size limits. Download it from the Kaggle link above and place it in `data/creditcard.csv` before running the notebook.

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── creditcard.csv          # not included — download from Kaggle
│   └── demo_samples.csv        # sample transactions used by the dashboard
├── notebooks/
│   └── FraudDetection.ipynb    # full EDA -> modeling -> evaluation pipeline
├── models/
│   └── final_fraud_model.pkl   # trained XGBoost pipeline + threshold + scaler
├── src/
│   └── app.py                  # Flask dashboard
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── images/                 # confusion matrix, ROC, PR curve (generated)
├── requirements.txt
└── README.md
```

## Approach

1. **Exploratory Data Analysis** — examined class imbalance, transaction amount/time distributions, and feature correlation with fraud.
2. **Preprocessing** — scaled `Amount` and `Time` with `StandardScaler`; used a **stratified** train/test split so both sets preserve the real-world 0.17% fraud rate.
3. **Imbalance handling** — used `class_weight='balanced'` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost) rather than synthetic oversampling (SMOTE), to avoid introducing artificial data points and keep the model's decision boundary grounded in real transactions.
4. **Model comparison** — trained and evaluated Logistic Regression, Random Forest, and XGBoost, all under class-weighted loss.
5. **Unsupervised baseline** — trained an **Isolation Forest** with no access to fraud labels, to benchmark supervised performance against a purely anomaly-detection-based approach.
6. **Threshold tuning** — moved the decision threshold off the default 0.5 to maximize F1-score on the fraud class, reflecting the real business trade-off between missed fraud and false alarms.
7. **Deployment** — packaged the final model into a Flask dashboard for interactive transaction screening.

## Results

Evaluated on a held-out test set of 56,962 transactions (stratified, never seen during training):

| Model | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.056 | 0.874 | 0.106 | 0.672 | 0.966 |
| Random Forest | 0.774 | 0.758 | 0.766 | 0.778 | 0.980 |
| **XGBoost (selected)** | **0.961** | **0.779** | **0.861** | **0.825** | **0.973** |
| Isolation Forest (unsupervised) | 0.242 | 0.253 | 0.247 | 0.155 | 0.940 |

**XGBoost was selected** as the final model — it achieved the highest PR-AUC (0.825), which was used as the primary selection metric instead of accuracy or ROC-AUC, since PR-AUC is far more informative than either on a dataset this imbalanced.

**Top predictive features** (by XGBoost gain): `V14`, `V4`, `V12`, `V8`, `V3`, `V19` — these anonymized components carry the strongest signal for distinguishing fraud from legitimate activity.

## Key Concepts Used

**Why not accuracy?** On a 0.17%-fraud dataset, accuracy is close to meaningless — a model that never predicts fraud still scores 99.8%. This project uses **Precision, Recall, F1, and PR-AUC** instead, since they specifically measure performance on the rare, high-stakes fraud class.

**Supervised vs. unsupervised — why both?** The Isolation Forest baseline exists to make a specific point: supervised models (XGBoost, etc.) only learn to recognize fraud patterns present in their training labels — they can miss genuinely novel fraud types. Unsupervised anomaly detection has no such blind spot (it just flags "unusual" behavior, labeled or not) but pays for that generality with much lower precision, as the results table shows. Real production systems often run both in tandem — a supervised model for known fraud patterns, and an anomaly detector as a safety net for new ones.

**Threshold tuning as a business decision.** A model's default 0.5 cutoff is arbitrary. In fraud detection, a missed fraud (false negative) usually costs more than a false alarm (false positive) — so the deployed threshold was tuned to maximize F1 rather than left at the default, explicitly treating the cutoff as a policy choice, not just a modeling detail.

## How to Run This Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Set up the environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

**3. Get the dataset** (only needed if you want to re-run the notebook from scratch)

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`.

**4. Run the dashboard**
```bash
cd src
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

> The dashboard runs directly off the pre-trained model in `models/final_fraud_model.pkl` and `data/demo_samples.csv` — you don't need the full dataset or to re-run the notebook just to try the demo.

## How to Use the Dashboard

1. The top of the dashboard shows dataset statistics and the trained model's performance metrics at a glance.
2. Scroll to **"Screen a transaction"**, select one of the sample transactions (pulled from the real, held-out test set — never seen during training) from the dropdown.
3. Click **Run Model**. The dashboard shows:
   - The model's fraud probability, visualized as a radial gauge
   - Its final verdict (Fraud / Legitimate) at the tuned decision threshold
   - The transaction's actual ground-truth label, so you can see whether the model got it right
4. **Top predictive features** and **evaluation charts** (confusion matrix, ROC curve, PR curve) are further down for a deeper look at how the model behaves.

> **Why sample transactions instead of free-form input?** The model's features (`V1`–`V28`) are anonymized PCA components with no real-world meaning a user could type in directly — this is standard for public fraud datasets, since exposing the true underlying features would itself be a data leak. Sampling real, labeled test transactions keeps the demo honest rather than faking realism with made-up inputs.

## Future Improvements

- **Live-stream simulator** — a background process that periodically scores a new transaction automatically, mimicking a real-time transaction feed, to better illustrate how a production fraud system operates continuously rather than on-demand.
- **Feature-level "what-if" explorer** — sliders on the top predictive features (e.g., `V14`, `V4`), holding all others at their training-set mean, to let users see how the model's prediction shifts as key signals change.
- **Production-style feature engineering** — in a real system, features like these would be computed live from streaming transaction history (e.g., via Kafka + a feature store) rather than precomputed in a static CSV; this project currently uses the static, pre-engineered Kaggle dataset.
- **Model monitoring** — track prediction drift over time and trigger retraining as fraud patterns evolve.

## Tech Stack

**ML/Data:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
**App:** Flask, Jinja2, HTML/CSS
**Tooling:** Jupyter Notebook, joblib, Git

---

## Author

Built by Ram — [GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

If you find this useful or have suggestions, feel free to open an issue or reach out.