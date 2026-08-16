"""
train_models.py
---------------
Downloads the Heart Disease UCI dataset, preprocesses it, trains 5 classification
models, evaluates them on 6 metrics, saves all model artifacts, and exports test_data.csv.

Run: python train_models.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────────────────────────────────────

# Heart Disease UCI – merged Cleveland + VA + Switzerland + Hungary (920 rows, 14 cols)
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
]

data_path = "data/heart_disease.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

if not os.path.exists(data_path):
    print("📥 Downloading Heart Disease dataset from UCI...")
    frames = []
    for url in URLS:
        try:
            df = pd.read_csv(url, header=None, names=COLUMN_NAMES, na_values="?")
            frames.append(df)
            print(f"   ✓ {url.split('/')[-1]}  →  {len(df)} rows")
        except Exception as e:
            print(f"   ✗ Failed: {url} — {e}")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(data_path, index=False)
    print(f"   Saved to {data_path}  ({len(raw)} total rows)\n")
else:
    print(f"📂 Dataset already exists at {data_path}")
    raw = pd.read_csv(data_path)

print(f"Dataset shape: {raw.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Preprocess
# ──────────────────────────────────────────────────────────────────────────────

df = raw.copy()

# Binarise target: 0 = no disease, 1 = disease
df["target"] = (df["target"] > 0).astype(int)

# Drop rows with too many missing values (>50% of features missing)
df = df.dropna(thresh=int(0.5 * df.shape[1]))

# Fill remaining NaNs with column median
df = df.fillna(df.median(numeric_only=True))

print(f"After preprocessing: {df.shape}  |  target distribution:\n{df['target'].value_counts().to_dict()}\n")

X = df.drop("target", axis=1)
y = df["target"]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Train / Test Split
# ──────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# Save test data (unscaled, with target) for Streamlit upload demo
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("💾 Saved test_data.csv\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Define & Train Models
# ──────────────────────────────────────────────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "kNN":                  KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":          GaussianNB(),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
}

MODEL_FILENAMES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree":        "model/decision_tree.pkl",
    "kNN":                  "model/knn.pkl",
    "Naive Bayes":          "model/naive_bayes.pkl",
    "Random Forest":        "model/random_forest.pkl",
}

# ──────────────────────────────────────────────────────────────────────────────
# 5. Evaluate & Save
# ──────────────────────────────────────────────────────────────────────────────

results = []

print("🏋️  Training models …\n")
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = (
        model.predict_proba(X_test_sc)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test_sc)
    )

    metrics = {
        "ML Model Name": name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    joblib.dump(model, MODEL_FILENAMES[name])
    print(f"  ✓ {name:25s}  acc={metrics['Accuracy']:.4f}  auc={metrics['AUC']:.4f}  f1={metrics['F1']:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Print Comparison Table
# ──────────────────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results).set_index("ML Model Name")
print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)
print(results_df.to_string())
print("=" * 80)

# Determine winner by F1 Score
winner = results_df["F1"].idxmax()
print(f"\n🏆 Overall Winner (best F1): {winner}")

# Save results to CSV for README reference
results_df.to_csv("model/metrics_results.csv")
print("\n💾 Saved model/metrics_results.csv")
print("\n✅ All done! Run `streamlit run app.py` to launch the web app.")
