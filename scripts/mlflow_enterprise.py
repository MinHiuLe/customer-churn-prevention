"""
MLflow Enterprise Script — dùng src/mlops/model_registry.py
Chạy độc lập hoặc được gọi bởi Airflow retrain DAG
"""
import os
import json
import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Import module mới
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mlops.model_registry import (
    train_and_log, register_model,
    champion_challenger, load_champion, get_registry_summary
)

load_dotenv()

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_DIR  = Path('data/processed/models')
DATA_DIR   = Path('data/processed')

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment('churn-prediction')

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'features.csv')
with open(MODEL_DIR / 'feature_cols.json') as f:
    feature_cols = json.load(f)

X = df[feature_cols]
y = df['Churn_binary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

params = {
    'objective':        'binary',
    'metric':           'auc',
    'num_leaves':       31,
    'learning_rate':    0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    'random_state':     42,
    'verbose':          -1,
}

# ── Run pipeline ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Train + Auto-log")
print("=" * 60)
run_id, auc, model = train_and_log(
    X_train, X_test, y_train, y_test, params
)

print("\n" + "=" * 60)
print("STEP 2 — Register Model")
print("=" * 60)
version = register_model(run_id, version_alias='staging')

print("\n" + "=" * 60)
print("STEP 3 — Champion/Challenger")
print("=" * 60)
promoted = champion_challenger(version, auc)

print("\n" + "=" * 60)
print("STEP 4 — Load Champion")
print("=" * 60)
champion = load_champion()
preds = champion.predict(X_test[:5])
print(f"Champion loaded ✅")
print(f"Sample predictions: {preds.round(3)}")

print("\n" + "=" * 60)
print("STEP 5 — Registry Summary")
print("=" * 60)
summary = get_registry_summary()
print(summary.to_string(index=False))

print(f"\n✅ MLflow Enterprise complete!")
print(f"Check: {MLFLOW_URI}/#/models/churn-lgbm-classifier")
