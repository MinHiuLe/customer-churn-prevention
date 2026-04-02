"""
MLflow Enterprise Setup — chạy sau mỗi lần train model
"""
import os
import json
import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score)

load_dotenv()

# ── Config ────────────────────────────────────────────────────
MLFLOW_URI  = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_DIR   = Path('data/processed/models')
DATA_DIR    = Path('data/processed')
EXPERIMENT  = 'churn-prediction'
BASELINE_AUC = 0.8445

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'features.csv')
with open(MODEL_DIR / 'feature_cols.json') as f:
    feature_cols = json.load(f)

X = df[feature_cols]
y = df['Churn_binary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── STEP 1: Auto-logging + Signature ─────────────────────────
print("=" * 60)
print("STEP 1 — LightGBM với Auto-logging + Signature")
print("=" * 60)

mlflow.lightgbm.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=False,
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

with mlflow.start_run(run_name='lgbm_autolog') as run:
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_test,  label=y_test)

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Tính metrics thật
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.446).astype(int)
    auc       = roc_auc_score(y_test, y_prob)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)

    mlflow.log_metrics({
        'auc_roc':   auc,
        'f1':        f1,
        'precision': precision,
        'recall':    recall,
        'threshold': 0.446,
    })

    # Model Signature
    signature    = infer_signature(X_test, y_prob)
    input_example = X_test.iloc[:3]

    # Log model với signature
    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path='lgbm_churn',
        signature=signature,
        input_example=input_example,
    )

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"AUC: {auc:.4f}")

mlflow.lightgbm.autolog(disable=True)

# ── STEP 2: Model Registry ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Model Registry: Register + Stage")
print("=" * 60)

client = mlflow.tracking.MlflowClient(MLFLOW_URI)
model_name = 'churn-lgbm-classifier'

# Register model
model_uri = f'runs:/{run_id}/lgbm_churn'
try:
    registered = mlflow.register_model(model_uri, model_name)
    version = registered.version
    print(f"Registered model version: {version}")
except Exception as e:
    print(f"Register note: {e}")
    versions = client.search_model_versions(f"name='{model_name}'")
    version = versions[0].version if versions else '1'

# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage='Staging',
    archive_existing_versions=False,
)
print(f"Model v{version} → Staging ✅")

# ── STEP 3: Champion/Challenger ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Champion/Challenger Comparison")
print("=" * 60)

# Lấy model đang Production (Champion)
production_versions = client.get_latest_versions(
    model_name, stages=['Production']
)

challenger_auc = auc

if production_versions:
    champion_version = production_versions[0]
    champion_run     = client.get_run(champion_version.run_id)
    champion_auc     = champion_run.data.metrics.get('auc_roc', BASELINE_AUC)

    print(f"Champion (v{champion_version.version}): AUC = {champion_auc:.4f}")
    print(f"Challenger (v{version}):              AUC = {challenger_auc:.4f}")

    if challenger_auc > champion_auc:
        print(f"✅ Challenger wins! Promoting to Production...")

        # Archive champion
        client.transition_model_version_stage(
            name=model_name,
            version=champion_version.version,
            stage='Archived',
        )

        # Promote challenger
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage='Production',
            archive_existing_versions=True,
        )

        # Set alias
        try:
            client.set_registered_model_alias(
                model_name, 'champion', version
            )
        except Exception:
            pass

        print(f"Model v{version} → Production ✅ (new champion)")
    else:
        print(f"❌ Champion wins — keeping current Production model")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage='Archived',
        )
else:
    # Không có Production model → promote thẳng
    print(f"No existing champion — promoting v{version} to Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage='Production',
        archive_existing_versions=True,
    )
    try:
        client.set_registered_model_alias(
            model_name, 'champion', version
        )
    except Exception:
        pass
    print(f"Model v{version} → Production ✅ (first champion)")

# ── STEP 4: Load Production model ────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Load Champion model từ Registry")
print("=" * 60)

try:
    champion = mlflow.lightgbm.load_model(
        f'models:/{model_name}/Production'
    )
    test_pred = champion.predict(X_test[:5])
    print(f"Champion model loaded ✅")
    print(f"Sample predictions: {test_pred.round(3)}")
except Exception as e:
    print(f"Load via alias: {e}")
    champion = mlflow.lightgbm.load_model(
        f'models:/{model_name}@champion'
    )

# ── STEP 5: Summary ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Registry Summary")
print("=" * 60)

all_versions = client.search_model_versions(f"name='{model_name}'")
for v in all_versions:
    run_metrics = client.get_run(v.run_id).data.metrics
    auc_val = run_metrics.get('auc_roc', 0)
    print(f"v{v.version:<3} | {v.current_stage:<12} | AUC={auc_val:.4f}")

print(f"\n✅ MLflow Enterprise setup complete!")
print(f"Check: {MLFLOW_URI}/#/models/{model_name}")
