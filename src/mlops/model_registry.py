"""
MLflow Enterprise Model Registry
- Auto-logging + Signature
- Champion/Challenger comparison
- DVC data versioning link
"""
import os
import json
import subprocess
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score)

MODEL_NAME   = 'churn-lgbm-classifier'
BASELINE_AUC = 0.8445
MODEL_DIR    = Path('data/processed/models')
DATA_DIR     = Path('data/processed')


def get_dvc_commit(dvc_file: str) -> str:
    """Lấy git commit hash của .dvc file để link data version"""
    try:
        return subprocess.check_output(
            ['git', 'log', '--format=%H', '-1', dvc_file],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def train_and_log(X_train, X_test, y_train, y_test,
                  params: dict, run_name: str = 'lgbm_autolog') -> tuple:
    """
    Train LightGBM + log tất cả vào MLflow.
    Returns: (run_id, auc)
    """
    mlflow.lightgbm.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        silent=False,
    )

    with mlflow.start_run(run_name=run_name) as run:
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data   = lgb.Dataset(X_test,  label=y_test)

        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        # Metrics thật
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.446).astype(int)

        metrics = {
            'auc_roc':   roc_auc_score(y_test, y_prob),
            'f1':        f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred),
            'threshold': 0.446,
        }
        mlflow.log_metrics(metrics)

        # Model signature
        signature     = infer_signature(X_test, y_prob)
        input_example = X_test.iloc[:3]
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path='lgbm_churn',
            signature=signature,
            input_example=input_example,
        )

        # DVC data version tag
        dvc_hash = get_dvc_commit('data/processed/features.csv.dvc')
        mlflow.set_tag('dvc_data_commit', dvc_hash)
        mlflow.set_tag('dvc_data_file', 'data/processed/features.csv')

        run_id = run.info.run_id
        auc    = metrics['auc_roc']

        print(f"Run ID: {run_id}")
        print(f"AUC:    {auc:.4f}")
        print(f"DVC:    {dvc_hash[:8]}")

    mlflow.lightgbm.autolog(disable=True)
    return run_id, auc, model


def register_model(run_id: str, version_alias: str = 'staging') -> str:
    """Register model vào MLflow Registry → set alias"""
    client  = mlflow.tracking.MlflowClient()
    model_uri = f'runs:/{run_id}/lgbm_churn'

    registered = mlflow.register_model(model_uri, MODEL_NAME)
    version    = registered.version

    client.set_registered_model_alias(MODEL_NAME, version_alias, version)
    print(f"Model v{version} registered → alias='{version_alias}' ✅")
    return version


def champion_challenger(challenger_version: str, challenger_auc: float) -> bool:
    """
    So sánh challenger vs champion hiện tại.
    Returns True nếu challenger được promote lên champion.
    """
    client = mlflow.tracking.MlflowClient()

    # Lấy champion hiện tại
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, 'champion')
        champion_run = client.get_run(champion.run_id)
        champion_auc = champion_run.data.metrics.get('auc_roc', BASELINE_AUC)
        champion_version = champion.version

        print(f"Champion  (v{champion_version}): AUC = {champion_auc:.4f}")
        print(f"Challenger (v{challenger_version}): AUC = {challenger_auc:.4f}")

        if challenger_auc > champion_auc:
            print("✅ Challenger wins — promoting to champion")
            client.set_registered_model_alias(
                MODEL_NAME, 'archived', champion_version
            )
            client.set_registered_model_alias(
                MODEL_NAME, 'champion', challenger_version
            )
            return True
        else:
            print("❌ Champion wins — archiving challenger")
            client.set_registered_model_alias(
                MODEL_NAME, 'archived', challenger_version
            )
            return False

    except Exception:
        print("No existing champion — promoting challenger directly")
        client.set_registered_model_alias(
            MODEL_NAME, 'champion', challenger_version
        )
        return True


def load_champion():
    """Load champion model từ Registry"""
    return mlflow.lightgbm.load_model(f'models:/{MODEL_NAME}@champion')


def get_registry_summary() -> pd.DataFrame:
    """Trả về bảng tóm tắt tất cả model versions"""
    client   = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    rows = []
    for v in versions:
        run_metrics = client.get_run(v.run_id).data.metrics
        run_tags    = client.get_run(v.run_id).data.tags
        rows.append({
            'version':    v.version,
            'auc_roc':    round(run_metrics.get('auc_roc', 0), 4),
            'f1':         round(run_metrics.get('f1', 0), 4),
            'dvc_commit': run_tags.get('dvc_data_commit', 'N/A')[:8],
            'aliases':    v.aliases,
        })

    return pd.DataFrame(rows)
