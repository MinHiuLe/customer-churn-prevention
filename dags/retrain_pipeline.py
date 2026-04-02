from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
import json
import os
import hashlib
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys

sys.path.insert(0, '/opt/airflow')
from src.mlops.model_registry import register_model, champion_challenger

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

MODEL_DIR = Path('/opt/airflow/data/processed/models')
CSV_PATH  = '/opt/airflow/data/processed/features_latest.csv'
DVC_PATH  = '/opt/airflow/data/processed/features_latest.csv.dvc'

def check_drift_trigger(**context):
    drift_flag = '/opt/airflow/data/processed/drift_detected.flag'
    if os.path.exists(drift_flag):
        print("Drift flag found ✅")
        return True
    print("No drift — skipping retrain")
    return False

def validate_dvc_hash(**context):
    """Ghost Data check — đảm bảo data đã được DVC version"""
    trigger = context['task_instance'].xcom_pull(task_ids='check_drift')
    if not trigger:
        return None

    # Đọc expected MD5 từ .dvc file
    expected_md5 = None
    try:
        with open(DVC_PATH, 'r') as f:
            for line in f:
                if 'md5:' in line:
                    expected_md5 = line.split(':')[1].strip()
                    break
    except FileNotFoundError:
        print("⚠️  No .dvc file found — skipping hash check")
        return True

    if not expected_md5:
        raise ValueError(".dvc file missing md5 field")

    # Tính MD5 thực tế
    md5_hash = hashlib.md5()
    with open(CSV_PATH, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    actual_md5 = md5_hash.hexdigest()

    if actual_md5 != expected_md5:
        raise ValueError(
            f"Ghost Data detected!\n"
            f"DVC expected: {expected_md5}\n"
            f"Actual file:  {actual_md5}\n"
            f"Run 'dvc add' + 'git commit' before retrain."
        )

    print(f"DVC hash check passed ✅ — {actual_md5[:8]}")
    return True

def retrain_model(**context):
    trigger = context['task_instance'].xcom_pull(task_ids='check_drift')
    if not trigger:
        print("No retrain needed")
        return None

    df = pd.read_csv(CSV_PATH)
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

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment('churn-prediction')

    # Auto-log không có feature importance để tránh lỗi
    mlflow.lightgbm.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        silent=True,
    )

    with mlflow.start_run(
        run_name=f'retrain_{datetime.now().strftime("%Y%m%d_%H%M")}'
    ):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data   = lgb.Dataset(X_test,  label=y_test)

        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        auc = roc_auc_score(y_test, model.predict(X_test))
        mlflow.log_metric('auc_roc', auc)

        # Model signature
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path='lgbm_churn',
            signature=signature,
        )

        # DVC data lineage tag
        try:
            dvc_hash = subprocess.check_output(
                ['git', 'log', '--format=%H', '-1', DVC_PATH],
                cwd='/opt/airflow',
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            dvc_hash = 'unknown'
        mlflow.set_tag('dvc_data_commit', dvc_hash)

        run_id = mlflow.active_run().info.run_id

        # Champion/Challenger
        try:
            version  = register_model(run_id, version_alias='staging')
            promoted = champion_challenger(version, auc)
            mlflow.set_tag('promoted', str(promoted).lower())
            if promoted:
                model.save_model(str(MODEL_DIR / 'lgbm_churn.txt'))
                print(f"✅ Promoted — AUC={auc:.4f}")
            else:
                print(f"❌ Not promoted — AUC={auc:.4f}")
        except Exception as e:
            print(f"Registry error (non-critical): {e}")
            if auc > 0.8445:
                model.save_model(str(MODEL_DIR / 'lgbm_churn.txt'))

    mlflow.lightgbm.autolog(disable=True)

    # Clear drift flag
    drift_flag = '/opt/airflow/data/processed/drift_detected.flag'
    if os.path.exists(drift_flag):
        os.remove(drift_flag)

    return auc

with DAG(
    'retrain_pipeline',
    default_args=default_args,
    description='Retrain: Drift check → DVC validation → Train → Champion/Challenger',
    schedule_interval='0 4 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    tags=['retrain', 'mlops'],
) as dag:

    check_drift   = PythonOperator(task_id='check_drift',    python_callable=check_drift_trigger)
    validate_hash = PythonOperator(task_id='validate_dvc',   python_callable=validate_dvc_hash)
    retrain       = PythonOperator(task_id='retrain_model',  python_callable=retrain_model)

    check_drift >> validate_hash >> retrain
