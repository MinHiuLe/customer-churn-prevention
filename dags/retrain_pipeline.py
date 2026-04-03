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
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
)
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

# Threshold minimum untuk promote ke production — diambil dari env agar
# mudah diubah tanpa menyentuh kode.
MIN_AUC_TO_PROMOTE = float(os.getenv('MIN_AUC_TO_PROMOTE', '0.84'))


def check_drift_trigger(**context):
    drift_flag = '/opt/airflow/data/processed/drift_detected.flag'
    return os.path.exists(drift_flag)


def validate_dvc_hash(**context):
    trigger = context['task_instance'].xcom_pull(task_ids='check_drift')
    if not trigger:
        return None

    expected_md5 = None
    try:
        with open(DVC_PATH, 'r') as f:
            for line in f:
                if 'md5:' in line:
                    expected_md5 = line.split(':')[1].strip()
                    break
    except FileNotFoundError:
        return True

    if not expected_md5:
        raise ValueError(".dvc file missing md5 field")

    md5_hash = hashlib.md5()
    with open(CSV_PATH, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    actual_md5 = md5_hash.hexdigest()

    if actual_md5 != expected_md5:
        raise ValueError(
            f"Ghost Data detected!\n"
            f"DVC expected : {expected_md5}\n"
            f"Actual file  : {actual_md5}\n"
            f"Run 'dvc add' + 'git commit' before retrain."
        )

    return True


def retrain_model(**context):
    trigger = context['task_instance'].xcom_pull(task_ids='check_drift')
    if not trigger:
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
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        # Tính toán metrics và threshold tối ưu
        y_pred_proba = model.predict(X_test)
        auc          = roc_auc_score(y_test, y_pred_proba)

        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores     = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx      = f1_scores.argmax()
        best_threshold = float(thresholds[best_idx])

        y_pred_tuned = (y_pred_proba >= best_threshold).astype(int)

        mlflow.log_metrics({
            'auc_roc':   auc,
            'f1':        f1_score(y_test, y_pred_tuned),
            'precision': precision_score(y_test, y_pred_tuned),
            'recall':    recall_score(y_test, y_pred_tuned),
            'threshold': best_threshold,
        })

        signature = infer_signature(X_test, y_pred_proba)
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path='lgbm_churn',
            signature=signature,
        )

        # Tag DVC commit untuk reproducibility
        try:
            dvc_hash = subprocess.check_output(
                ['git', 'log', '--format=%H', '-1', DVC_PATH],
                cwd='/opt/airflow',
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            dvc_hash = 'unknown'
        mlflow.set_tag('dvc_data_commit', dvc_hash)

        run_id = mlflow.active_run().info.run_id

        # Champion/Challenger: promote jika AUC di atas minimum
        promoted = False
        try:
            version  = register_model(run_id, version_alias='staging')
            promoted = champion_challenger(version, auc)
            mlflow.set_tag('promoted', str(promoted).lower())
        except Exception:
            promoted = auc >= MIN_AUC_TO_PROMOTE

        if promoted:
            model.save_model(str(MODEL_DIR / 'lgbm_churn.txt'))

            # Simpan threshold ke model_config.json agar API dan batch
            # scoring bisa load threshold yang sama — tidak ada nilai hardcode
            model_config = {
                'churn_threshold': best_threshold,
                'auc_roc':         round(auc, 6),
                'trained_at':      datetime.now().isoformat(),
                'mlflow_run_id':   run_id,
            }
            with open(MODEL_DIR / 'model_config.json', 'w') as f:
                json.dump(model_config, f, indent=2)

            mlflow.log_artifact(str(MODEL_DIR / 'model_config.json'))
            print(f"Model promoted. threshold={best_threshold:.4f}  auc={auc:.4f}")
        else:
            print(f"Model NOT promoted. auc={auc:.4f} < required={MIN_AUC_TO_PROMOTE}")

    mlflow.lightgbm.autolog(disable=True)

    # Hapus drift flag setelah retrain selesai
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

    check_drift   = PythonOperator(task_id='check_drift',   python_callable=check_drift_trigger)
    validate_hash = PythonOperator(task_id='validate_dvc',  python_callable=validate_dvc_hash)
    retrain       = PythonOperator(task_id='retrain_model', python_callable=retrain_model)

    check_drift >> validate_hash >> retrain