from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

MODEL_DIR = Path('/opt/airflow/data/processed/models')

def check_drift_trigger(**context):
    drift_flag = '/opt/airflow/data/processed/drift_detected.flag'
    if os.path.exists(drift_flag):
        print("Drift flag found — proceeding with retrain ✅")
        return True
    print("No drift flag — skipping retrain")
    return False

def retrain_model(**context):
    trigger = context['task_instance'].xcom_pull(task_ids='check_drift')
    if not trigger:
        print("No retrain needed")
        return None

    df = pd.read_csv('/opt/airflow/data/processed/features_latest.csv')
    with open(MODEL_DIR / 'feature_cols.json') as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    y = df['Churn_binary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': 42,
        'verbose': -1,
    }

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('churn-prediction')

    with mlflow.start_run(run_name=f'retrain_{datetime.now().strftime("%Y%m%d_%H%M")}'):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test)
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        auc = roc_auc_score(y_test, model.predict(X_test))
        mlflow.log_metric('auc_roc', auc)
        mlflow.log_params(params)

        baseline_auc = 0.8445
        if auc > baseline_auc:
            model.save_model(str(MODEL_DIR / 'lgbm_churn.txt'))
            mlflow.set_tag('promoted', 'true')
            print(f"New model promoted ✅ AUC: {auc:.4f} > {baseline_auc}")
        else:
            mlflow.set_tag('promoted', 'false')
            print(f"Model NOT promoted — AUC: {auc:.4f} < {baseline_auc}")

        drift_flag = '/opt/airflow/data/processed/drift_detected.flag'
        if os.path.exists(drift_flag):
            os.remove(drift_flag)
        return auc

with DAG(
    'retrain_pipeline',
    default_args=default_args,
    description='Retrain model when drift detected',
    schedule_interval='0 4 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['retrain', 'mlops'],
) as dag:
    check_drift = PythonOperator(task_id='check_drift', python_callable=check_drift_trigger)
    retrain = PythonOperator(task_id='retrain_model', python_callable=retrain_model)
    check_drift >> retrain
