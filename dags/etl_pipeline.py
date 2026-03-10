from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import redis
import json
import psycopg2
import os

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def data_quality_check(**context):
    df = pd.read_csv('/opt/airflow/data/processed/telco_clean.csv')
    null_rate = df.isnull().sum() / len(df)
    assert (null_rate < 0.05).all(), f"Null rate too high: {null_rate[null_rate >= 0.05]}"
    assert df['tenure'].min() >= 0, "Negative tenure detected"
    assert df['MonthlyCharges'].between(0, 200).all(), "MonthlyCharges out of range"
    required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_binary']
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"Data quality check passed ✅ — {len(df)} rows")
    return len(df)

def compute_features(**context):
    df = pd.read_csv('/opt/airflow/data/processed/telco_clean.csv')
    df['recency_risk'] = 1 / (df['tenure'] + 1)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']
    df['service_count'] = df[service_cols].sum(axis=1)
    df['charge_per_month'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['clv_proxy'] = df['TotalCharges'] * (1 - df['recency_risk'])
    df['is_high_value'] = (df['clv_proxy'] > df['clv_proxy'].quantile(0.75)).astype(int)
    df['contract_stability'] = df['Contract_encoded'] * df['tenure']
    df['digital_engagement'] = (
        df['PaperlessBilling'] +
        df['PaymentMethod_encoded'].apply(lambda x: 1 if x >= 2 else 0)
    )
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['monetary_value'] = df['TotalCharges']
    df.to_csv('/opt/airflow/data/processed/features_latest.csv', index=False)
    print(f"Features computed ✅ — {df.shape}")
    return df.shape[0]

def push_to_feature_store(**context):
    df = pd.read_csv('/opt/airflow/data/processed/features_latest.csv')
    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True
    )
    feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'recency_risk',
                    'service_count', 'charge_per_month', 'clv_proxy',
                    'is_high_value', 'contract_stability', 'digital_engagement',
                    'monthly_to_total_ratio', 'monetary_value', 'SeniorCitizen',
                    'Contract_encoded', 'InternetService_encoded', 'PaymentMethod_encoded']
    pushed = 0
    for idx, row in df.iterrows():
        user_id = row.get('customerID', f'user_{idx}')
        features = {col: float(row[col]) for col in feature_cols if col in row.index}
        r.setex(f"features:{user_id}", 86400, json.dumps(features))
        pushed += 1
    r.set('feature_store:last_updated', datetime.now().isoformat())
    r.set('feature_store:row_count', pushed)
    print(f"Pushed {pushed} users to Redis Feature Store ✅")
    return pushed

with DAG(
    'etl_pipeline',
    default_args=default_args,
    description='Daily ETL: quality check + feature engineering + Redis push',
    schedule_interval='0 2 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['etl', 'features'],
) as dag:
    quality_check = PythonOperator(task_id='data_quality_check', python_callable=data_quality_check)
    feature_engineering = PythonOperator(task_id='compute_features', python_callable=compute_features)
    redis_push = PythonOperator(task_id='push_to_feature_store', python_callable=push_to_feature_store)
    quality_check >> feature_engineering >> redis_push
