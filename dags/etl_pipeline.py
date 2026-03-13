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

def get_pg_conn():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', 5432),
        dbname=os.getenv('POSTGRES_DB', 'churn_db'),
        user=os.getenv('POSTGRES_USER', 'churn_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'churn_pass'),
    )

def data_quality_check(**context):
    """Load từ PostgreSQL và validate"""
    conn = get_pg_conn()
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()

    assert len(df) > 0, "No customers in database"
    assert df['tenure'].min() >= 0, "Negative tenure detected"
    assert df['monthly_charges'].between(0, 200).all(), "MonthlyCharges out of range"
    null_rate = df.isnull().sum() / len(df)
    assert (null_rate < 0.10).all(), f"Null rate too high: {null_rate[null_rate >= 0.10]}"

    print(f"Data quality check passed ✅ — {len(df)} customers in DB")
    return len(df)

def compute_and_save_features(**context):
    """ETL: PostgreSQL → feature engineering → save back to PostgreSQL"""
    conn = get_pg_conn()
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()

    # Rename columns để match feature engineering
    df = df.rename(columns={
        'customer_id': 'customerID',
        'senior_citizen': 'SeniorCitizen',
        'partner': 'Partner',
        'dependents': 'Dependents',
        'phone_service': 'PhoneService',
        'multiple_lines': 'MultipleLines',
        'internet_service': 'InternetService_encoded',
        'online_security': 'OnlineSecurity',
        'online_backup': 'OnlineBackup',
        'device_protection': 'DeviceProtection',
        'tech_support': 'TechSupport',
        'streaming_tv': 'StreamingTV',
        'streaming_movies': 'StreamingMovies',
        'contract_encoded': 'Contract_encoded',
        'paperless_billing': 'PaperlessBilling',
        'payment_method_encoded': 'PaymentMethod_encoded',
        'monthly_charges': 'MonthlyCharges',
        'total_charges': 'TotalCharges',
        'churn_binary': 'Churn_binary',
    })

    # Feature Engineering
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']
    df['recency_risk'] = 1 / (df['tenure'] + 1)
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

    # Save features back to PostgreSQL
    conn = get_pg_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS customer_features (
            customer_id VARCHAR(50) PRIMARY KEY,
            tenure FLOAT, senior_citizen INT, partner INT, dependents INT,
            phone_service INT, multiple_lines INT, online_security INT,
            online_backup INT, device_protection INT, tech_support INT,
            streaming_tv INT, streaming_movies INT, contract_encoded INT,
            paperless_billing INT, payment_method_encoded INT,
            monthly_charges FLOAT, total_charges FLOAT,
            internet_service_encoded INT, churn_binary INT,
            recency_risk FLOAT, service_count INT, charge_per_month FLOAT,
            clv_proxy FLOAT, is_high_value INT, contract_stability FLOAT,
            digital_engagement INT, monthly_to_total_ratio FLOAT,
            monetary_value FLOAT, updated_at TIMESTAMP DEFAULT NOW()
        )
    """)

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO customer_features VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s, NOW()
            ) ON CONFLICT (customer_id) DO UPDATE SET
                tenure=EXCLUDED.tenure,
                monthly_charges=EXCLUDED.monthly_charges,
                clv_proxy=EXCLUDED.clv_proxy,
                recency_risk=EXCLUDED.recency_risk,
                charge_per_month=EXCLUDED.charge_per_month,
                contract_stability=EXCLUDED.contract_stability,
                updated_at=NOW()
        """, (
            row['customerID'], row['tenure'],
            int(row['SeniorCitizen']), int(row['Partner']),
            int(row['Dependents']), int(row['PhoneService']),
            int(row['MultipleLines']), int(row['OnlineSecurity']),
            int(row['OnlineBackup']), int(row['DeviceProtection']),
            int(row['TechSupport']), int(row['StreamingTV']),
            int(row['StreamingMovies']), int(row['Contract_encoded']),
            int(row['PaperlessBilling']), int(row['PaymentMethod_encoded']),
            float(row['MonthlyCharges']), float(row['TotalCharges']),
            int(row['InternetService_encoded']), int(row['Churn_binary']),
            float(row['recency_risk']), int(row['service_count']),
            float(row['charge_per_month']), float(row['clv_proxy']),
            int(row['is_high_value']), float(row['contract_stability']),
            int(row['digital_engagement']), float(row['monthly_to_total_ratio']),
            float(row['monetary_value']),
        ))

    conn.commit()
    cur.close()
    conn.close()

    # Also save CSV for batch scoring compatibility
    df.to_csv('/opt/airflow/data/processed/features_latest.csv', index=False)
    print(f"Features saved to PostgreSQL + CSV ✅ — {len(df)} customers")
    return len(df)

def push_to_feature_store(**context):
    """PostgreSQL features → Redis Feature Store"""
    conn = get_pg_conn()
    df = pd.read_sql("SELECT * FROM customer_features", conn)
    conn.close()

    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True
    )

    pushed = 0
    for _, row in df.iterrows():
        features = row.to_dict()
        features.pop('customer_id', None)
        features.pop('updated_at', None)
        r.setex(
            f"features:{row['customer_id']}",
            86400,
            json.dumps({k: float(v) if v is not None else 0.0
                       for k, v in features.items()})
        )
        pushed += 1

    r.set('feature_store:last_updated', datetime.now().isoformat())
    r.set('feature_store:row_count', pushed)
    print(f"Pushed {pushed} users to Redis ✅")
    return pushed

with DAG(
    'etl_pipeline',
    default_args=default_args,
    description='Daily ETL: PostgreSQL → feature engineering → Redis',
    schedule_interval='0 2 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['etl', 'features'],
) as dag:

    quality_check = PythonOperator(
        task_id='data_quality_check',
        python_callable=data_quality_check,
    )
    feature_engineering = PythonOperator(
        task_id='compute_features',
        python_callable=compute_and_save_features,
    )
    redis_push = PythonOperator(
        task_id='push_to_feature_store',
        python_callable=push_to_feature_store,
    )

    quality_check >> feature_engineering >> redis_push
