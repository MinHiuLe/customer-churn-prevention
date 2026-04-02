from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import redis
import json
import os

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def get_pg_conn():
    return PostgresHook(postgres_conn_id='churn_postgres_conn').get_conn()

def data_validation(**context):
    conn = get_pg_conn()
    df   = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()

    if len(df) == 0:
        raise ValueError("Data Validation Failed: Database is empty")

    null_counts = df.isnull().sum().sum()
    churn_rate  = df['churn_binary'].mean()
    avg_monthly = df['monthly_charges'].mean()

    print("-" * 40)
    print("DATA HEALTH REPORT")
    print("-" * 40)
    print(f"Total records:    {len(df)}")
    print(f"Churn rate:       {churn_rate:.2%}")
    print(f"Avg monthly:      ${avg_monthly:.2f}")
    print(f"Total nulls:      {null_counts}")
    print("-" * 40)

    if churn_rate > 0.8:
        raise ValueError(f"Churn rate too high: {churn_rate:.2%}")
    if null_counts / (len(df) * len(df.columns)) > 0.05:
        raise ValueError(f"Null rate too high: {null_counts}")

    return len(df)

def compute_features(**context):
    conn = get_pg_conn()
    df   = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()

    df = df.rename(columns={
        'customer_id':          'customerID',
        'senior_citizen':       'SeniorCitizen',
        'partner':              'Partner',
        'dependents':           'Dependents',
        'phone_service':        'PhoneService',
        'multiple_lines':       'MultipleLines',
        'internet_service':     'InternetService_encoded',
        'online_security':      'OnlineSecurity',
        'online_backup':        'OnlineBackup',
        'device_protection':    'DeviceProtection',
        'tech_support':         'TechSupport',
        'streaming_tv':         'StreamingTV',
        'streaming_movies':     'StreamingMovies',
        'contract_encoded':     'Contract_encoded',
        'paperless_billing':    'PaperlessBilling',
        'payment_method_encoded': 'PaymentMethod_encoded',
        'monthly_charges':      'MonthlyCharges',
        'total_charges':        'TotalCharges',
        'churn_binary':         'Churn_binary',
    })

    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']

    df['recency_risk']         = 1 / (df['tenure'] + 1)
    df['service_count']        = df[service_cols].sum(axis=1)
    df['charge_per_month']     = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['clv_proxy']            = df['TotalCharges'] * (1 - df['recency_risk'])
    df['is_high_value']        = (df['clv_proxy'] > df['clv_proxy'].quantile(0.75)).astype(int)
    df['contract_stability']   = df['Contract_encoded'] * df['tenure']
    df['digital_engagement']   = (
        df['PaperlessBilling'] +
        df['PaymentMethod_encoded'].apply(lambda x: 1 if x >= 2 else 0)
    )
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['monetary_value']         = df['TotalCharges']

    # Save to CSV for DVC tracking
    csv_path = '/opt/airflow/data/processed/features_latest.csv'
    df.to_csv(csv_path, index=False)
    print(f"Features saved to CSV ✅ — {len(df)} rows")
    return len(df)

def push_to_redis(**context):
    conn = get_pg_conn()
    df   = pd.read_sql("SELECT * FROM customer_features", conn)
    conn.close()

    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True
    )

    pushed = 0
    for _, row in df.iterrows():
        features = row.to_dict()
        cust_id  = features.pop('customer_id')
        features.pop('updated_at', None)
        r.setex(
            f"features:{cust_id}",
            86400,
            json.dumps({k: float(v) if v is not None else 0.0
                       for k, v in features.items()})
        )
        pushed += 1

    r.set('feature_store:last_updated', datetime.now().isoformat())
    r.set('feature_store:row_count', pushed)
    print(f"Pushed {pushed} customers to Redis ✅")

with DAG(
    'etl_pipeline',
    default_args=default_args,
    description='ETL: Validate → Feature Engineering → Redis sync',
    schedule_interval='0 2 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=1),
    tags=['etl', 'features'],
) as dag:

    validate  = PythonOperator(task_id='data_validation',  python_callable=data_validation)
    transform = PythonOperator(task_id='compute_features', python_callable=compute_features)
    sync      = PythonOperator(task_id='sync_to_redis',    python_callable=push_to_redis)

    validate >> transform >> sync
