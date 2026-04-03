from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

CSV_PATH = '/opt/airflow/data/processed/features_latest.csv'

def run_etl_logic(**context):
    # 1. Trích xuất dữ liệu từ Postgres
    hook = PostgresHook(postgres_conn_id='churn_postgres_conn')
    df = hook.get_pandas_df("SELECT * FROM customers")

    if df.empty:
        print("⚠️ Database rỗng!")
        return

    # 2. Làm sạch (Logic từ Notebook 01)
    df['total_charges'] = pd.to_numeric(df['total_charges'].replace(" ", np.nan), errors='coerce').fillna(0)

    # 3. Feature Engineering (Logic từ Notebook 02)
    binary_map = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
    for col in ['partner', 'dependents', 'phone_service', 'paperless_billing', 'churn', 'gender']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    service_cols = ['multiple_lines', 'online_security', 'online_backup', 
                    'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    for col in service_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    df['InternetService_encoded'] = df['internet_service'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2}).fillna(2)
    df['Contract_encoded'] = df['contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2}).fillna(0)
    df['PaymentMethod_encoded'] = df['payment_method'].map({
        'Electronic check': 0, 'Mailed check': 1, 'Bank transfer': 2, 'Credit card': 3
    }).fillna(0)

    # Các biến phái sinh quan trọng
    df['recency_risk'] = 1 / (df['tenure'] + 1)
    df['service_count'] = df[service_cols].sum(axis=1)
    df['charge_per_month'] = df['monthly_charges'] / (df['tenure'] + 1)
    df['clv_proxy'] = df['total_charges'] * (1 - df['recency_risk'])
    df['contract_stability'] = df['Contract_encoded'] * df['tenure']
    df['monetary_value'] = df['total_charges']
    df['monthly_to_total_ratio'] = df['monthly_charges'] / (df['total_charges'] + 1)
    df['is_high_value'] = (df['clv_proxy'] > df['clv_proxy'].quantile(0.75)).astype(int)
    df['digital_engagement'] = (df['paperless_billing'] == 1).astype(int) + (df['PaymentMethod_encoded'] >= 2).astype(int)

    # Rename cột để đồng bộ
    df = df.rename(columns={
        'customer_id': 'customerID', 'senior_citizen': 'SeniorCitizen',
        'partner': 'Partner', 'dependents': 'Dependents',
        'phone_service': 'PhoneService', 'multiple_lines': 'MultipleLines',
        'online_security': 'OnlineSecurity', 'online_backup': 'OnlineBackup',
        'device_protection': 'DeviceProtection', 'tech_support': 'TechSupport',
        'streaming_tv': 'StreamingTV', 'streaming_movies': 'StreamingMovies',
        'paperless_billing': 'PaperlessBilling', 'monthly_charges': 'MonthlyCharges',
        'total_charges': 'TotalCharges', 'churn': 'Churn_binary'
    })

    # 4. Lưu ra file CSV "nguyên liệu"
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ ETL hoàn tất. Đã lưu features cho {len(df)} khách hàng.")

with DAG(
    'etl_pipeline',
    default_args=default_args,
    schedule_interval='0 2 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['data_eng']
) as dag:
    PythonOperator(task_id='run_etl', python_callable=run_etl_logic)