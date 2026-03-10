from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import psycopg2
import os
from pathlib import Path

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

MODEL_DIR = Path('/opt/airflow/data/processed/models')

def score_all_users(**context):
    df = pd.read_csv('/opt/airflow/data/processed/features_latest.csv')
    lgbm_model = lgb.Booster(model_file=str(MODEL_DIR / 'lgbm_churn.txt'))
    t0_model = joblib.load(MODEL_DIR / 'uplift_t0.pkl')
    t1_model = joblib.load(MODEL_DIR / 'uplift_t1.pkl')
    with open(MODEL_DIR / 'feature_cols.json') as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    churn_probs = lgbm_model.predict(X)

    X_uplift = df[['tenure', 'MonthlyCharges', 'Contract_encoded',
                   'service_count', 'clv_proxy', 'SeniorCitizen',
                   'Partner', 'digital_engagement']].rename(columns={
        'MonthlyCharges': 'monthly_charges',
        'Contract_encoded': 'contract_type',
        'SeniorCitizen': 'senior_citizen',
        'Partner': 'has_partner',
    })

    p_no_treatment = t0_model.predict_proba(X_uplift)[:, 1]
    p_with_treatment = t1_model.predict_proba(X_uplift)[:, 1]
    uplift_scores = p_no_treatment - p_with_treatment
    uplift_pct = pd.Series(uplift_scores).rank(pct=True)

    def assign_segment(pct, uplift, churn):
        if pct >= 0.70 and churn >= 0.50:
            return 'Persuadable'
        elif uplift < -0.10:
            return 'Sleeping_Dog'
        elif churn >= 0.60 and pct < 0.70:
            return 'Lost_Cause'
        else:
            return 'Sure_Thing'

    segments = [assign_segment(p, u, c)
                for p, u, c in zip(uplift_pct, uplift_scores, churn_probs)]

    df_scores = pd.DataFrame({
        'user_id': df.get('customerID', pd.Series(range(len(df)))),
        'churn_probability': churn_probs.round(4),
        'uplift_score': uplift_scores.round(4),
        'segment': segments,
        'clv_proxy': df['clv_proxy'].round(2),
        'scored_at': datetime.now().isoformat(),
    })
    df_scores.to_csv('/opt/airflow/data/processed/batch_scores_latest.csv', index=False)
    print(f"Batch scoring done ✅ — {len(df_scores)} users")
    print(pd.Series(segments).value_counts())
    return len(df_scores)

def save_to_postgres(**context):
    df = pd.read_csv('/opt/airflow/data/processed/batch_scores_latest.csv')
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', 5432),
        dbname=os.getenv('POSTGRES_DB', 'churn_db'),
        user=os.getenv('POSTGRES_USER', 'churn_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'churn_pass'),
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS churn_scores (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50),
            churn_probability FLOAT,
            uplift_score FLOAT,
            segment VARCHAR(50),
            clv_proxy FLOAT,
            scored_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO churn_scores
                (user_id, churn_probability, uplift_score, segment, clv_proxy, scored_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (row['user_id'], row['churn_probability'], row['uplift_score'],
              row['segment'], row['clv_proxy'], row['scored_at']))
    conn.commit()
    cur.close()
    conn.close()
    print(f"Saved {len(df)} scores to PostgreSQL ✅")

with DAG(
    'batch_scoring',
    default_args=default_args,
    description='Nightly batch scoring for all users',
    schedule_interval='0 3 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['scoring', 'batch'],
) as dag:
    score_users = PythonOperator(task_id='score_all_users', python_callable=score_all_users)
    save_postgres = PythonOperator(task_id='save_to_postgres', python_callable=save_to_postgres)
    score_users >> save_postgres
