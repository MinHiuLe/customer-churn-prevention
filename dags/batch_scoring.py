from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import lightgbm as lgb
import joblib
import json
import psycopg2
from psycopg2.extras import execute_batch
import os
from pathlib import Path

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

MODEL_DIR = Path('/opt/airflow/data/processed/models')
FEATURES_CSV = '/opt/airflow/data/processed/features_latest.csv'

def score_all_users(**context):
    # 1. Load Features và Models
    df = pd.read_csv(FEATURES_CSV)
    lgbm_model = lgb.Booster(model_file=str(MODEL_DIR / 'lgbm_churn.txt'))
    t0_model = joblib.load(MODEL_DIR / 'uplift_t0.pkl')
    t1_model = joblib.load(MODEL_DIR / 'uplift_t1.pkl')
    
    try:
        cox_model = joblib.load(MODEL_DIR / 'coxph_model.pkl')
        has_survival = True
    except:
        has_survival = False

    with open(MODEL_DIR / 'feature_cols.json') as f:
        feature_cols = json.load(f)

    # 2. Chấm điểm Churn
    X = df[feature_cols]
    churn_probs = lgbm_model.predict(X)

    # 3. Chấm điểm Uplift (Logic Marketing)
    # Map tên cột cho mô hình Uplift
    X_uplift = df[['tenure', 'MonthlyCharges', 'Contract_encoded', 'service_count', 
                   'clv_proxy', 'SeniorCitizen', 'Partner', 'digital_engagement']].rename(columns={
        'MonthlyCharges': 'monthly_charges', 'Contract_encoded': 'contract_type',
        'SeniorCitizen': 'senior_citizen', 'Partner': 'has_partner',
    })

    p_no_treatment = t0_model.predict_proba(X_uplift)[:, 1]
    p_with_treatment = t1_model.predict_proba(X_uplift)[:, 1]
    uplift_scores = p_no_treatment - p_with_treatment
    uplift_pct = pd.Series(uplift_scores).rank(pct=True)

    def assign_segment(pct, uplift, churn):
        if pct >= 0.70 and churn >= 0.50: return 'Persuadable'
        if uplift < -0.10: return 'Sleeping_Dog'
        if churn >= 0.60: return 'Lost_Cause'
        return 'Sure_Thing'

    segments = [assign_segment(p, u, c) for p, u, c in zip(uplift_pct, uplift_scores, churn_probs)]

    # 4. Chấm điểm Survival (Thời gian còn lại)
    if has_survival:
        # Expected tenure trừ đi tenure hiện tại
        remaining = cox_model.predict_expectation(X) - df['tenure']
        remaining = remaining.apply(lambda x: max(0, round(x, 1)))
    else:
        remaining = [-1] * len(df)

    # 5. Tổng hợp kết quả chấm điểm
    df_results = pd.DataFrame({
        'user_id': df['customerID'],
        'churn_probability': churn_probs.round(4),
        'uplift_score': uplift_scores.round(4),
        'segment': segments,
        'expected_tenure_remaining': remaining,
        'clv_proxy': df['clv_proxy'].round(2),
        'scored_at': datetime.now().isoformat()
    })
    
    df_results.to_csv('/opt/airflow/data/processed/batch_scores_latest.csv', index=False)

from airflow.providers.postgres.hooks.postgres import PostgresHook
from psycopg2.extras import execute_batch

def save_to_postgres(**context):
    df = pd.read_csv('/opt/airflow/data/processed/batch_scores_latest.csv')
    
    # 1. Gọi Hook bằng ID kết nối sẽ cấu hình trên Web UI
    hook = PostgresHook(postgres_conn_id='churn_postgres_conn')
    conn = hook.get_conn()
    cur = conn.cursor()
    
    # 2. Logic tạo bảng (giữ nguyên)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS churn_scores (
            user_id VARCHAR(50) PRIMARY KEY,
            churn_probability FLOAT,
            uplift_score FLOAT,
            segment VARCHAR(50),
            expected_tenure_remaining FLOAT,
            clv_proxy FLOAT,
            scored_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # 3. Logic UPSERT tốc độ cao (giữ nguyên)
    insert_query = """
        INSERT INTO churn_scores (user_id, churn_probability, uplift_score, segment, expected_tenure_remaining, clv_proxy, scored_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            churn_probability = EXCLUDED.churn_probability,
            uplift_score = EXCLUDED.uplift_score,
            segment = EXCLUDED.segment,
            expected_tenure_remaining = EXCLUDED.expected_tenure_remaining,
            clv_proxy = EXCLUDED.clv_proxy,
            scored_at = EXCLUDED.scored_at;
    """
    
    data = list(df.itertuples(index=False, name=None))
    execute_batch(cur, insert_query, data)
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Đã UPSERT {len(df)} bản ghi vào Postgres thông qua PostgresHook an toàn.")

with DAG(
    'batch_scoring',
    default_args=default_args,
    schedule_interval='0 3 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['inference']
) as dag:
    PythonOperator(task_id='score_users', python_callable=score_all_users) >> \
    PythonOperator(task_id='save_postgres', python_callable=save_to_postgres)