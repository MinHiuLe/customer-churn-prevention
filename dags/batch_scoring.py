from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import lightgbm as lgb
import joblib
import json
import os
from pathlib import Path
from psycopg2.extras import execute_batch

# Shared segment logic — single source of truth for batch AND real-time
from src.mlops.segment import classify_segment

default_args = {
    'owner': 'churn_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

MODEL_DIR    = Path('/opt/airflow/data/processed/models')
FEATURES_CSV = '/opt/airflow/data/processed/features_latest.csv'
SCORES_CSV   = '/opt/airflow/data/processed/batch_scores_latest.csv'


def score_all_users(**context):
    # 1. Load features
    df = pd.read_csv(FEATURES_CSV)

    # 2. Load models
    lgbm_model = lgb.Booster(model_file=str(MODEL_DIR / 'lgbm_churn.txt'))
    t0_model   = joblib.load(MODEL_DIR / 'uplift_t0.pkl')
    t1_model   = joblib.load(MODEL_DIR / 'uplift_t1.pkl')

    try:
        cox_model   = joblib.load(MODEL_DIR / 'coxph_model.pkl')
        has_survival = True
    except Exception:
        has_survival = False

    with open(MODEL_DIR / 'feature_cols.json') as f:
        feature_cols = json.load(f)

    # 3. Load threshold được tính tại lúc retrain — không hardcode
    config_path = MODEL_DIR / 'model_config.json'
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        threshold = model_config.get('churn_threshold', 0.5)
    else:
        threshold = 0.5
        print(f"⚠️ model_config.json not found. Using default threshold={threshold}")

    # 4. Chấm điểm Churn
    X          = df[feature_cols]
    churn_probs = lgbm_model.predict(X)

    # 5. Chấm điểm Uplift
    X_uplift = df[[
        'tenure', 'MonthlyCharges', 'Contract_encoded', 'service_count',
        'clv_proxy', 'SeniorCitizen', 'Partner', 'digital_engagement'
    ]].rename(columns={
        'MonthlyCharges':    'monthly_charges',
        'Contract_encoded':  'contract_type',
        'SeniorCitizen':     'senior_citizen',
        'Partner':           'has_partner',
    })

    p_no_treatment   = t0_model.predict_proba(X_uplift)[:, 1]
    p_with_treatment = t1_model.predict_proba(X_uplift)[:, 1]
    uplift_scores    = p_no_treatment - p_with_treatment

    # classify_segment dùng cùng logic với /segment endpoint trong FastAPI
    segments = [
        classify_segment(float(u), float(c))
        for u, c in zip(uplift_scores, churn_probs)
    ]

    # 6. Chấm điểm Survival
    if has_survival:
        remaining = cox_model.predict_expectation(X) - df['tenure']
        remaining = remaining.apply(lambda x: max(0, round(x, 1)))
    else:
        remaining = [-1] * len(df)

    # 7. Tổng hợp kết quả
    df_results = pd.DataFrame({
        'user_id':                  df['customerID'],
        'churn_probability':        churn_probs.round(4),
        'churn_predicted':          (churn_probs >= threshold).astype(int),
        'uplift_score':             uplift_scores.round(4),
        'segment':                  segments,
        'expected_tenure_remaining': remaining,
        'clv_proxy':                df['clv_proxy'].round(2),
        'scored_at':                datetime.now().isoformat(),
    })

    df_results.to_csv(SCORES_CSV, index=False)
    print(f"✅ Scored {len(df)} users with threshold={threshold:.4f}")


def save_to_postgres(**context):
    df   = pd.read_csv(SCORES_CSV)
    hook = PostgresHook(postgres_conn_id='churn_postgres_conn')
    conn = hook.get_conn()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS churn_scores (
            user_id                   VARCHAR(50) PRIMARY KEY,
            churn_probability         FLOAT,
            churn_predicted           INT,
            uplift_score              FLOAT,
            segment                   VARCHAR(50),
            expected_tenure_remaining FLOAT,
            clv_proxy                 FLOAT,
            scored_at                 TIMESTAMP,
            created_at                TIMESTAMP DEFAULT NOW()
        )
    """)

    insert_query = """
        INSERT INTO churn_scores (
            user_id, churn_probability, churn_predicted, uplift_score,
            segment, expected_tenure_remaining, clv_proxy, scored_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            churn_probability         = EXCLUDED.churn_probability,
            churn_predicted           = EXCLUDED.churn_predicted,
            uplift_score              = EXCLUDED.uplift_score,
            segment                   = EXCLUDED.segment,
            expected_tenure_remaining = EXCLUDED.expected_tenure_remaining,
            clv_proxy                 = EXCLUDED.clv_proxy,
            scored_at                 = EXCLUDED.scored_at;
    """

    data = list(df.itertuples(index=False, name=None))
    execute_batch(cur, insert_query, data)

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Upserted {len(df)} records into churn_scores via PostgresHook.")


with DAG(
    'batch_scoring',
    default_args=default_args,
    schedule_interval='0 3 * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['inference'],
) as dag:
    PythonOperator(task_id='score_users',   python_callable=score_all_users) >> \
    PythonOperator(task_id='save_postgres', python_callable=save_to_postgres)