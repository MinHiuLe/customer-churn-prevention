cd /mnt/c/Users/Lmhie/custormer_churn
source .env

echo "================================================"
echo "STEP 1 — DOCKER SERVICES"
echo "================================================"
docker compose ps

echo ""
echo "================================================"
echo "STEP 2 — SERVICE ENDPOINTS"
echo "================================================"
curl -s http://localhost:8000/health && echo " ← FastAPI OK" || echo "❌ FastAPI DOWN"
curl -s http://localhost:5000/health && echo " ← MLflow OK" || echo "❌ MLflow DOWN"
curl -s http://localhost:8501 > /dev/null && echo "✅ Streamlit OK" || echo "❌ Streamlit DOWN"
curl -s http://localhost:8080 > /dev/null && echo "✅ Airflow OK" || echo "❌ Airflow DOWN"

echo ""
echo "================================================"
echo "STEP 3 — AIRFLOW DAG RUNS"
echo "================================================"
docker compose exec airflow airflow dags list-runs --dag-id faker_data_generator
echo ""
docker compose exec airflow airflow dags list-runs --dag-id etl_pipeline
echo ""
docker compose exec airflow airflow dags list-runs --dag-id batch_scoring
echo ""
docker compose exec airflow airflow dags list-runs --dag-id retrain_pipeline

echo ""
echo "================================================"
echo "STEP 4 — POSTGRESQL DATA"
echo "================================================"
docker compose exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT
  'customers'        AS table_name, COUNT(*) AS rows, MAX(created_at) AS latest FROM customers
UNION ALL
SELECT
  'customer_features', COUNT(*), MAX(updated_at) FROM customer_features
UNION ALL
SELECT
  'churn_scores',      COUNT(*), MAX(created_at) FROM churn_scores
UNION ALL
SELECT
  'action_logs',       COUNT(*), MAX(executed_at) FROM action_logs;
"

echo ""
echo "================================================"
echo "STEP 5 — REDIS FEATURE STORE"
echo "================================================"
docker compose exec redis redis-cli DBSIZE
docker compose exec redis redis-cli GET feature_store:last_updated
docker compose exec redis redis-cli GET feature_store:row_count

echo ""
echo "================================================"
echo "STEP 6 — ML MODELS"
echo "================================================"
ls -lh data/processed/models/
python3 -c "
import lightgbm as lgb, joblib, json
from pathlib import Path
MODEL_DIR = Path('data/processed/models')
m = lgb.Booster(model_file=str(MODEL_DIR / 'lgbm_churn.txt'))
print(f'✅ LightGBM loaded — trees: {m.num_trees()}')
cph = joblib.load(MODEL_DIR / 'coxph_model.pkl')
print(f'✅ CoxPH loaded')
t0 = joblib.load(MODEL_DIR / 'uplift_t0.pkl')
t1 = joblib.load(MODEL_DIR / 'uplift_t1.pkl')
print(f'✅ T-Learner T0/T1 loaded')
with open(MODEL_DIR / 'feature_cols.json') as f:
    cols = json.load(f)
print(f'✅ Feature cols: {len(cols)} features')
"

echo ""
echo "================================================"
echo "STEP 7 — FASTAPI ENDPOINTS"
echo "================================================"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure":3,"SeniorCitizen":0,"MonthlyCharges":95.0,
    "TotalCharges":285.0,"Partner":0,"Dependents":0,
    "PhoneService":1,"PaperlessBilling":1,"MultipleLines":0,
    "OnlineSecurity":0,"OnlineBackup":0,"DeviceProtection":0,
    "TechSupport":0,"StreamingTV":1,"StreamingMovies":1,
    "Contract_encoded":0,"InternetService_encoded":2,
    "PaymentMethod_encoded":0,"recency_risk":0.25,
    "service_count":3,"monetary_value":285.0,
    "monthly_to_total_ratio":0.33,"charge_per_month":31.0,
    "clv_proxy":213.75,"is_high_value":0,
    "contract_stability":0,"digital_engagement":1
  }' | python3 -m json.tool

echo ""
echo "================================================"
echo "STEP 8 — BATCH SCORES & DECISION ENGINE"
echo "================================================"
python3 -c "
import pandas as pd
from pathlib import Path

scores = Path('data/processed/batch_scores_latest.csv')
action = Path('data/processed/action_plan.csv')
targeted = Path('data/processed/targeted_users.csv')

if scores.exists():
    df = pd.read_csv(scores)
    print(f'✅ Batch scores: {len(df)} users — scored at {df.scored_at.max()}')
    print(f'   Segments: {df.segment.value_counts().to_dict()}')
else:
    print('❌ No batch scores found')

if action.exists():
    df = pd.read_csv(action)
    print(f'✅ Action plan: {len(df)} users')
else:
    print('❌ No action plan found')

if targeted.exists():
    df = pd.read_csv(targeted)
    print(f'✅ Targeted users: {len(df)}')
    print(f'   Total cost: \${df.action_cost.sum():,.0f}')
    print(f'   Expected CLV saved: \${df.expected_clv_saved.sum():,.0f}')
else:
    print('❌ No targeted users found')
"

echo ""
echo "================================================"
echo "STEP 9 — DRIFT STATUS"
echo "================================================"
python3 -c "
import json
from pathlib import Path

summary = Path('data/processed/drift_reports/drift_summary_latest.json')
flag = Path('data/processed/drift_detected.flag')

if summary.exists():
    with open(summary) as f:
        d = json.load(f)
    print(f'✅ Last drift check: {d[\"timestamp\"]}')
    print(f'   Drift score: {d[\"drift_score\"]:.3f} (threshold: {d[\"threshold\"]})')
    print(f'   Status: {\"⚠️  DRIFT\" if d[\"drift_detected\"] else \"✅ Stable\"}')
else:
    print('❌ No drift report found')

print(f'   Drift flag: {\"⚠️  SET\" if flag.exists() else \"✅ Clear\"}')
"

echo ""
echo "================================================"
echo "STEP 10 — MLFLOW RUNS"
echo "================================================"
python3 -c "
import mlflow, os
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri("http://localhost:5000"'))
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('churn-prediction')
if exp:
    runs = client.search_runs(exp.experiment_id, order_by=['start_time DESC'], max_results=5)
    print(f'✅ MLflow experiment: {exp.name} — {len(runs)} recent runs')
    for r in runs:
        promoted = r.data.tags.get('promoted', 'N/A')
        print(f'   {r.info.run_name:<25} | {dict(r.data.metrics)} | promoted={promoted}')
else:
    print('❌ No MLflow experiment found')
"

echo ""
echo "================================================"
echo "✅ HEALTH CHECK COMPLETE"
echo "================================================"
