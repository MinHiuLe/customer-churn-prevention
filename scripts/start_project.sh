#!/bin/bash
echo "🚀 Starting Customer Churn Prevention System..."

PROJECT_ROOT="/mnt/c/Users/Lmhie/custormer_churn"
cd $PROJECT_ROOT
source custormer_churn/bin/activate

# 1. Docker services
echo "📦 Starting Docker services..."
docker compose up -d postgres redis mlflow airflow
sleep 15

# 2. FastAPI
echo "🌐 Starting FastAPI..."
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload &
echo $! > /tmp/fastapi.pid
sleep 3

# 3. MLflow local
echo "📊 Starting MLflow..."
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///${PROJECT_ROOT}/mlflow_artifacts/mlflow.db \
  --default-artifact-root ${PROJECT_ROOT}/mlflow_artifacts &
echo $! > /tmp/mlflow.pid
sleep 3

# 4. Batch scoring (nếu chưa có file hoặc file cũ hơn 1 ngày)
SCORES_FILE="data/processed/batch_scores_latest.csv"
if [ ! -f "$SCORES_FILE" ] || [ $(find "$SCORES_FILE" -mtime +1 | wc -l) -gt 0 ]; then
  echo "🔄 Running batch scoring..."
  python dags/batch_scoring.py 2>/dev/null || python3 << 'PYEOF'
import pandas as pd, numpy as np, lightgbm as lgb, joblib, json
from datetime import datetime
from pathlib import Path

MODEL_DIR = Path('data/processed/models')
df = pd.read_csv('data/processed/features.csv')
lgbm_model = lgb.Booster(model_file=str(MODEL_DIR / 'lgbm_churn.txt'))
t0_model = joblib.load(MODEL_DIR / 'uplift_t0.pkl')
t1_model = joblib.load(MODEL_DIR / 'uplift_t1.pkl')
with open(MODEL_DIR / 'feature_cols.json') as f:
    feature_cols = json.load(f)

X = df[feature_cols]
churn_probs = lgbm_model.predict(X)
X_uplift = df[['tenure','MonthlyCharges','Contract_encoded','service_count',
               'clv_proxy','SeniorCitizen','Partner','digital_engagement']].rename(columns={
    'MonthlyCharges':'monthly_charges','Contract_encoded':'contract_type',
    'SeniorCitizen':'senior_citizen','Partner':'has_partner'})
p_no = t0_model.predict_proba(X_uplift)[:,1]
p_with = t1_model.predict_proba(X_uplift)[:,1]
uplift_scores = p_no - p_with
uplift_pct = pd.Series(uplift_scores).rank(pct=True)

def seg(pct, u, c):
    if pct >= 0.70 and c >= 0.50: return 'Persuadable'
    elif u < -0.10: return 'Sleeping_Dog'
    elif c >= 0.60 and pct < 0.70: return 'Lost_Cause'
    else: return 'Sure_Thing'

segments = [seg(p,u,c) for p,u,c in zip(uplift_pct, uplift_scores, churn_probs)]
pd.DataFrame({
    'user_id': df.get('customerID', pd.Series(range(len(df)))),
    'churn_probability': churn_probs.round(4),
    'uplift_score': uplift_scores.round(4),
    'segment': segments,
    'clv_proxy': df['clv_proxy'].round(2),
    'scored_at': datetime.now().isoformat(),
}).to_csv('data/processed/batch_scores_latest.csv', index=False)
print(f"Batch scores saved ✅")
PYEOF
fi

# 5. Decision Engine
echo "⚙️  Running Decision Engine..."
python src/decision_engine.py > /dev/null 2>&1
echo "Decision Engine done ✅"

# 6. Streamlit
if ! lsof -i:8501 > /dev/null 2>&1; then
  echo "📱 Starting Streamlit Dashboard..."
  streamlit run src/dashboard/app.py \
    --server.port 8501 \
    --server.headless true &
  echo $! > /tmp/streamlit.pid
else
  echo "📱 Streamlit already running on :8501"
fi

echo ""
echo "✅ All services started!"
echo "   MLflow:    http://localhost:5000"
echo "   FastAPI:   http://localhost:8000/docs"
echo "   Airflow:   http://localhost:8080  (admin/admin)"
echo "   Dashboard: http://localhost:8501"
