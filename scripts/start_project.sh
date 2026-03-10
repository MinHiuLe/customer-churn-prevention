#!/bin/bash
echo "🚀 Starting Customer Churn Prevention System..."

PROJECT_ROOT="/mnt/c/Users/Lmhie/custormer_churn"
cd $PROJECT_ROOT

# 1. Activate venv
source custormer_churn/bin/activate

# 2. Start Docker services
docker compose up -d postgres redis mlflow
sleep 10

# 3. Start Airflow
docker compose up -d airflow
sleep 30

# 4. Start FastAPI
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload &
echo $! > /tmp/fastapi.pid

# 5. Start MLflow (local)
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///${PROJECT_ROOT}/mlflow_artifacts/mlflow.db \
  --default-artifact-root ${PROJECT_ROOT}/mlflow_artifacts &
echo $! > /tmp/mlflow.pid

echo ""
echo "✅ All services started!"
echo "   MLflow:  http://localhost:5000"
echo "   FastAPI: http://localhost:8000/docs"
echo "   Airflow: http://localhost:8080 (admin/admin)"
