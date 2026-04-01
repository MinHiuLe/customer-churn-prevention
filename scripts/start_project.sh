#!/bin/bash
set -e

PROJECT_ROOT="/mnt/c/Users/Lmhie/custormer_churn"
cd $PROJECT_ROOT
source custormer_churn/bin/activate

echo "🚀 Starting Customer Churn Prevention System..."

# ── 1. Docker services ────────────────────────────────────────
echo "📦 Starting Docker services..."
docker compose up -d postgres redis mlflow airflow

# Chờ postgres healthy thay vì sleep cứng
echo "⏳ Waiting for PostgreSQL..."
scripts/wait-for-it.sh localhost:5432 --timeout=60 --strict -- echo "✅ PostgreSQL ready"

echo "⏳ Waiting for MLflow..."
scripts/wait-for-it.sh localhost:5000 --timeout=60 --strict -- echo "✅ MLflow ready"

# ── 2. FastAPI ────────────────────────────────────────────────
if ! lsof -i:8000 > /dev/null 2>&1; then
    echo "🌐 Starting FastAPI..."
    uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload \
        > logs/fastapi.log 2>&1 &
    echo $! > /tmp/fastapi.pid
    scripts/wait-for-it.sh localhost:8000 --timeout=30 -- echo "✅ FastAPI ready"
else
    echo "🌐 FastAPI already running on :8000"
fi

# ── 3. Batch scoring nếu chưa có data hôm nay ────────────────
SCORES_FILE="data/processed/batch_scores_latest.csv"
if [ ! -f "$SCORES_FILE" ] || [ $(find "$SCORES_FILE" -mtime +1 | wc -l) -gt 0 ]; then
    echo "🔄 Running batch scoring..."
    python src/models/batch_scoring_local.py
else
    echo "✅ Batch scores up to date"
fi

# ── 4. Decision Engine ────────────────────────────────────────
echo "⚙️  Running Decision Engine..."
python src/decision_engine.py

# ── 5. Streamlit ──────────────────────────────────────────────
if ! lsof -i:8501 > /dev/null 2>&1; then
    echo "📱 Starting Streamlit Dashboard..."
    streamlit run src/dashboard/app.py \
        --server.port 8501 \
        --server.headless true \
        > logs/streamlit.log 2>&1 &
    echo $! > /tmp/streamlit.pid
    scripts/wait-for-it.sh localhost:8501 --timeout=30 -- echo "✅ Streamlit ready"
else
    echo "📱 Streamlit already running on :8501"
fi

echo ""
echo "✅ All services started!"
echo "   MLflow:    http://localhost:5000"
echo "   FastAPI:   http://localhost:8000/docs"
echo "   Airflow:   http://localhost:8080  (admin/admin)"
echo "   Dashboard: http://localhost:8501"
