#!/bin/bash
PROJECT_ROOT="/mnt/c/Users/Lmhie/custormer_churn"
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///${PROJECT_ROOT}/mlflow_artifacts/mlflow.db \
  --default-artifact-root ${PROJECT_ROOT}/mlflow_artifacts
