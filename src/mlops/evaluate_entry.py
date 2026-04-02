"""
MLflow Projects entry point — evaluate champion model
"""
import argparse
import json
import mlflow
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score,
                             classification_report)

load_dotenv()

sys_path = str(Path(__file__).parent.parent.parent)
import sys
sys.path.insert(0, sys_path)
from src.mlops.model_registry import load_champion, get_registry_summary

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate champion model')
    parser.add_argument('--model_version', type=str, default='champion')
    return parser.parse_args()

def main():
    args = parse_args()

    MLFLOW_URI = (
        os.getenv('MLFLOW_TRACKING_URI') or
        'https://dagshub.com/MinHiuLe/customer-churn-prevention.mlflow'
    )
    mlflow.set_tracking_uri(MLFLOW_URI)

    username = os.getenv('MLFLOW_TRACKING_USERNAME')
    password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    if username and password:
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password

    DATA_DIR  = Path('data/processed')
    MODEL_DIR = Path('data/processed/models')

    df = pd.read_csv(DATA_DIR / 'features.csv')
    with open(MODEL_DIR / 'feature_cols.json') as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    y = df['Churn_binary']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Loading {args.model_version} model from Registry...")
    model  = load_champion()
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.446).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)

    print(f"\n=== Champion Model Evaluation ===")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"F1:      {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Stay', 'Churn']))

    print(f"\n=== Registry Summary ===")
    summary = get_registry_summary()
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
