"""
MLflow Projects entry point — train model với params từ CLI
Reproducible: bất kỳ ai cũng chạy được với cùng kết quả
"""
import argparse
import json
import mlflow
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

sys_path = str(Path(__file__).parent.parent.parent)
import sys
sys.path.insert(0, sys_path)
from src.mlops.model_registry import (
    train_and_log, register_model, champion_challenger
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train churn model')
    parser.add_argument('--num_leaves',       type=int,   default=31)
    parser.add_argument('--learning_rate',    type=float, default=0.05)
    parser.add_argument('--feature_fraction', type=float, default=0.9)
    parser.add_argument('--bagging_fraction', type=float, default=0.8)
    parser.add_argument('--bagging_freq',     type=int,   default=5)
    parser.add_argument('--threshold',        type=float, default=0.446)
    parser.add_argument('--test_size',        type=float, default=0.2)
    return parser.parse_args()

def main():
    args = parse_args()

    # Ưu tiên env var, fallback về DagsHub
    MLFLOW_URI = (
        os.getenv('MLFLOW_TRACKING_URI') or
        os.getenv('MLFLOW_TRACKING_URI_DEFAULT') or
        'https://dagshub.com/MinHiuLe/customer-churn-prevention.mlflow'
    )
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Set DagsHub credentials nếu có
    username = os.getenv('MLFLOW_TRACKING_USERNAME')
    password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    if username and password:
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
    mlflow.set_experiment('churn-prediction')

    # Load data
    DATA_DIR   = Path('data/processed')
    MODEL_DIR  = Path('data/processed/models')

    df = pd.read_csv(DATA_DIR / 'features.csv')
    with open(MODEL_DIR / 'feature_cols.json') as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    y = df['Churn_binary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    params = {
        'objective':        'binary',
        'metric':           'auc',
        'num_leaves':       args.num_leaves,
        'learning_rate':    args.learning_rate,
        'feature_fraction': args.feature_fraction,
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq':     args.bagging_freq,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state':     42,
        'verbose':          -1,
    }

    print(f"Training with params: {params}")

    # Train + log
    run_id, auc, model = train_and_log(
        X_train, X_test, y_train, y_test,
        params,
        run_name=f'mlproject_lr{args.learning_rate}_leaves{args.num_leaves}'
    )

    # Register + Champion/Challenger
    version  = register_model(run_id, version_alias='staging')
    promoted = champion_challenger(version, auc)

    if promoted:
        model.save_model(str(MODEL_DIR / 'lgbm_churn.txt'))
        print(f"Model promoted to champion ✅ AUC={auc:.4f}")
    else:
        print(f"Model archived ❌ AUC={auc:.4f} — champion unchanged")

    print(f"\nRun complete — view at: {MLFLOW_URI}")

if __name__ == '__main__':
    main()
