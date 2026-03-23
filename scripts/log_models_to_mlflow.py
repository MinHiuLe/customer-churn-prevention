import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment('churn-prediction')

with mlflow.start_run(run_name='lightgbm_baseline'):
    mlflow.log_params({'objective': 'binary', 'num_leaves': 31,
                       'learning_rate': 0.05, 'scale_pos_weight': 2.77})
    mlflow.log_metrics({'auc_roc': 0.8445, 'f1': 0.6365,
                        'precision': 0.5742, 'recall': 0.7139,
                        'optimal_threshold': 0.446})
    mlflow.set_tag('status', 'production')
    mlflow.set_tag('model_path', 'data/processed/models/lgbm_churn.txt')
    print("LightGBM logged ✅")

with mlflow.start_run(run_name='coxph_survival'):
    mlflow.log_metric('c_index', 0.9415)
    mlflow.set_tag('status', 'production')
    mlflow.set_tag('model_path', 'data/processed/models/coxph_model.pkl')
    print("CoxPH logged ✅")

with mlflow.start_run(run_name='uplift_t_learner'):
    mlflow.log_metric('auuc', 0.2014)
    mlflow.log_metric('segment_accuracy', 0.73)
    mlflow.set_tag('status', 'production')
    mlflow.set_tag('model_path', 'data/processed/models/uplift_t0.pkl')
    print("T-Learner logged ✅")

print("\nAll models logged ✅ — http://localhost:5000")
