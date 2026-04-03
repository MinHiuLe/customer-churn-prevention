# Customer Churn Prevention System

A production-grade MLOps pipeline for predicting, explaining, and preventing customer churn. The system combines churn probability scoring, survival analysis, and uplift modeling to generate actionable retention recommendations — not just predictions.

---

## Business Problem

Customer churn is one of the highest-cost problems in subscription businesses. Acquiring a new customer costs 5–10x more than retaining an existing one. This system addresses three questions that matter to business teams:

1. **Who is likely to churn?** — LightGBM classifier with tuned decision threshold
2. **How long do we have?** — CoxPH survival model predicting time-to-churn
3. **Will an intervention actually help?** — Uplift T-Learner distinguishing customers worth targeting from those who will leave regardless

---

## Architecture

```
Faker DAG (daily, 01:00)
    └── generates synthetic raw customers → Postgres

ETL Pipeline (daily, 02:00)
    └── cleans, engineers features → features_latest.csv

Batch Scoring (daily, 03:00)
    └── churn probability + uplift score + survival time → churn_scores table

Retrain Pipeline (daily, 04:00)
    └── drift check → DVC hash validation → LightGBM retrain → Champion/Challenger → MLflow
```

All pipelines are orchestrated by Apache Airflow. Model artifacts are tracked in MLflow with a Champion/Challenger promotion system. Data integrity is enforced via DVC hash validation before every retrain.

---

## Models

| Model | Purpose | Key Metric |
|---|---|---|
| LightGBM (GBDT) | Churn probability | AUC-ROC |
| CoxPH | Time-to-churn survival curve | C-index |
| T-Learner (Uplift) | Estimate treatment effect of retention offer | AUUC |

### Handling Class Imbalance

Churn datasets are inherently imbalanced (typically 15–30% positive class). This project applies three complementary techniques:

- `scale_pos_weight` in LightGBM to penalize false negatives
- SMOTE oversampling during training
- Threshold tuning via Precision-Recall curve to maximize F1 on the minority class

### Customer Segments (Uplift)

| Segment | Condition | Recommended Action |
|---|---|---|
| Persuadable | High churn risk + positive uplift | Send retention voucher |
| Sleeping Dog | Negative uplift (offer increases churn) | Suppress — do not contact |
| Lost Cause | Very high churn risk, uplift ineffective | Escalate to CS team |
| Sure Thing | Low churn risk | No action needed |

---

## Stack

**Orchestration:** Apache Airflow  
**ML:** LightGBM, scikit-learn (CoxPH via lifelines), Causalml (T-Learner)  
**Experiment Tracking:** MLflow  
**Data Versioning:** DVC  
**Serving:** FastAPI  
**Monitoring:** Evidently (data drift detection)  
**Dashboard:** Streamlit  
**Database:** PostgreSQL  
**Containerization:** Docker Compose

---

## API Endpoints

All endpoints are served by FastAPI. Interactive docs available at `/docs`.

### `POST /predict`

Returns churn probability and survival curve for a single customer.

```json
{
  "churn_probability": 0.7241,
  "churn_predicted": 1,
  "survival": {
    "p_active_7d": 0.8901,
    "p_active_30d": 0.6134,
    "p_active_90d": 0.2201
  }
}
```

### `POST /explain`

Returns top SHAP drivers pushing the customer toward or away from churn.

```json
{
  "base_value": 0.2814,
  "top_churn_drivers": {
    "recency_risk": 0.1823,
    "Contract_encoded": 0.1204
  },
  "top_churn_protectors": {
    "clv_proxy": -0.0912,
    "tenure": -0.0741
  }
}
```

### `POST /segment`

Returns uplift score and retention action recommendation.

```json
{
  "uplift_score": 0.3120,
  "segment": "Persuadable",
  "recommended_action": "Send voucher 20%",
  "priority": "HIGH"
}
```

---

## Dashboard Pages

| Page | Contents |
|---|---|
| Overview | Customer count, churn rate, CLV distribution, segment breakdown |
| Customer Segments | Segment table pulled from `churn_scores`, filterable |
| Real-time Prediction | Manual input form → live API call → churn prob + survival chart + segment |
| Model Performance | Live champion metrics from MLflow, SHAP feature importance, model registry table |
| Drift Monitor | Evidently drift score gauge, drifted column count, drift injection tool for demos |

---

## Pipeline Details

### ETL Pipeline

Reads raw customer records from Postgres, applies cleaning (handles whitespace in `total_charges`), encodes categorical features, and engineers derived features including:

- `recency_risk` = 1 / (tenure + 1)
- `clv_proxy` = total_charges × (1 − recency_risk)
- `service_count` = sum of active service columns
- `digital_engagement` = paperless billing + digital payment method
- `contract_stability` = contract_encoded × tenure

### Retrain Pipeline

Triggered daily at 04:00, but only retrains if a drift flag is present. Before training, the pipeline validates that the feature CSV matches its DVC-tracked MD5 hash — preventing retraining on unversioned data ("Ghost Data" detection).

If the new model outperforms the current champion in AUC-ROC, it is promoted automatically via `champion_challenger()` and saved to the model directory. All runs, metrics, signatures, and the associated DVC commit hash are logged to MLflow.

### Drift Detection

Evidently compares the current feature distribution against a reference baseline. If the drift score exceeds the threshold (0.3), a flag file is written to disk, which the Retrain Pipeline picks up on its next scheduled run.

---

## Project Structure

```
.
├── dags/
│   ├── faker_dag.py          # Synthetic data generation
│   ├── etl_pipeline.py       # Feature engineering
│   ├── batch_scoring.py      # Daily batch inference
│   └── retrain_pipeline.py   # Drift-triggered retraining
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI serving layer
│   ├── dashboard/
│   │   └── app.py            # Streamlit dashboard
│   └── mlops/
│       └── model_registry.py # Champion/Challenger logic
├── data/
│   └── processed/
│       ├── features_latest.csv
│       ├── features_latest.csv.dvc
│       └── models/
├── docker-compose.yml
└── .env
```

---

## Setup

**Requirements:** Docker, Docker Compose, Python 3.10+

```bash
git clone https://github.com/MinHiuLe/customer-churn-prevention.git
cd customer-churn-prevention

# Configure environment
cp config/.env.example config/.env
# Fill in POSTGRES credentials, MLFLOW_TRACKING_URI, etc.

# Start all services
docker compose up -d

# Verify health
docker compose ps
curl http://localhost:8000/health
```

Airflow UI: `http://localhost:8080`  
MLflow UI: `http://localhost:5000`  
Dashboard: `http://localhost:8501`  
API docs: `http://localhost:8000/docs`

---

## Key Design Decisions

**No Kafka.** News and customer data at this frequency does not justify the operational complexity of a message broker. Airflow scheduled DAGs with PostgreSQL provide sufficient throughput and are far easier to maintain.

**DVC over full dataset versioning.** Only the MD5 hash of the feature CSV is tracked in Git via DVC. This gives reproducibility guarantees without storing large files in the repository.

**Threshold tuning over raw probability.** The classification threshold (default 0.446) is computed per-run via Precision-Recall curve optimization and logged to MLflow — it is not fixed in code.

**Uplift modeling over pure churn ranking.** Ranking customers by churn probability alone leads to wasted spend on "Lost Cause" customers and suppresses intervention for "Sure Thing" customers. Uplift modeling estimates the causal effect of a retention offer, making targeting materially more efficient.

---

## Author

**MinHiuLe** — Data Science & AI Engineering  
[GitHub](https://github.com/MinHiuLe)
