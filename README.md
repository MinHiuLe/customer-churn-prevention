# 🎯 Customer Churn Prevention System

> End-to-end MLOps system for customer churn prevention with real business impact.
> **Core question:** *"With $5,000/month budget, which customers should we target to maximize CLV retained?"*

---

## 📊 Results

| Model | Metric | Score |
|---|---|---|
| LightGBM Churn Classifier | AUC-ROC | **0.8445** |
| CoxPH Survival Analysis | C-index | **0.9415** |
| T-Learner Uplift Model | AUUC | **0.2014** (2.2x vs random) |
| Uplift Segmentation | Accuracy | **73%** |

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│  Data Layer: PostgreSQL + Redis Feature Store           │
├─────────────────────────────────────────────────────────┤
│  Feature Pipeline: Airflow ETL → RFM + Behavioral       │
├─────────────────────────────────────────────────────────┤
│  ML Pipeline: LightGBM + SHAP + CoxPH + T-Learner      │
├─────────────────────────────────────────────────────────┤
│  Serving: FastAPI (/predict, /explain, /segment)        │
├─────────────────────────────────────────────────────────┤
│  Monitoring: Evidently drift → auto-retrain loop        │
├─────────────────────────────────────────────────────────┤
│  Dashboard: Streamlit business UI                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Tech Stack

**Infrastructure:** WSL2 + Docker Compose + Git + GitHub Actions

**Data:** PostgreSQL · Redis Feature Store

**ML:** LightGBM · SHAP · CoxPH (lifelines) · T-Learner Uplift

**MLOps:** MLflow · Apache Airflow · FastAPI · Evidently AI

**UI:** Streamlit

> **Note:** Kafka excluded — batch problem, overkill. Can integrate when scaling to millions of real-time events.

---

## 📁 Project Structure
```
custormer_churn/
├── data/
│   ├── raw/                    # Telco dataset (7,043 customers)
│   └── processed/              # Features, scores, models
│       └── models/             # lgbm_churn.txt, coxph, uplift T0/T1
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_shap_explainability.ipynb
│   ├── 05_survival_analysis.ipynb
│   └── 06_uplift_model.ipynb
├── dags/
│   ├── etl_pipeline.py         # Daily: quality check + features + Redis
│   ├── batch_scoring.py        # Nightly: score all users + PostgreSQL
│   └── retrain_pipeline.py     # On drift: retrain + promote if better
├── src/
│   ├── serving/
│   │   └── main.py             # FastAPI: /predict /explain /segment
│   ├── dashboard/
│   │   └── app.py              # Streamlit business dashboard
│   ├── monitoring/
│   │   └── drift_detector.py   # Evidently drift detection
│   └── decision_engine.py      # Budget-constrained action optimizer
├── scripts/
│   ├── start_project.sh        # One-command startup
│   ├── stop_project.sh         # Graceful shutdown
│   └── data_generator.py       # Synthetic uplift data with causal effect
└── docker-compose.yml          # PostgreSQL + Redis + MLflow + Airflow
```

---

## ⚡ Quick Start
```bash
git clone https://github.com/<your-username>/custormer_churn.git
cd custormer_churn

# Setup venv
python -m venv custormer_churn
source custormer_churn/bin/activate
pip install -r requirements.txt

# Start all services
./scripts/start_project.sh
```

| Service | URL | Credentials |
|---|---|---|
| Streamlit Dashboard | http://localhost:8501 | — |
| FastAPI Swagger | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8080 | admin/admin |

---

## 🎯 Decision Engine Logic
```python
# Percentile-based targeting — no hardcoded thresholds
top_churn = df['churn_probability'].quantile(0.90)
action_list = df[
    (df['churn_probability'] >= top_churn) &
    (df['uplift_segment'] == 'Persuadable')
].nlargest(budget // avg_cost, 'expected_clv_saved')
```

| Segment | Churn Risk | Voucher Effect | Action |
|---|---|---|---|
| **Persuadable VIP** | High | Reduces 39% | Voucher 20% + CS Call |
| **Persuadable Regular** | High | Reduces 39% | Email Voucher 10% |
| **Lost Cause** | High | Minimal | Escalate CS |
| **Sleeping Dog** | Medium | INCREASES 30% | ⚠️ SUPPRESS |
| **Sure Thing** | Low | Unnecessary | No action |

---

## 🔄 MLOps Pipeline
```
Daily 2AM  → etl_pipeline DAG     → Feature Engineering → Redis
Daily 3AM  → batch_scoring DAG    → Score all users → PostgreSQL  
Daily 4AM  → retrain_pipeline DAG → Check drift flag → Retrain if needed

Evidently monitors data drift weekly:
  drift_score > 0.3 → set drift_detected.flag → Airflow triggers retrain
  New model AUC > baseline (0.8445) → promote to production
```

---

## 📈 Key Business Insights

From SHAP + CoxPH analysis:

1. **Contract type** (HR=0.43) — Annual contract reduces churn risk **57%**. Cheapest retention lever.
2. **Fiber optic** (HR=1.31) — High expectation customers. Monitor service quality complaints.
3. **charge_per_month** — Customers feeling "not worth it" churn fastest.
4. **Tenure** — First 3 months critical. Strengthen onboarding experience.

---

## 🗺️ 8-Week Roadmap

| Week | Milestone | Status |
|---|---|---|
| W1 | Infra setup + EDA | ✅ |
| W2 | Feature Engineering + Data Generator | ✅ |
| W3 | LightGBM + SHAP + CoxPH + Uplift | ✅ |
| W4 | MLflow + FastAPI (3 endpoints) | ✅ |
| W5 | Airflow (3 DAGs) | ✅ |
| W6 | Decision Engine | ✅ |
| W7 | Evidently drift detection | ✅ |
| W8 | Streamlit Dashboard + README | ✅ |

---

## 📝 Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features, 26.5% churn rate.

---

*Built with ❤️ as an end-to-end MLOps portfolio project.*
