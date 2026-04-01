from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import shap
import json
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR = Path("data/processed/models")

# ─── Global models ───────────────────────────────────────────────────────────
lgbm_model = None
coxph_model = None
t0_model = None
t1_model = None
explainer = None
feature_cols = None

# ─── Load models ─────────────────────────────────────────────────────────────
def _load_models():
    global lgbm_model, coxph_model, t0_model, t1_model, explainer, feature_cols
    lgbm_model  = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_churn.txt"))
    coxph_model = joblib.load(MODEL_DIR / "coxph_model.pkl")
    t0_model    = joblib.load(MODEL_DIR / "uplift_t0.pkl")
    t1_model    = joblib.load(MODEL_DIR / "uplift_t1.pkl")
    with open(MODEL_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    explainer = shap.TreeExplainer(lgbm_model)
    print("All models loaded ✅")

@asynccontextmanager
async def lifespan(app):
    _load_models()
    yield

app = FastAPI(
    lifespan=lifespan,
    title="Customer Churn Prevention API",
    description="Predict churn probability, explain decisions, segment customers",
    version="1.0.0"
)

# ─── Schemas ─────────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    tenure: float
    SeniorCitizen: float
    MonthlyCharges: float
    TotalCharges: float
    Partner: float
    Dependents: float
    PhoneService: float
    PaperlessBilling: float
    MultipleLines: float
    OnlineSecurity: float
    OnlineBackup: float
    DeviceProtection: float
    TechSupport: float
    StreamingTV: float
    StreamingMovies: float
    Contract_encoded: float
    InternetService_encoded: float
    PaymentMethod_encoded: float
    recency_risk: float
    service_count: float
    monetary_value: float
    monthly_to_total_ratio: float
    charge_per_month: float
    clv_proxy: float
    is_high_value: float
    contract_stability: float
    digital_engagement: float

class UpliftFeatures(BaseModel):
    tenure: float
    monthly_charges: float
    contract_type: float
    service_count: float
    clv_proxy: float
    senior_citizen: float
    has_partner: float
    digital_engagement: float

# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        data = pd.DataFrame([customer.model_dump()])[feature_cols]
        churn_prob  = float(lgbm_model.predict(data)[0])
        churn_label = int(churn_prob > 0.446)

        surv_features = data[['tenure', 'MonthlyCharges',
                               'InternetService_encoded', 'Contract_encoded',
                               'SeniorCitizen', 'service_count',
                               'charge_per_month', 'digital_engagement']].copy()

        surv_7  = float(coxph_model.predict_survival_function(surv_features, times=[7]).values[0][0])
        surv_30 = float(coxph_model.predict_survival_function(surv_features, times=[30]).values[0][0])
        surv_90 = float(coxph_model.predict_survival_function(surv_features, times=[90]).values[0][0])

        return {
            "churn_probability": round(churn_prob, 4),
            "churn_predicted":   churn_label,
            "survival": {
                "p_active_7d":  round(surv_7, 4),
                "p_active_30d": round(surv_30, 4),
                "p_active_90d": round(surv_90, 4),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(customer: CustomerFeatures):
    try:
        data = pd.DataFrame([customer.model_dump()])[feature_cols]
        shap_vals = explainer.shap_values(data)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_series  = pd.Series(shap_vals[0], index=feature_cols)
        top_positive = shap_series.nlargest(5).to_dict()
        top_negative = shap_series.nsmallest(5).to_dict()

        base_val = explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[1]

        return {
            "base_value":           float(base_val),
            "top_churn_drivers":    {k: round(v, 4) for k, v in top_positive.items()},
            "top_churn_protectors": {k: round(v, 4) for k, v in top_negative.items()},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment")
def segment(customer: UpliftFeatures):
    try:
        data = pd.DataFrame([customer.model_dump()])
        p_no_treatment   = float(t0_model.predict_proba(data)[:, 1][0])
        p_with_treatment = float(t1_model.predict_proba(data)[:, 1][0])
        uplift_score     = p_no_treatment - p_with_treatment

        if uplift_score > 0.25 and p_no_treatment > 0.5:
            seg    = "Persuadable"
            action = "Send voucher 20%"
            priority = "HIGH"
        elif uplift_score < -0.10:
            seg    = "Sleeping_Dog"
            action = "Do NOT contact"
            priority = "SUPPRESS"
        elif p_no_treatment > 0.6 and uplift_score < 0.10:
            seg    = "Lost_Cause"
            action = "Escalate to CS team"
            priority = "LOW"
        else:
            seg    = "Sure_Thing"
            action = "No action needed"
            priority = "NONE"

        return {
            "uplift_score":         round(uplift_score, 4),
            "p_churn_no_action":    round(p_no_treatment, 4),
            "p_churn_with_voucher": round(p_with_treatment, 4),
            "segment":              seg,
            "recommended_action":   action,
            "priority":             priority,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
