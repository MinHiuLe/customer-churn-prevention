from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
import pandas as pd
import lightgbm as lgb
import joblib
import shap
import json
import logging
import os
from pathlib import Path

# Shared segment logic — single source of truth for batch AND real-time
from src.mlops.segment import classify_segment, get_action

# ─── 1. Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── 2. Paths from environment ────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/opt/airflow/data/processed/models"))

# ─── 3. Lifespan: load models into app.state ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading models...")

        app.state.lgbm_model  = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_churn.txt"))
        app.state.coxph_model = joblib.load(MODEL_DIR / "coxph_model.pkl")
        app.state.t0_model    = joblib.load(MODEL_DIR / "uplift_t0.pkl")
        app.state.t1_model    = joblib.load(MODEL_DIR / "uplift_t1.pkl")

        with open(MODEL_DIR / "feature_cols.json") as f:
            app.state.feature_cols = json.load(f)

        # Load threshold được tính khi retrain — không hardcode
        config_path = MODEL_DIR / "model_config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
            app.state.threshold = model_config.get("churn_threshold", 0.5)
            logger.info(f"Loaded churn threshold: {app.state.threshold:.4f}")
        else:
            app.state.threshold = 0.5
            logger.warning("model_config.json not found. Using default threshold=0.5")

        app.state.explainer = shap.TreeExplainer(app.state.lgbm_model)
        logger.info("All models loaded successfully.")
        yield

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


app = FastAPI(
    lifespan=lifespan,
    title="Churn Prevention API",
    description="Churn probability, survival curve, SHAP explanations, and uplift segmentation.",
    version="1.0.0",
)

# ─── 4. Schemas ───────────────────────────────────────────────────────────────
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 12.0, "SeniorCitizen": 0.0, "MonthlyCharges": 55.0,
                "TotalCharges": 660.0, "Partner": 1.0, "Dependents": 0.0,
                "PhoneService": 1.0, "PaperlessBilling": 1.0, "MultipleLines": 0.0,
                "OnlineSecurity": 1.0, "OnlineBackup": 0.0, "DeviceProtection": 0.0,
                "TechSupport": 1.0, "StreamingTV": 0.0, "StreamingMovies": 0.0,
                "Contract_encoded": 0.0, "InternetService_encoded": 1.0,
                "PaymentMethod_encoded": 0.0, "recency_risk": 0.076,
                "service_count": 2.0, "monetary_value": 660.0,
                "monthly_to_total_ratio": 0.08, "charge_per_month": 4.2,
                "clv_proxy": 610.0, "is_high_value": 0.0,
                "contract_stability": 0.0, "digital_engagement": 1.0,
            }
        }
    )


class UpliftFeatures(BaseModel):
    tenure: float
    monthly_charges: float
    contract_type: float
    service_count: float
    clv_proxy: float
    senior_citizen: float
    has_partner: float
    digital_engagement: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 12.0, "monthly_charges": 55.0, "contract_type": 0.0,
                "service_count": 2.0, "clv_proxy": 610.0, "senior_citizen": 0.0,
                "has_partner": 1.0, "digital_engagement": 1.0,
            }
        }
    )


# ─── 5. Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict")
def predict(customer: CustomerFeatures, request: Request):
    try:
        state = request.app.state
        data  = pd.DataFrame([customer.model_dump()])[state.feature_cols]

        churn_prob  = float(state.lgbm_model.predict(data)[0])
        churn_label = int(churn_prob >= state.threshold)

        surv_cols = [
            "tenure", "MonthlyCharges", "InternetService_encoded", "Contract_encoded",
            "SeniorCitizen", "service_count", "charge_per_month", "digital_engagement",
        ]
        surv_features = data[surv_cols].copy()

        surv_7  = float(state.coxph_model.predict_survival_function(surv_features, times=[7]).values[0][0])
        surv_30 = float(state.coxph_model.predict_survival_function(surv_features, times=[30]).values[0][0])
        surv_90 = float(state.coxph_model.predict_survival_function(surv_features, times=[90]).values[0][0])

        return {
            "churn_probability": round(churn_prob, 4),
            "churn_predicted":   churn_label,
            "threshold_used":    round(state.threshold, 4),
            "survival": {
                "p_active_7d":  round(surv_7,  4),
                "p_active_30d": round(surv_30, 4),
                "p_active_90d": round(surv_90, 4),
            },
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain(customer: CustomerFeatures, request: Request):
    try:
        state = request.app.state
        data  = pd.DataFrame([customer.model_dump()])[state.feature_cols]

        shap_vals = state.explainer.shap_values(data)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_series  = pd.Series(shap_vals[0], index=state.feature_cols)
        top_positive = shap_series.nlargest(5).to_dict()
        top_negative = shap_series.nsmallest(5).to_dict()

        base_val = state.explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[1]

        return {
            "base_value":          float(base_val),
            "top_churn_drivers":   {k: round(v, 4) for k, v in top_positive.items()},
            "top_churn_protectors": {k: round(v, 4) for k, v in top_negative.items()},
        }
    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment")
def segment(customer: UpliftFeatures, request: Request):
    try:
        state = request.app.state
        data  = pd.DataFrame([customer.model_dump()])

        p_no_treatment   = float(state.t0_model.predict_proba(data)[:, 1][0])
        p_with_treatment = float(state.t1_model.predict_proba(data)[:, 1][0])
        uplift_score     = p_no_treatment - p_with_treatment

        # classify_segment dùng cùng logic với batch_scoring DAG
        seg    = classify_segment(uplift_score, p_no_treatment)
        action = get_action(seg)

        return {
            "uplift_score":          round(uplift_score, 4),
            "p_churn_no_action":     round(p_no_treatment, 4),
            "p_churn_with_voucher":  round(p_with_treatment, 4),
            "segment":               seg,
            "recommended_action":    action["action"],
            "priority":              action["priority"],
        }
    except Exception as e:
        logger.error(f"Segment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))