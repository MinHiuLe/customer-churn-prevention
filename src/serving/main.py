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

# ─── 1. Cấu hình Logging chuyên nghiệp ───────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── 2. Đường dẫn linh hoạt qua Biến môi trường ──────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/opt/airflow/data/processed/models"))

# ─── 3. Load Model an toàn vào app.state ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Đang nạp các mô hình AI...")
        # Gắn trực tiếp model vào trạng thái của app (Best Practice của FastAPI)
        app.state.lgbm_model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_churn.txt"))
        app.state.coxph_model = joblib.load(MODEL_DIR / "coxph_model.pkl")
        app.state.t0_model = joblib.load(MODEL_DIR / "uplift_t0.pkl")
        app.state.t1_model = joblib.load(MODEL_DIR / "uplift_t1.pkl")
        
        with open(MODEL_DIR / "feature_cols.json") as f:
            app.state.feature_cols = json.load(f)
            
        app.state.explainer = shap.TreeExplainer(app.state.lgbm_model)
        logger.info("✅ Nạp toàn bộ mô hình thành công!")
        yield
    except Exception as e:
        logger.error(f"❌ Lỗi nạp mô hình: {e}")
        raise e

app = FastAPI(
    lifespan=lifespan,
    title="Slotcheck MLOps API",
    description="Hệ thống API phục vụ dự báo Churn, Uplift và Survival Analysis.",
    version="1.0.0"
)

# ─── 4. Cung cấp Dữ liệu mẫu (ConfigDict) cho Swagger UI ─────────────────────
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

    # Config này giúp giao diện /docs có sẵn số liệu để bạn bấm "Try it out"
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 12.0, "SeniorCitizen": 0.0, "MonthlyCharges": 55.0, "TotalCharges": 660.0,
                "Partner": 1.0, "Dependents": 0.0, "PhoneService": 1.0, "PaperlessBilling": 1.0,
                "MultipleLines": 0.0, "OnlineSecurity": 1.0, "OnlineBackup": 0.0, "DeviceProtection": 0.0,
                "TechSupport": 1.0, "StreamingTV": 0.0, "StreamingMovies": 0.0, "Contract_encoded": 0.0,
                "InternetService_encoded": 1.0, "PaymentMethod_encoded": 0.0, "recency_risk": 0.076,
                "service_count": 2.0, "monetary_value": 660.0, "monthly_to_total_ratio": 0.08,
                "charge_per_month": 4.2, "clv_proxy": 610.0, "is_high_value": 0.0,
                "contract_stability": 0.0, "digital_engagement": 1.0
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
                "tenure": 12.0, "monthly_charges": 55.0, "contract_type": 0.0, "service_count": 2.0,
                "clv_proxy": 610.0, "senior_citizen": 0.0, "has_partner": 1.0, "digital_engagement": 1.0
            }
        }
    )

# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict")
def predict(customer: CustomerFeatures, request: Request):
    try:
        # Lấy model từ request.app.state thay vì global
        state = request.app.state
        data = pd.DataFrame([customer.model_dump()])[state.feature_cols]
        
        churn_prob = float(state.lgbm_model.predict(data)[0])
        churn_label = int(churn_prob > 0.446)

        # Lấy an toàn các cột cho Survival Model
        surv_cols = ['tenure', 'MonthlyCharges', 'InternetService_encoded', 'Contract_encoded',
                     'SeniorCitizen', 'service_count', 'charge_per_month', 'digital_engagement']
        surv_features = data[surv_cols].copy()

        surv_7 = float(state.coxph_model.predict_survival_function(surv_features, times=[7]).values[0][0])
        surv_30 = float(state.coxph_model.predict_survival_function(surv_features, times=[30]).values[0][0])
        surv_90 = float(state.coxph_model.predict_survival_function(surv_features, times=[90]).values[0][0])

        return {
            "churn_probability": round(churn_prob, 4),
            "churn_predicted": churn_label,
            "survival": {
                "p_active_7d": round(surv_7, 4),
                "p_active_30d": round(surv_30, 4),
                "p_active_90d": round(surv_90, 4),
            }
        }
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(customer: CustomerFeatures, request: Request):
    try:
        state = request.app.state
        data = pd.DataFrame([customer.model_dump()])[state.feature_cols]
        
        shap_vals = state.explainer.shap_values(data)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_series = pd.Series(shap_vals[0], index=state.feature_cols)
        top_positive = shap_series.nlargest(5).to_dict()
        top_negative = shap_series.nsmallest(5).to_dict()

        base_val = state.explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[1]

        return {
            "base_value": float(base_val),
            "top_churn_drivers": {k: round(v, 4) for k, v in top_positive.items()},
            "top_churn_protectors": {k: round(v, 4) for k, v in top_negative.items()},
        }
    except Exception as e:
        logger.error(f"Explain Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment")
def segment(customer: UpliftFeatures, request: Request):
    try:
        state = request.app.state
        data = pd.DataFrame([customer.model_dump()])
        
        p_no_treatment = float(state.t0_model.predict_proba(data)[:, 1][0])
        p_with_treatment = float(state.t1_model.predict_proba(data)[:, 1][0])
        uplift_score = p_no_treatment - p_with_treatment

        # Logic gom cụm gọn gàng hơn
        if uplift_score > 0.25 and p_no_treatment > 0.5:
            seg, action, priority = "Persuadable", "Send voucher 20%", "HIGH"
        elif uplift_score < -0.10:
            seg, action, priority = "Sleeping_Dog", "Do NOT contact", "SUPPRESS"
        elif p_no_treatment > 0.6 and uplift_score < 0.10:
            seg, action, priority = "Lost_Cause", "Escalate to CS team", "LOW"
        else:
            seg, action, priority = "Sure_Thing", "No action needed", "NONE"

        return {
            "uplift_score": round(uplift_score, 4),
            "p_churn_no_action": round(p_no_treatment, 4),
            "p_churn_with_voucher": round(p_with_treatment, 4),
            "segment": seg,
            "recommended_action": action,
            "priority": priority,
        }
    except Exception as e:
        logger.error(f"Segment Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))