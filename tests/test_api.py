"""
Tests cho FastAPI endpoints.
Chạy: pytest tests/ -v
"""
import pytest
# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def sample_customer():
    return {
        "tenure": 3,
        "SeniorCitizen": 0,
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "MultipleLines": 0,
        "OnlineSecurity": 0,
        "OnlineBackup": 0,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 1,
        "StreamingMovies": 1,
        "Contract_encoded": 0,
        "InternetService_encoded": 2,
        "PaymentMethod_encoded": 0,
        "recency_risk": 0.25,
        "service_count": 3,
        "monetary_value": 285.0,
        "monthly_to_total_ratio": 0.33,
        "charge_per_month": 31.0,
        "clv_proxy": 213.75,
        "is_high_value": 0,
        "contract_stability": 0,
        "digital_engagement": 1,
    }

@pytest.fixture
def loyal_customer():
    return {
        "tenure": 60,
        "SeniorCitizen": 0,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "Partner": 1,
        "Dependents": 1,
        "PhoneService": 1,
        "PaperlessBilling": 0,
        "MultipleLines": 1,
        "OnlineSecurity": 1,
        "OnlineBackup": 1,
        "DeviceProtection": 1,
        "TechSupport": 1,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "Contract_encoded": 2,
        "InternetService_encoded": 1,
        "PaymentMethod_encoded": 2,
        "recency_risk": 0.016,
        "service_count": 7,
        "monetary_value": 2700.0,
        "monthly_to_total_ratio": 0.017,
        "charge_per_month": 0.74,
        "clv_proxy": 2655.6,
        "is_high_value": 1,
        "contract_stability": 120,
        "digital_engagement": 2,
    }

@pytest.fixture
def sample_uplift():
    return {
        "tenure": 3,
        "monthly_charges": 95.0,
        "contract_type": 0,
        "service_count": 3,
        "clv_proxy": 213.75,
        "senior_citizen": 0,
        "has_partner": 0,
        "digital_engagement": 1,
    }

# ── Health Check ──────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_models_loaded(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True

# ── /predict ──────────────────────────────────────────────────
class TestPredict:
    def test_predict_returns_200(self, client, sample_customer):
        r = client.post("/predict", json=sample_customer)
        assert r.status_code == 200

    def test_predict_has_required_fields(self, client, sample_customer):
        r = client.post("/predict", json=sample_customer)
        data = r.json()
        assert "churn_probability" in data
        assert "churn_predicted" in data
        assert "survival" in data

    def test_churn_probability_range(self, client, sample_customer):
        r = client.post("/predict", json=sample_customer)
        prob = r.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_churn_predicted_is_binary(self, client, sample_customer):
        r = client.post("/predict", json=sample_customer)
        pred = r.json()["churn_predicted"]
        assert pred in [0, 1]

    def test_survival_fields_exist(self, client, sample_customer):
        r = client.post("/predict", json=sample_customer)
        surv = r.json()["survival"]
        assert "p_active_7d" in surv
        assert "p_active_30d" in surv
        assert "p_active_90d" in surv

    def test_high_risk_customer(self, client, sample_customer):
        """New customer + Fiber + Month-to-month → high churn"""
        r = client.post("/predict", json=sample_customer)
        prob = r.json()["churn_probability"]
        assert prob > 0.5, f"Expected high risk, got {prob}"

    def test_loyal_customer_lower_risk(self, client, loyal_customer, sample_customer):
        """Loyal customer should have lower churn prob than new risky customer"""
        r_loyal = client.post("/predict", json=loyal_customer)
        r_risky = client.post("/predict", json=sample_customer)
        assert r_loyal.json()["churn_probability"] < r_risky.json()["churn_probability"]

    def test_missing_field_returns_422(self, client, sample_customer):
        incomplete = {k: v for k, v in sample_customer.items() if k != "tenure"}
        r = client.post("/predict", json=incomplete)
        assert r.status_code == 422

# ── /explain ──────────────────────────────────────────────────
class TestExplain:
    def test_explain_returns_200(self, client, sample_customer):
        r = client.post("/explain", json=sample_customer)
        assert r.status_code == 200

    def test_explain_has_shap_fields(self, client, sample_customer):
        r = client.post("/explain", json=sample_customer)
        data = r.json()
        assert "base_value" in data
        assert "top_churn_drivers" in data
        assert "top_churn_protectors" in data

    def test_explain_top_drivers_not_empty(self, client, sample_customer):
        r = client.post("/explain", json=sample_customer)
        data = r.json()
        assert len(data["top_churn_drivers"]) > 0
        assert len(data["top_churn_protectors"]) > 0

# ── /segment ──────────────────────────────────────────────────
class TestSegment:
    VALID_SEGMENTS = {"Persuadable", "Sleeping_Dog", "Lost_Cause", "Sure_Thing"}
    VALID_PRIORITIES = {"HIGH", "SUPPRESS", "LOW", "NONE"}

    def test_segment_returns_200(self, client, sample_uplift):
        r = client.post("/segment", json=sample_uplift)
        assert r.status_code == 200

    def test_segment_has_required_fields(self, client, sample_uplift):
        r = client.post("/segment", json=sample_uplift)
        data = r.json()
        assert "uplift_score" in data
        assert "segment" in data
        assert "recommended_action" in data
        assert "priority" in data

    def test_segment_is_valid(self, client, sample_uplift):
        r = client.post("/segment", json=sample_uplift)
        seg = r.json()["segment"]
        assert seg in self.VALID_SEGMENTS

    def test_priority_is_valid(self, client, sample_uplift):
        r = client.post("/segment", json=sample_uplift)
        priority = r.json()["priority"]
        assert priority in self.VALID_PRIORITIES

    def test_uplift_score_is_float(self, client, sample_uplift):
        r = client.post("/segment", json=sample_uplift)
        score = r.json()["uplift_score"]
        assert isinstance(score, float)

# ── Decision Engine ───────────────────────────────────────────
class TestDecisionEngine:
    def test_decision_engine_runs(self):
        import pandas as pd
        from src.decision_engine import DecisionEngine

        engine = DecisionEngine(monthly_budget=5000)
        df = pd.read_csv("data/processed/batch_scores_latest.csv")
        df_result, within_budget = engine.run(df)

        assert len(df_result) > 0
        assert "segment" in df_result.columns
        assert "recommended_action" in df_result.columns

    def test_budget_constraint(self):
        import pandas as pd
        from src.decision_engine import DecisionEngine

        budget = 1000
        engine = DecisionEngine(monthly_budget=budget)
        df = pd.read_csv("data/processed/batch_scores_latest.csv")
        _, within_budget = engine.run(df)

        assert within_budget["action_cost"].sum() <= budget

    def test_sleeping_dogs_suppressed(self):
        import pandas as pd
        from src.decision_engine import DecisionEngine

        engine = DecisionEngine(monthly_budget=5000)
        df = pd.read_csv("data/processed/batch_scores_latest.csv")
        df_result, within_budget = engine.run(df)

        # Sleeping Dogs không được target
        targeted_sleeping = within_budget[
            within_budget["segment"] == "Sleeping_Dog"
        ]
        assert len(targeted_sleeping) == 0
