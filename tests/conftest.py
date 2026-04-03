import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent))
####D
# Feature columns must match CustomerFeatures schema in main.py exactly
FEATURE_COLS = [
    "tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges", "Partner",
    "Dependents", "PhoneService", "PaperlessBilling", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract_encoded", "InternetService_encoded",
    "PaymentMethod_encoded", "recency_risk", "service_count", "monetary_value",
    "monthly_to_total_ratio", "charge_per_month", "clv_proxy", "is_high_value",
    "contract_stability", "digital_engagement",
]

N_FEATURES = len(FEATURE_COLS)


def _make_lgbm_mock() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = [0.35]
    return m


def _make_coxph_mock() -> MagicMock:
    m = MagicMock()
    # predict_survival_function returns an object whose .values[0][0] is iterated
    survival_result = MagicMock()
    survival_result.values = [[[0.95], [0.80], [0.60]]]
    m.predict_survival_function.return_value = survival_result
    return m


def _make_uplift_mock(proba: list[float]) -> MagicMock:
    m = MagicMock()
    m.predict_proba.return_value = [proba]
    return m


def _make_explainer_mock() -> MagicMock:
    m = MagicMock()
    m.shap_values.return_value = [[0.05] * N_FEATURES]
    m.expected_value = 0.28
    return m


def _joblib_side_effect(mock_coxph, mock_t0, mock_t1):
    def _load(path):
        p = str(path)
        if "coxph" in p:
            return mock_coxph
        if "t0" in p:
            return mock_t0
        if "t1" in p:
            return mock_t1
        return MagicMock()
    return _load


@pytest.fixture(scope="session")
def client():
    """
    Provides a FastAPI TestClient with all model I/O mocked.

    Why this is needed:
        The app lifespan loads model files from disk at startup.
        Those files do not exist in the GitHub Actions runner,
        so every test would fail at fixture setup without this mock.
    """
    from fastapi.testclient import TestClient
    from src.serving.main import app

    mock_lgbm   = _make_lgbm_mock()
    mock_coxph  = _make_coxph_mock()
    mock_t0     = _make_uplift_mock([0.3, 0.7])
    mock_t1     = _make_uplift_mock([0.4, 0.6])
    mock_shap   = _make_explainer_mock()

    feature_cols_json = json.dumps(FEATURE_COLS)

    with (
        patch("src.serving.main.lgb.Booster", return_value=mock_lgbm),
        patch("src.serving.main.joblib.load", side_effect=_joblib_side_effect(mock_coxph, mock_t0, mock_t1)),
        patch("src.serving.main.shap.TreeExplainer", return_value=mock_shap),
        patch("builtins.open", mock_open(read_data=feature_cols_json)),
        patch("json.load", return_value=FEATURE_COLS),
    ):
        with TestClient(app) as c:
            yield c