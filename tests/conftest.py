import pytest
import sys
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """
    lgbm_model.predict() trả về numpy array shape (n_samples,).
    Code: float(state.lgbm_model.predict(data)[0])
    Cần: array[0] phải là scalar float, không phải list.
    """
    m = MagicMock()
    m.predict.return_value = np.array([0.35])
    return m


def _make_coxph_mock() -> MagicMock:
    """
    predict_survival_function() được gọi 3 lần với times=[7], [30], [90].
    Mỗi lần trả về DataFrame shape (1 time_point, 1 sample).
    Code: .values[0][0] phải là scalar float.

    Dùng side_effect để mỗi call trả về đúng giá trị tương ứng.
    """
    def _survival_result(scalar_value: float) -> MagicMock:
        result = MagicMock()
        result.values = np.array([[scalar_value]])
        return result

    m = MagicMock()
    m.predict_survival_function.side_effect = [
        _survival_result(0.95),   # times=[7]
        _survival_result(0.80),   # times=[30]
        _survival_result(0.60),   # times=[90]
    ]
    return m


def _make_uplift_mock(proba: list) -> MagicMock:
    """
    predict_proba() trả về numpy array shape (n_samples, 2).
    Code: predict_proba(data)[:, 1]  <- NumPy column indexing
    Python list không hỗ trợ [:, 1] — phải dùng np.array.
    """
    m = MagicMock()
    m.predict_proba.return_value = np.array([proba])
    return m


def _make_explainer_mock() -> MagicMock:
    """
    shap_values() với binary classifier trả về list 2 phần tử:
      [shap_class_0, shap_class_1]
    Code: if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    Mock cũ chỉ có 1 phần tử -> IndexError khi lấy [1].
    """
    m = MagicMock()
    m.shap_values.return_value = [
        np.array([[-0.05] * N_FEATURES]),   # class 0
        np.array([[0.05]  * N_FEATURES]),   # class 1 <- code lấy cái này
    ]
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
    FastAPI TestClient với toàn bộ model I/O được mock.

    Tại sao cần mock:
        Lifespan của app load model files từ disk khi khởi động.
        Các file đó không tồn tại trong GitHub Actions runner,
        khiến mọi test fail ngay tại fixture setup.
    """
    from fastapi.testclient import TestClient
    from src.serving.main import app

    mock_lgbm  = _make_lgbm_mock()
    mock_coxph = _make_coxph_mock()
    mock_t0    = _make_uplift_mock([0.3, 0.7])
    mock_t1    = _make_uplift_mock([0.4, 0.6])
    mock_shap  = _make_explainer_mock()

    with (
        patch("src.serving.main.lgb.Booster", return_value=mock_lgbm),
        patch("src.serving.main.joblib.load",
              side_effect=_joblib_side_effect(mock_coxph, mock_t0, mock_t1)),
        patch("src.serving.main.shap.TreeExplainer", return_value=mock_shap),
        patch("builtins.open", mock_open(read_data=json.dumps(FEATURE_COLS))),
        patch("json.load", return_value=FEATURE_COLS),
    ):
        with TestClient(app) as c:
            yield c


def _sample_customer_df():
    """
    DataFrame mẫu đủ tất cả cột mà Decision Engine yêu cầu,
    bao gồm clv_proxy để tránh KeyError tại df['clv_proxy'].quantile(0.75).
    """
    import pandas as pd
    return pd.DataFrame([{
        "customerID":               "TEST_001",
        "tenure":                   12.0,
        "SeniorCitizen":            0,
        "MonthlyCharges":           55.0,
        "TotalCharges":             660.0,
        "Partner":                  1,
        "Dependents":               0,
        "PhoneService":             1,
        "PaperlessBilling":         1,
        "MultipleLines":            0,
        "OnlineSecurity":           1,
        "OnlineBackup":             0,
        "DeviceProtection":         0,
        "TechSupport":              1,
        "StreamingTV":              0,
        "StreamingMovies":          0,
        "Contract_encoded":         0,
        "InternetService_encoded":  1,
        "PaymentMethod_encoded":    0,
        "recency_risk":             round(1 / (12 + 1), 4),
        "service_count":            2,
        "monetary_value":           660.0,
        "monthly_to_total_ratio":   round(55.0 / (660.0 + 1), 4),
        "charge_per_month":         round(55.0 / (12 + 1), 4),
        "clv_proxy":                610.0,
        "is_high_value":            0,
        "contract_stability":       0.0,
        "digital_engagement":       1,
        "churn_probability":        0.72,
        "uplift_score":             0.31,
        "segment":                  "Persuadable",
    }])


@pytest.fixture(scope="session")
def sample_df():
    """Cung cấp DataFrame mẫu có đủ cột cho Decision Engine tests."""
    return _sample_customer_df()