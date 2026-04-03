import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import psycopg2
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Slotcheck Churn Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"
DATA_DIR = Path("data/processed")

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_USER     = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# ─── MLflow helpers ───────────────────────────────────────────────────────────

def _make_mlflow_client():
    """Khởi tạo MlflowClient với URI và credentials từ biến môi trường."""
    import mlflow
    from mlflow.tracking import MlflowClient

    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_URI)
    return MlflowClient()


def _get_experiment(client, name: str):
    """Lấy experiment theo tên; fallback về 'Default' nếu không tìm thấy."""
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp, name
    fallback = client.get_experiment_by_name("Default")
    if fallback:
        return fallback, "Default"
    return None, None


def _find_champion(runs, metric_key: str, promoted_tag: str = "promoted"):
    """
    Tìm Champion model theo thứ tự ưu tiên:
      1. Run gần nhất có tag promoted='true' (hoặc 'True')
      2. Nếu không có, trả về None thay vì lấy bừa.
    """
    candidates = [
        r for r in runs
        if r.data.tags.get(promoted_tag, "").lower() in ("true", "1", "yes")
        and metric_key in r.data.metrics
    ]
    if not candidates:
        return None
    # Sắp xếp theo thời gian bắt đầu mới nhất
    candidates.sort(key=lambda r: r.info.start_time, reverse=True)
    return candidates[0]


# ─── Cache: MLflow champion metrics ──────────────────────────────────────────

@st.cache_data(ttl=300)
def load_champion_metrics() -> dict:
    """
    Trả về dict với 3 key: lgbm, cox, uplift.
    Mỗi key chứa {'score': float|None, 'run_id': str, 'trained': str}.
    Nếu không kết nối được MLflow, trả về tất cả None.
    """
    default = {
        "lgbm":   {"score": None, "run_id": "N/A", "trained": "N/A"},
        "cox":    {"score": None, "run_id": "N/A", "trained": "N/A"},
        "uplift": {"score": None, "run_id": "N/A", "trained": "N/A"},
    }

    try:
        client = _make_mlflow_client()
        exp, exp_name = _get_experiment(client, "churn-prediction")
        if exp is None:
            return default

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=200,
        )

        def _fmt(run):
            ts = run.info.start_time
            return pd.Timestamp(ts, unit="ms").strftime("%Y-%m-%d %H:%M") if ts else "N/A"

        # ── LightGBM champion ──
        lgbm_run = _find_champion(
            [r for r in runs if any(k in r.data.metrics for k in ("auc_roc", "auc"))],
            metric_key="auc_roc",
        ) or _find_champion(
            [r for r in runs if "auc" in r.data.metrics],
            metric_key="auc",
        )
        if lgbm_run:
            score = lgbm_run.data.metrics.get("auc_roc") or lgbm_run.data.metrics.get("auc")
            default["lgbm"] = {
                "score": round(score, 4),
                "run_id": lgbm_run.info.run_id[:8],
                "trained": _fmt(lgbm_run),
            }

        # ── CoxPH champion ──
        cox_run = _find_champion(
            [r for r in runs if "c_index" in r.data.metrics],
            metric_key="c_index",
        )
        if cox_run:
            default["cox"] = {
                "score": round(cox_run.data.metrics["c_index"], 4),
                "run_id": cox_run.info.run_id[:8],
                "trained": _fmt(cox_run),
            }

        # ── Uplift champion ──
        uplift_run = _find_champion(
            [r for r in runs if "auuc" in r.data.metrics],
            metric_key="auuc",
        )
        if uplift_run:
            default["uplift"] = {
                "score": round(uplift_run.data.metrics["auuc"], 4),
                "run_id": uplift_run.info.run_id[:8],
                "trained": _fmt(uplift_run),
            }

        return default

    except Exception as e:
        st.warning(f"⚠️ Không lấy được metrics từ MLflow: {e}")
        return default


@st.cache_data(ttl=300)
def load_model_registry_table() -> pd.DataFrame | None:
    """
    Trả về DataFrame đầy đủ tất cả runs để hiển thị trong bảng Model Registry.
    """
    try:
        client = _make_mlflow_client()
        exp, exp_name = _get_experiment(client, "churn-prediction")
        if exp is None:
            return None

        if exp_name != "churn-prediction":
            st.warning(
                f"⚠️ Không thấy experiment 'churn-prediction'. "
                f"Đang dùng '{exp_name}'. Hãy retrain để đẩy đúng chỗ."
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
        )

        rows = []
        for run in runs:
            name       = run.info.run_name or ""
            name_lower = name.lower()
            metrics    = run.data.metrics
            tags       = run.data.tags
            ts         = run.info.start_time
            trained    = pd.Timestamp(ts, unit="ms").strftime("%Y-%m-%d %H:%M") if ts else "N/A"
            promoted   = tags.get("promoted", "false").lower()
            status     = "✅ Production" if promoted in ("true", "1", "yes") else "🔄 Staging"

            if "auc_roc" in metrics or "auc" in metrics or "lightgbm" in name_lower or "lgbm" in name_lower:
                score_key = "auc_roc" if "auc_roc" in metrics else "auc"
                rows.append({
                    "Model": name or "LightGBM",
                    "Metric": "AUC-ROC",
                    "Score": round(metrics.get(score_key, 0), 4),
                    "Status": status,
                    "Trained": trained,
                    "Run ID": run.info.run_id[:8],
                })
            elif "c_index" in metrics or "cox" in name_lower or "survival" in name_lower:
                rows.append({
                    "Model": name or "CoxPH",
                    "Metric": "C-index",
                    "Score": round(metrics.get("c_index", 0), 4),
                    "Status": status,
                    "Trained": trained,
                    "Run ID": run.info.run_id[:8],
                })
            elif "auuc" in metrics or "uplift" in name_lower:
                rows.append({
                    "Model": name or "Uplift T-Learner",
                    "Metric": "AUUC",
                    "Score": round(metrics.get("auuc", 0), 4),
                    "Status": status,
                    "Trained": trained,
                    "Run ID": run.info.run_id[:8],
                })
            elif metrics:
                first_key = next(iter(metrics))
                rows.append({
                    "Model": name or "Unknown",
                    "Metric": first_key,
                    "Score": round(metrics[first_key], 4),
                    "Status": status,
                    "Trained": trained,
                    "Run ID": run.info.run_id[:8],
                })

        return pd.DataFrame(rows) if rows else None

    except Exception as e:
        st.error(f"Không thể kết nối MLflow Registry: {e}")
        return None


# ─── Cache: Postgres / CSV data ───────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_scores() -> pd.DataFrame | None:
    """Ưu tiên Postgres; fallback về CSV local nếu lỗi."""
    try:
        conn = psycopg2.connect(
            host="localhost",                           # luôn localhost (ngoài Docker)
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("POSTGRES_DB", "churn_db"),
            user=os.getenv("POSTGRES_USER", "churn_user"),
            password=os.getenv("POSTGRES_PASSWORD", "churn_pass"),
        )
        df = pd.read_sql("SELECT * FROM churn_scores", conn)
        conn.close()
        return df if not df.empty else _scores_from_csv()
    except Exception as e:
        st.warning(f"⚠️ DB lỗi, dùng CSV local: {e}")
        return _scores_from_csv()


def _scores_from_csv() -> pd.DataFrame | None:
    path = DATA_DIR / "batch_scores_latest.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=300)
def load_action_plan() -> pd.DataFrame | None:
    path = DATA_DIR / "action_plan.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=300)
def load_shap() -> pd.DataFrame | None:
    path = DATA_DIR / "shap_importance.csv"
    return pd.read_csv(path) if path.exists() else None


def load_drift_summary() -> dict | None:
    path = DATA_DIR / "drift_reports/drift_summary_latest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/bullseye-emoji.png", width=60)
st.sidebar.title("Slotcheck MLOps")
st.sidebar.markdown("**Retention Dashboard v1.0**")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "🎯 Action Plan",
    "🔍 Customer Lookup",
    "📈 Model Performance",
    "⚠️ Drift Monitor",
])

budget = st.sidebar.slider("Monthly Budget ($)", 1000, 20_000, 5000, 500)
st.sidebar.divider()
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ─── Load all data ────────────────────────────────────────────────────────────
df_scores    = load_scores()
df_plan      = load_action_plan()
df_shap      = load_shap()
drift_summary = load_drift_summary()

# ─── PAGE 1: Overview ─────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Customer Churn Overview")

    if df_scores is None:
        st.warning("No batch scores found. Run batch_scoring DAG first.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    total       = len(df_scores)
    high_risk   = (df_scores["churn_probability"] > 0.7).sum()
    persuadable = (df_scores["segment"] == "Persuadable").sum()
    avg_churn   = df_scores["churn_probability"].mean()

    col1.metric("Total Customers", f"{total:,}")
    col2.metric("High Risk (>70%)", f"{high_risk:,}",
                delta=f"{high_risk/total*100:.1f}%", delta_color="inverse")
    col3.metric("Persuadable", f"{persuadable:,}",
                delta=f"{persuadable/total*100:.1f}%")
    col4.metric("Avg Churn Prob", f"{avg_churn:.1%}")

    st.divider()
    col_left, col_right = st.columns(2)

    colors = {
        "Persuadable":  "#2ECC71",
        "Sure_Thing":   "#3498DB",
        "Lost_Cause":   "#E74C3C",
        "Sleeping_Dog": "#F39C12",
    }

    with col_left:
        st.subheader("Segment Distribution")
        seg_counts = df_scores["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(seg_counts, values="Count", names="Segment",
                     color="Segment", color_discrete_map=colors)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(df_scores, x="churn_probability",
                           nbins=50, color_discrete_sequence=["#E74C3C"])
        fig.add_vline(x=0.446, line_dash="dash",
                      annotation_text="Threshold 0.446")
        fig.update_layout(xaxis_title="Churn Probability", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Uplift Score vs Churn Probability")
    fig = px.scatter(df_scores, x="churn_probability", y="uplift_score",
                     color="segment", color_discrete_map=colors,
                     hover_data=["user_id", "clv_proxy"], opacity=0.6)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.446, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGE 2: Action Plan ──────────────────────────────────────────────────────
elif page == "🎯 Action Plan":
    st.title("🎯 Action Plan — Budget Optimizer")

    if df_plan is None:
        st.warning("No action plan found. Run decision engine first.")
        st.stop()

    persuadable = df_plan[df_plan["segment"] == "Persuadable"].copy()
    persuadable = persuadable.sort_values("roi", ascending=False)
    persuadable["cumulative_cost"] = persuadable["action_cost"].cumsum()
    within_budget = persuadable[persuadable["cumulative_cost"] <= budget]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users to Target", f"{len(within_budget):,}")
    col2.metric("Total Cost", f"${within_budget['action_cost'].sum():,.0f}")
    col3.metric("Expected CLV Saved",
                f"${within_budget['expected_clv_saved'].sum():,.0f}")
    col4.metric(
        "Avg ROI",
        f"{within_budget['roi'].mean():.1f}x" if len(within_budget) > 0 else "N/A",
    )

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Action Breakdown")
        if len(within_budget) > 0:
            action_counts = within_budget["recommended_action"].value_counts().reset_index()
            action_counts.columns = ["Action", "Count"]
            fig = px.bar(action_counts, x="Action", y="Count", color="Action",
                         color_discrete_sequence=["#2ECC71", "#3498DB"])
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("ROI Distribution")
        if len(within_budget) > 0:
            fig = px.histogram(within_budget, x="roi", nbins=30,
                               color_discrete_sequence=["#2ECC71"])
            fig.update_layout(xaxis_title="ROI", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Target List ({len(within_budget)} users)")
    if len(within_budget) > 0:
        display_cols = [
            c for c in ["user_id", "churn_probability", "uplift_score",
                         "recommended_action", "action_cost",
                         "expected_clv_saved", "roi"]
            if c in within_budget.columns
        ]
        st.dataframe(
            within_budget[display_cols].head(100)
            .style.background_gradient(subset=["churn_probability"], cmap="RdYlGn_r")
            .background_gradient(subset=["roi"], cmap="Greens"),
            use_container_width=True,
        )

# ─── PAGE 3: Customer Lookup ──────────────────────────────────────────────────
elif page == "🔍 Customer Lookup":
    st.title("🔍 Customer Lookup — Real-time Prediction")
    st.info("Nhập thông tin khách hàng để nhận dự đoán real-time từ FastAPI")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure           = st.slider("Tenure (months)", 0, 72, 3)
        monthly_charges  = st.slider("Monthly Charges ($)", 20, 120, 95)
        contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        internet         = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        senior           = st.checkbox("Senior Citizen")
        partner          = st.checkbox("Has Partner")
    with col3:
        service_count       = st.slider("Number of Services", 0, 8, 3)
        digital_engagement  = st.slider("Digital Engagement", 0, 2, 1)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}

    total_charges   = monthly_charges * max(tenure, 1)
    charge_per_month = monthly_charges / (tenure + 1)
    clv_proxy       = total_charges * (1 - 1 / (tenure + 1))

    if st.button("🔮 Predict", type="primary"):
        payload = {
            "tenure": tenure,
            "SeniorCitizen": int(senior),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Partner": int(partner),
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
            "Contract_encoded": contract_map[contract],
            "InternetService_encoded": internet_map[internet],
            "PaymentMethod_encoded": 0,
            "recency_risk": 1 / (tenure + 1),
            "service_count": service_count,
            "monetary_value": total_charges,
            "monthly_to_total_ratio": monthly_charges / (total_charges + 1),
            "charge_per_month": charge_per_month,
            "clv_proxy": clv_proxy,
            "is_high_value": int(clv_proxy > 500),
            "contract_stability": contract_map[contract] * tenure,
            "digital_engagement": digital_engagement,
        }

        uplift_payload = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_map[contract],
            "service_count": service_count,
            "clv_proxy": clv_proxy,
            "senior_citizen": int(senior),
            "has_partner": int(partner),
            "digital_engagement": digital_engagement,
        }

        try:
            r1 = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            r2 = requests.post(f"{API_URL}/segment", json=uplift_payload, timeout=5)
            pred = r1.json()
            seg  = r2.json()

            churn_prob = pred["churn_probability"]
            color_icon = "🔴" if churn_prob > 0.7 else "🟡" if churn_prob > 0.4 else "🟢"

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{color_icon} Churn Probability", f"{churn_prob:.1%}")
            col2.metric("Uplift Score", f"{seg['uplift_score']:.3f}")
            col3.metric("Segment", seg["segment"])

            surv = pred["survival"]
            st.subheader("Survival Probability")
            surv_df = pd.DataFrame({
                "Time":     ["7 days", "30 days", "90 days"],
                "P(Active)": [surv["p_active_7d"], surv["p_active_30d"], surv["p_active_90d"]],
            })
            fig = px.bar(surv_df, x="Time", y="P(Active)",
                         color_discrete_sequence=["#3498DB"])
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            action_icon = {
                "Persuadable":  "🟢",
                "Sleeping_Dog": "🔴",
                "Lost_Cause":   "🟡",
                "Sure_Thing":   "🔵",
            }
            emoji = action_icon.get(seg["segment"], "⚪")
            st.success(
                f"{emoji} **Recommended Action:** {seg['recommended_action']} "
                f"| Priority: {seg['priority']}"
            )

        except Exception as e:
            st.error(f"API Error: {e} — Make sure FastAPI is running on port 8000")

# ─── PAGE 4: Model Performance ────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    # ── Lấy champion metrics động từ MLflow ──────────────────────────────────
    champions = load_champion_metrics()

    lgbm_score   = champions["lgbm"]["score"]
    cox_score    = champions["cox"]["score"]
    uplift_score = champions["uplift"]["score"]

    def _fmt_score(score, suffix="") -> str:
        return f"{score}{suffix}" if score is not None else "N/A"

    def _delta(score, baseline, suffix="") -> str | None:
        if score is None:
            return None
        diff = round(score - baseline, 4)
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff}{suffix} vs baseline"

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "LightGBM AUC-ROC  🏆",
        _fmt_score(lgbm_score),
        delta=_delta(lgbm_score, 0.80),
        help=f"Run ID: {champions['lgbm']['run_id']}  |  Trained: {champions['lgbm']['trained']}",
    )
    col2.metric(
        "CoxPH C-index  🏆",
        _fmt_score(cox_score),
        delta=_delta(cox_score, 0.90),
        help=f"Run ID: {champions['cox']['run_id']}  |  Trained: {champions['cox']['trained']}",
    )
    col3.metric(
        "T-Learner AUUC  🏆",
        _fmt_score(uplift_score),
        delta=_delta(uplift_score, 0.10),
        help=f"Run ID: {champions['uplift']['run_id']}  |  Trained: {champions['uplift']['trained']}",
    )

    if all(v["score"] is None for v in champions.values()):
        st.info(
            "Chưa tìm thấy Champion model nào có tag `promoted=true`. "
            "Hãy promote model trên MLflow hoặc kiểm tra kết nối DAGsHub."
        )

    st.divider()

    # ── SHAP Feature Importance ───────────────────────────────────────────────
    if df_shap is not None:
        st.subheader("Feature Importance (SHAP)")
        df_shap_sorted = df_shap.sort_values("mean_abs_shap", ascending=True).tail(15)
        fig = px.bar(
            df_shap_sorted, x="mean_abs_shap", y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP data not found at data/processed/shap_importance.csv")

    # ── Model Registry table ──────────────────────────────────────────────────
    st.subheader("Model Registry")
    model_df = load_model_registry_table()
    if model_df is not None and not model_df.empty:
        st.dataframe(model_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Chưa có model nào được train trên DAGsHub. "
            "Hãy trigger DAG Retrain Pipeline trên Airflow!"
        )

# ─── PAGE 5: Drift Monitor ────────────────────────────────────────────────────
elif page == "⚠️ Drift Monitor":
    st.title("⚠️ Data Drift Monitor")

    if drift_summary:
        drift_score    = drift_summary["drift_score"]
        drift_detected = drift_summary["drift_detected"]

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Drift Score",
            f"{drift_score:.3f}",
            delta="Above threshold" if drift_detected else "Normal",
            delta_color="inverse" if drift_detected else "normal",
        )
        col2.metric(
            "Drifted Columns",
            f"{drift_summary['n_drifted_columns']}/{drift_summary['n_total_columns']}",
        )
        col3.metric(
            "Status",
            "🚨 DRIFT DETECTED" if drift_detected else "✅ Stable",
        )

        if drift_detected:
            st.error("🚨 Data drift detected! Retrain pipeline will be triggered automatically.")
        else:
            st.success("✅ No significant drift detected.")

        st.divider()
        st.subheader("Drift Score Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=drift_score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#E74C3C" if drift_detected else "#2ECC71"},
                "steps": [
                    {"range": [0, 0.3],   "color": "#2ECC71"},
                    {"range": [0.3, 0.6], "color": "#F39C12"},
                    {"range": [0.6, 1.0], "color": "#E74C3C"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 0.3,
                },
            },
            title={"text": "Drift Score"},
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drift Details")
        st.json(drift_summary)
    else:
        st.info("No drift report found. Run drift_detector.py first.")

    st.divider()
    st.subheader("🧪 Simulate Drift (Demo)")
    if st.button("Inject Drift (Week 5 Scenario)"):
        import subprocess
        result = subprocess.run(
            [
                "python3", "-c",
                """
import pandas as pd
df = pd.read_csv('data/processed/features.csv').copy()
df['MonthlyCharges'] *= 1.3
df['tenure'] *= 0.7
df['charge_per_month'] *= 1.4
df['Contract_encoded'] = df['Contract_encoded'].apply(lambda x: max(0, x - 0.5))
df.to_csv('data/processed/features_latest.csv', index=False)
print('Done')
""",
            ],
            capture_output=True,
            text=True,
        )
        st.success("Drift injected! Re-run detector to see results.")