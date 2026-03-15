import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import requests
from pathlib import Path
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prevention Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"
DATA_DIR = Path("data/processed")

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_scores():
    path = DATA_DIR / "batch_scores_latest.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=300)
def load_action_plan():
    path = DATA_DIR / "action_plan.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=300)
def load_shap():
    path = DATA_DIR / "shap_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def load_drift_summary():
    path = DATA_DIR / "drift_reports/drift_summary_latest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/bullseye-emoji.png", width=60)
st.sidebar.title("Churn Prevention")
st.sidebar.markdown("**MLOps Dashboard v1.0**")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "🎯 Action Plan",
    "🔍 Customer Lookup",
    "📈 Model Performance",
    "⚠️  Drift Monitor",
])

budget = st.sidebar.slider("Monthly Budget ($)", 1000, 20000, 5000, 500)
st.sidebar.divider()
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ─── Load all data ────────────────────────────────────────────────────────────
df_scores = load_scores()
df_plan = load_action_plan()
df_shap = load_shap()
drift_summary = load_drift_summary()

# ─── PAGE 1: Overview ─────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Customer Churn Overview")

    if df_scores is None:
        st.warning("No batch scores found. Run batch_scoring DAG first.")
        st.stop()

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_scores)
    high_risk = (df_scores['churn_probability'] > 0.7).sum()
    persuadable = (df_scores['segment'] == 'Persuadable').sum()
    avg_churn = df_scores['churn_probability'].mean()

    col1.metric("Total Customers", f"{total:,}")
    col2.metric("High Risk (>70%)", f"{high_risk:,}",
                delta=f"{high_risk/total*100:.1f}%", delta_color="inverse")
    col3.metric("Persuadable", f"{persuadable:,}",
                delta=f"{persuadable/total*100:.1f}%")
    col4.metric("Avg Churn Prob", f"{avg_churn:.1%}")

    st.divider()
    col_left, col_right = st.columns(2)

    # Segment distribution
    with col_left:
        st.subheader("Segment Distribution")
        seg_counts = df_scores['segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        colors = {
            'Persuadable': '#2ECC71',
            'Sure_Thing': '#3498DB',
            'Lost_Cause': '#E74C3C',
            'Sleeping_Dog': '#F39C12'
        }
        fig = px.pie(seg_counts, values='Count', names='Segment',
                     color='Segment',
                     color_discrete_map=colors)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # Churn probability distribution
    with col_right:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(df_scores, x='churn_probability',
                          nbins=50, color_discrete_sequence=['#E74C3C'])
        fig.add_vline(x=0.446, line_dash="dash",
                      annotation_text="Threshold 0.446")
        fig.update_layout(xaxis_title="Churn Probability",
                         yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Uplift vs Churn scatter
    st.subheader("Uplift Score vs Churn Probability")
    fig = px.scatter(df_scores, x='churn_probability', y='uplift_score',
                    color='segment', color_discrete_map=colors,
                    hover_data=['user_id', 'clv_proxy'],
                    opacity=0.6)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.446, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGE 2: Action Plan ──────────────────────────────────────────────────────
elif page == "🎯 Action Plan":
    st.title("🎯 Action Plan — Budget Optimizer")

    if df_plan is None:
        st.warning("No action plan found. Run decision engine first.")
        st.stop()

    # Filter by budget
    persuadable = df_plan[df_plan['segment'] == 'Persuadable'].copy()
    persuadable = persuadable.sort_values('roi', ascending=False)
    persuadable['cumulative_cost'] = persuadable['action_cost'].cumsum()
    within_budget = persuadable[persuadable['cumulative_cost'] <= budget]

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users to Target", f"{len(within_budget):,}")
    col2.metric("Total Cost", f"${within_budget['action_cost'].sum():,.0f}")
    col3.metric("Expected CLV Saved",
                f"${within_budget['expected_clv_saved'].sum():,.0f}")
    col4.metric("Avg ROI",
                f"{within_budget['roi'].mean():.1f}x" if len(within_budget) > 0 else "N/A")

    st.divider()
    col_left, col_right = st.columns(2)

    # Action breakdown
    with col_left:
        st.subheader("Action Breakdown")
        if len(within_budget) > 0:
            action_counts = within_budget['recommended_action'].value_counts().reset_index()
            action_counts.columns = ['Action', 'Count']
            fig = px.bar(action_counts, x='Action', y='Count',
                        color='Action',
                        color_discrete_sequence=['#2ECC71', '#3498DB'])
            st.plotly_chart(fig, use_container_width=True)

    # ROI distribution
    with col_right:
        st.subheader("ROI Distribution")
        if len(within_budget) > 0:
            fig = px.histogram(within_budget, x='roi', nbins=30,
                              color_discrete_sequence=['#2ECC71'])
            fig.update_layout(xaxis_title="ROI", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    # Target list
    st.subheader(f"Target List ({len(within_budget)} users)")
    if len(within_budget) > 0:
        display_cols = ['user_id', 'churn_probability', 'uplift_score',
                       'recommended_action', 'action_cost',
                       'expected_clv_saved', 'roi']
        display_cols = [c for c in display_cols if c in within_budget.columns]
        st.dataframe(
            within_budget[display_cols].head(100).style.background_gradient(
                subset=['churn_probability'], cmap='RdYlGn_r'
            ).background_gradient(subset=['roi'], cmap='Greens'),
            use_container_width=True
        )

# ─── PAGE 3: Customer Lookup ──────────────────────────────────────────────────
elif page == "🔍 Customer Lookup":
    st.title("🔍 Customer Lookup — Real-time Prediction")

    st.info("Nhập thông tin khách hàng để nhận dự đoán real-time từ FastAPI")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 3)
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 95)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        senior = st.checkbox("Senior Citizen")
        partner = st.checkbox("Has Partner")
    with col3:
        service_count = st.slider("Number of Services", 0, 8, 3)
        digital_engagement = st.slider("Digital Engagement", 0, 2, 1)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}

    total_charges = monthly_charges * max(tenure, 1)
    charge_per_month = monthly_charges / (tenure + 1)
    clv_proxy = total_charges * (1 - 1/(tenure+1))

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
            "recency_risk": 1/(tenure+1),
            "service_count": service_count,
            "monetary_value": total_charges,
            "monthly_to_total_ratio": monthly_charges/(total_charges+1),
            "charge_per_month": charge_per_month,
            "clv_proxy": clv_proxy,
            "is_high_value": int(clv_proxy > 500),
            "contract_stability": contract_map[contract] * tenure,
            "digital_engagement": digital_engagement
        }

        uplift_payload = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_map[contract],
            "service_count": service_count,
            "clv_proxy": clv_proxy,
            "senior_citizen": int(senior),
            "has_partner": int(partner),
            "digital_engagement": digital_engagement
        }

        try:
            r1 = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            r2 = requests.post(f"{API_URL}/segment", json=uplift_payload, timeout=5)
            pred = r1.json()
            seg = r2.json()

            col1, col2, col3 = st.columns(3)
            churn_prob = pred['churn_probability']
            color = "🔴" if churn_prob > 0.7 else "🟡" if churn_prob > 0.4 else "🟢"

            col1.metric(f"{color} Churn Probability",
                       f"{churn_prob:.1%}")
            col2.metric("Uplift Score", f"{seg['uplift_score']:.3f}")
            col3.metric("Segment", seg['segment'])

            # Survival curve
            surv = pred['survival']
            st.subheader("Survival Probability")
            surv_df = pd.DataFrame({
                'Time': ['7 days', '30 days', '90 days'],
                'P(Active)': [surv['p_active_7d'],
                              surv['p_active_30d'],
                              surv['p_active_90d']]
            })
            fig = px.bar(surv_df, x='Time', y='P(Active)',
                        color_discrete_sequence=['#3498DB'])
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Recommended action
            action_color = {
                'Persuadable': '🟢',
                'Sleeping_Dog': '🔴',
                'Lost_Cause': '🟡',
                'Sure_Thing': '🔵'
            }
            emoji = action_color.get(seg['segment'], '⚪')
            st.success(f"{emoji} **Recommended Action:** {seg['recommended_action']} | Priority: {seg['priority']}")

        except Exception as e:
            st.error(f"API Error: {e} — Make sure FastAPI is running on port 8000")

# ─── PAGE 4: Model Performance ────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("LightGBM AUC-ROC", "0.8445", delta="+0.0445 vs baseline")
    col2.metric("CoxPH C-index", "0.9415", delta="Excellent")
    col3.metric("T-Learner AUUC", "0.2014", delta="2.2x vs random")

    st.divider()

    # SHAP importance
    if df_shap is not None:
        st.subheader("Feature Importance (SHAP)")
        df_shap_sorted = df_shap.sort_values('mean_abs_shap', ascending=True).tail(15)
        fig = px.bar(df_shap_sorted, x='mean_abs_shap', y='feature',
                    orientation='h',
                    color='mean_abs_shap',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(showlegend=False,
                         xaxis_title="Mean |SHAP Value|",
                         yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP data not found at data/processed/shap_importance.csv")

    # Model Registry — pull từ MLflow thật
    st.subheader("Model Registry")
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        exp = client.get_experiment_by_name('churn-prediction')
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=['start_time DESC']
        )

        rows = []
        for run in runs:
            name = run.info.run_name
            metrics = run.data.metrics
            tags = run.data.tags
            start = run.info.start_time
            trained = pd.Timestamp(start, unit='ms').strftime('%Y-%m-%d %H:%M') if start else 'N/A'
            promoted = tags.get('promoted', 'true')
            status = '✅ Production' if promoted != 'false' else '🔄 Not promoted'

            if 'lightgbm' in name:
                rows.append({'Model': name, 'Metric': 'AUC-ROC',
                             'Score': round(metrics.get('auc_roc', 0), 4),
                             'Status': status, 'Trained': trained,
                             'Run ID': run.info.run_id[:8]})
            elif 'coxph' in name:
                rows.append({'Model': name, 'Metric': 'C-index',
                             'Score': round(metrics.get('c_index', 0), 4),
                             'Status': status, 'Trained': trained,
                             'Run ID': run.info.run_id[:8]})
            elif 'uplift' in name:
                rows.append({'Model': name, 'Metric': 'AUUC',
                             'Score': round(metrics.get('auuc', 0), 4),
                             'Status': status, 'Trained': trained,
                             'Run ID': run.info.run_id[:8]})

        if rows:
            model_df = pd.DataFrame(rows)
            st.dataframe(model_df, use_container_width=True, hide_index=True)
        else:
            st.info("No runs found in MLflow")
    except Exception as e:
        st.error(f"Cannot connect to MLflow: {e}")

# ─── PAGE 5: Drift Monitor ────────────────────────────────────────────────────
elif page == "⚠️  Drift Monitor":
    st.title("⚠️ Data Drift Monitor")

    if drift_summary:
        drift_score = drift_summary['drift_score']
        drift_detected = drift_summary['drift_detected']

        col1, col2, col3 = st.columns(3)
        col1.metric("Drift Score",
                   f"{drift_score:.3f}",
                   delta="Above threshold" if drift_detected else "Normal",
                   delta_color="inverse" if drift_detected else "normal")
        col2.metric("Drifted Columns",
                   f"{drift_summary['n_drifted_columns']}/{drift_summary['n_total_columns']}")
        col3.metric("Status",
                   "🚨 DRIFT DETECTED" if drift_detected else "✅ Stable")

        if drift_detected:
            st.error("🚨 Data drift detected! Retrain pipeline will be triggered automatically.")
        else:
            st.success("✅ No significant drift detected.")

        st.divider()

        # Drift gauge
        st.subheader("Drift Score Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=drift_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': '#E74C3C' if drift_detected else '#2ECC71'},
                'steps': [
                    {'range': [0, 0.3], 'color': '#2ECC71'},
                    {'range': [0.3, 0.6], 'color': '#F39C12'},
                    {'range': [0.6, 1.0], 'color': '#E74C3C'},
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value': 0.3
                }
            },
            title={'text': "Drift Score"}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drift Details")
        st.json(drift_summary)
    else:
        st.info("No drift report found. Run drift_detector.py first.")

    # Trigger drift simulation
    st.divider()
    st.subheader("🧪 Simulate Drift (Demo)")
    if st.button("Inject Drift (Week 5 Scenario)"):
        import subprocess
        result = subprocess.run([
            'python3', '-c', """
import pandas as pd
df = pd.read_csv('data/processed/features.csv').copy()
df['MonthlyCharges'] *= 1.3
df['tenure'] *= 0.7
df['charge_per_month'] *= 1.4
df['Contract_encoded'] = df['Contract_encoded'].apply(lambda x: max(0, x - 0.5))
df.to_csv('data/processed/features_latest.csv', index=False)
print('Done')
"""
        ], capture_output=True, text=True)
        st.success("Drift injected! Re-run detector to see results.")
