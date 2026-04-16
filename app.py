import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Predictor", page_icon="💉", layout="wide")

# ── Load Models & Artifacts ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf  = pickle.load(open("rf_model.pkl",  "rb"))
    svm = pickle.load(open("svm_model.pkl", "rb"))
    lr  = pickle.load(open("lr_model.pkl",  "rb"))
    return {"Random Forest": rf, "SVM": svm, "Logistic Regression": lr}

@st.cache_data
def load_metrics():
    return pickle.load(open("model_metrics.pkl", "rb"))

@st.cache_data
def load_test_data():
    return pickle.load(open("test_data.pkl", "rb"))

models  = load_models()
metrics = load_metrics()
X_test, y_test = load_test_data()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3               { font-family: 'IBM Plex Mono', monospace !important; }

div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
div[data-testid="stMetric"] label { color: #8b949e !important; font-size: 0.8rem; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; letter-spacing:-1px;'>💉 Diabetes Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8b949e; font-size:0.95rem;'>ML-powered risk assessment · Best model: Random Forest</p>", unsafe_allow_html=True)
st.write("---")

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

def slider_input(label, min_val, max_val, default, step=1):
    c1, c2 = st.columns([4, 1])
    with c1:
        val = st.slider(label, min_val, max_val, default, step=step)
    with c2:
        val = st.number_input(f"##{label}", min_val, max_val, val, step=step, label_visibility="hidden")
    return val

with col1:
    st.subheader("🧾 Patient Details")
    pregnancies = st.number_input("Pregnancies", 0, 20, value=1)
    glucose     = slider_input("Glucose", 0, 200, 100)
    bp          = slider_input("Blood Pressure", 0, 150, 70)
    skin        = slider_input("Skin Thickness", 0, 100, 20)

with col2:
    st.subheader("📊 Health Metrics")
    insulin = slider_input("Insulin", 0, 900, 80)
    bmi     = slider_input("BMI", 0.0, 50.0, 25.0, step=0.1)
    dpf     = slider_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
    age     = slider_input("Age", 1, 120, 30)

st.write("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    rf_model   = models["Random Forest"]
    result     = rf_model.predict(input_data)
    prob       = rf_model.predict_proba(input_data)
    risk_pct   = prob[0][1] * 100

    st.subheader("🧠 Prediction Result")
    res_col, gauge_col = st.columns(2, gap="large")

    with res_col:
        if result[0] == 1:
            st.error(f"⚠️ **High Risk** — {risk_pct:.1f}% probability of diabetes")
        else:
            st.success(f"✅ **Low Risk** — {100 - risk_pct:.1f}% probability of no diabetes")

        # Feature Importance (RF native)
        feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                         "Insulin", "BMI", "Pedigree Fn", "Age"]
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance (%)": np.round(rf_model.feature_importances_ * 100, 2)
        }).sort_values("Importance (%)", ascending=True)

        fig_bar = px.bar(
            imp_df, x="Importance (%)", y="Feature", orientation="h",
            title="Feature Importance (Random Forest)",
            color="Importance (%)", color_continuous_scale="reds",
            template="plotly_dark",
        )
        fig_bar.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0), height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 36, "family": "IBM Plex Mono"}},
            title={"text": "Diabetes Risk Score", "font": {"size": 16}},
            delta={"reference": 50,
                   "increasing": {"color": "#f85149"},
                   "decreasing": {"color": "#3fb950"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8b949e"},
                "bar": {"color": "#f85149" if result[0] == 1 else "#3fb950", "thickness": 0.3},
                "bgcolor": "#161b22", "borderwidth": 0,
                "steps": [
                    {"range": [0,  30], "color": "#0d1117"},
                    {"range": [30, 60], "color": "#161b22"},
                    {"range": [60, 100], "color": "#1c1c1c"},
                ],
                "threshold": {"line": {"color": "#f0883e", "width": 3},
                               "thickness": 0.8, "value": 50},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e6edf3", "family": "IBM Plex Sans"},
            margin=dict(l=20, r=20, t=40, b=20), height=300,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

st.write("---")

# ── Model Comparison ──────────────────────────────────────────────────────────
st.subheader("🏆 Model Comparison")
st.markdown("<p style='color:#8b949e;'>All 3 models · 80/20 train-test split · stratified</p>", unsafe_allow_html=True)

model_keys  = {"rf": "Random Forest", "svm": "SVM", "lr": "Logistic Regression"}
metric_list = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
colors      = {"rf": "#3fb950", "svm": "#f0883e", "lr": "#58a6ff"}

m_col1, m_col2, m_col3 = st.columns(3)
for col, (key, name) in zip([m_col1, m_col2, m_col3], model_keys.items()):
    with col:
        is_best = key == "rf"
        st.markdown(f"<h4 style='color:{colors[key]};'>{'🥇 ' if is_best else ''}{name}</h4>", unsafe_allow_html=True)
        for metric in metric_list:
            st.metric(metric, f"{metrics[key][metric]}%")

st.write("")

# Grouped bar chart
fig_compare = go.Figure()
for key, name in model_keys.items():
    vals = [metrics[key][m] for m in metric_list]
    fig_compare.add_trace(go.Bar(
        name=name, x=metric_list, y=vals,
        marker_color=colors[key],
        text=[f"{v}%" for v in vals], textposition="outside",
    ))

fig_compare.update_layout(
    barmode="group", template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=0, r=0, t=40, b=0), height=380,
    yaxis=dict(range=[0, 115], title="Score (%)"),
    font={"family": "IBM Plex Sans"},
)
st.plotly_chart(fig_compare, use_container_width=True)

# ── ROC Curve ─────────────────────────────────────────────────────────────────
st.write("### ROC Curve")
fig_roc = go.Figure()

for key, name in model_keys.items():
    probas = models[name].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probas)
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"{name} (AUC {metrics[key]['ROC AUC']}%)",
        line=dict(color=colors[key], width=2.5),
    ))

fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="Random Chance",
    line=dict(color="#8b949e", width=1.5, dash="dash"),
))

fig_roc.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    legend=dict(orientation="h", y=-0.2),
    margin=dict(l=0, r=0, t=10, b=0), height=420,
    font={"family": "IBM Plex Sans"},
)
st.plotly_chart(fig_roc, use_container_width=True)

# ── Dataset Insights ──────────────────────────────────────────────────────────
st.write("---")
st.subheader("📊 Dataset Insights")
df = pd.read_csv("diabetes_prediction.csv")

with st.expander("📋 Data Preview", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("📈 Summary Statistics", expanded=False):
    st.dataframe(df.describe().T.style.background_gradient(cmap="Reds"), use_container_width=True)

st.write("### Correlation Heatmap")
corr = df.corr()
fig_heatmap = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(), y=corr.columns.tolist(),
    colorscale=[[0.0, "#1168b0"], [0.5, "#0d1117"], [1.0, "#f85149"]],
    zmin=-1, zmax=1,
    text=np.round(corr.values, 2), texttemplate="%{text}",
    textfont={"size": 11, "family": "IBM Plex Mono"},
))
fig_heatmap.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0), height=480,
    font={"family": "IBM Plex Sans"},
)
st.plotly_chart(fig_heatmap, use_container_width=True)