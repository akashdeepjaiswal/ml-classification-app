"""
app.py  –  Heart Disease Classification — Streamlit App
BITS Pilani NSP4 ML Assignment 2
"""

import os
import io
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Classifier | ML Assignment 2",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Main background: clean off-white ── */
.stApp { background: #f1f5f9; }

/* ── Sidebar: white with left border ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 2px solid #e2e8f0;
}
section[data-testid="stSidebar"] * { color: #1e293b !important; }

/* ── Metric Cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 1px 6px rgba(99,102,241,0.08);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.18);
    border-color: #6366f1;
}
.metric-card h2 {
    font-size: 2rem;
    font-weight: 700;
    margin: 6px 0 2px;
    color: #4f46e5;
}
.metric-card p {
    color: #64748b;
    font-size: 0.78rem;
    margin: 0;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 100%);
    border: 1.5px solid #c7d2fe;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
}
.hero h1 {
    font-size: 2.3rem;
    font-weight: 700;
    color: #4338ca;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}
.hero p { color: #64748b; font-size: 1.02rem; margin: 0; }

/* ── Section Headers ── */
.section-header {
    color: #4338ca;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #c7d2fe;
}

/* ── Winner Badge ── */
.winner-badge {
    display: inline-block;
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    color: #fff;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(245,158,11,0.25);
}

/* ── Table ── */
.dataframe { border-radius: 12px; overflow: hidden; }

/* ── Global text ── */
.stApp, .stMarkdown, p, span, label {
    color: #1e293b;
}
.stMarkdown h3 { color: #4338ca; }

/* Streamlit element overrides */
div[data-testid="stSelectbox"] label { color: #374151 !important; }
div[data-testid="stFileUploader"] label { color: #374151 !important; }

/* Radio buttons */
div[role="radiogroup"] label { color: #374151 !important; }

/* Code block */
.stCode { background: #f8fafc !important; border: 1px solid #e2e8f0; border-radius: 8px; }

/* Info / warning boxes */
div[data-testid="stInfo"] { background: #eff6ff; border-left: 4px solid #3b82f6; color: #1e3a8a; }
div[data-testid="stWarning"] { background: #fffbeb; border-left: 4px solid #f59e0b; color: #78350f; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree":        "model/decision_tree.pkl",
    "kNN":                  "model/knn.pkl",
    "Naive Bayes":          "model/naive_bayes.pkl",
    "Random Forest":        "model/random_forest.pkl",
}

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

MODEL_OBSERVATIONS = {
    "Logistic Regression": (
        "Achieved accuracy 0.8152 and AUC 0.8912 — a solid linear baseline. "
        "Assumes a linear decision boundary which slightly limits performance on this non-linear "
        "medical dataset. Feature coefficient analysis highlights `cp`, `ca`, and `thal` "
        "as the most influential predictors."
    ),
    "Decision Tree": (
        "Lowest overall performer (accuracy 0.7554, AUC 0.7519). Without depth pruning it "
        "overfits the training data, leading to the weakest generalisation on the test set. "
        "Most interpretable model — tree splits offer clinical insight, but not recommended "
        "for deployment alone."
    ),
    "kNN": (
        "Best overall model — highest accuracy (0.8696), F1 (0.8857), Precision (0.8611), "
        "Recall (0.9118), and MCC (0.7357). The dataset's moderate size and 13 well-scaled "
        "features suit a distance-based model. k=7 provides an excellent bias-variance tradeoff."
    ),
    "Naive Bayes": (
        "Accuracy 0.8261 with strong recall (0.8529). The Gaussian conditional independence "
        "assumption is imperfect for correlated clinical features but still generalises well. "
        "Best model when speed and simplicity are priorities — ideal for rapid screening."
    ),
    "Random Forest": (
        "Accuracy 0.8315 with the highest AUC (0.9029). Ensemble of 200 trees gives robust "
        "discrimination ability and best ranked risk-scoring performance. While kNN edges it "
        "on F1, Random Forest's superior AUC makes it the best choice when calibrated probability "
        "outputs matter. Features `ca`, `cp`, and `thal` rank highest in importance."
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")

@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_FILES[name])

@st.cache_resource
def load_all_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}

def compute_metrics(model, scaler, X: pd.DataFrame, y: pd.Series) -> dict:
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    y_prob = (
        model.predict_proba(Xs)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(Xs)
    )
    return {
        "Accuracy":  round(accuracy_score(y, y_pred), 4),
        "AUC":       round(roc_auc_score(y, y_prob), 4),
        "Precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y, y_pred), 4),
    }, y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        linewidths=1, linecolor="#f1f5f9",
        ax=ax, annot_kws={"size": 15, "weight": "bold", "color": "#1e293b"}
    )
    ax.set_xlabel("Predicted", color="#374151", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual", color="#374151", fontsize=11, labelpad=8)
    ax.set_title(f"Confusion Matrix — {model_name}", color="#1e293b", fontsize=12, pad=14, fontweight="bold")
    ax.tick_params(colors="#374151")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_prob, model_name: str) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.plot(fpr, tpr, color="#4f46e5", lw=2.5, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.2)
    ax.fill_between(fpr, tpr, alpha=0.10, color="#6366f1")
    ax.set_xlabel("False Positive Rate", color="#374151", labelpad=8)
    ax.set_ylabel("True Positive Rate", color="#374151", labelpad=8)
    ax.set_title(f"ROC Curve — {model_name}", color="#1e293b", fontsize=12, pad=14, fontweight="bold")
    ax.tick_params(colors="#374151")
    for sp in ax.spines.values(): sp.set_color("#e2e8f0")
    ax.legend(facecolor="#ffffff", edgecolor="#e2e8f0", labelcolor="#374151", fontsize=10)
    plt.tight_layout()
    return fig

def plot_feature_importance(model, model_name: str):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return None
    sorted_idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    # Indigo-to-rose gradient palette
    palette = ["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe",
               "#f43f5e", "#fb7185", "#fda4af", "#fecdd3",
               "#0ea5e9", "#38bdf8", "#7dd3fc", "#bae6fd", "#e0f2fe"]
    colors = palette[:len(FEATURES)]
    ax.bar(
        [FEATURES[i] for i in sorted_idx],
        imp[sorted_idx],
        color=[colors[j % len(colors)] for j in range(len(FEATURES))],
        edgecolor="#ffffff", linewidth=0.8
    )
    ax.set_title(f"Feature Importances — {model_name}", color="#1e293b", fontsize=12, pad=14, fontweight="bold")
    ax.set_ylabel("Importance", color="#374151", labelpad=8)
    ax.tick_params(colors="#374151", axis="both")
    ax.tick_params(axis="x", rotation=45)
    for sp in ax.spines.values(): sp.set_color("#e2e8f0")
    ax.yaxis.grid(True, color="#f1f5f9", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

def plot_model_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    metrics_to_plot = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    x = np.arange(len(metrics_df))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    palette = ["#6366f1", "#06b6d4", "#f43f5e", "#10b981", "#f59e0b", "#8b5cf6"]
    for i, (metric, color) in enumerate(zip(metrics_to_plot, palette)):
        ax.bar(x + i * width, metrics_df[metric], width, label=metric, color=color,
               alpha=0.88, edgecolor="#ffffff", linewidth=0.6)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(metrics_df.index, rotation=15, ha="right", color="#374151", fontsize=9.5)
    ax.tick_params(colors="#374151")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", color="#374151", labelpad=8)
    ax.set_title("Model Comparison — All Metrics", color="#1e293b", fontsize=13, pad=14, fontweight="bold")
    for sp in ax.spines.values(): sp.set_color("#e2e8f0")
    ax.yaxis.grid(True, color="#f1f5f9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(facecolor="#ffffff", edgecolor="#e2e8f0", labelcolor="#374151", fontsize=9.5)
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Disease Classifier")
    st.markdown("---")
    st.markdown("**BITS Pilani | NSP4 | ML Assignment 2**")
    st.markdown("**Dataset:** UCI Heart Disease (920 rows, 13 features)")
    st.markdown("---")

    # ── File Upload (Required feature 1) ──────────────────────────────────────
    st.markdown("### 📂 Upload Test Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload a CSV with the same columns as the dataset",
        type=["csv"],
        help="Upload test_data.csv or any compatible CSV file"
    )

    # ── Model Selection (Required feature 2) ──────────────────────────────────
    st.markdown("### 🤖 Select Model")
    selected_model_name = st.selectbox(
        "Choose a classification model:",
        list(MODEL_FILES.keys()),
        index=4,  # Default: Random Forest
    )

    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio(
        "Go to",
        ["🏠 Overview", "📊 Model Analysis", "📈 All Models Comparison"],
        label_visibility="collapsed"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Load models & scaler
# ──────────────────────────────────────────────────────────────────────────────
models_available = all(os.path.exists(p) for p in MODEL_FILES.values()) and os.path.exists("model/scaler.pkl")

if not models_available:
    st.error("⚠️ Model files not found. Please run `python train_models.py` first.")
    st.stop()

scaler = load_scaler()
selected_model = load_model(selected_model_name)

# ──────────────────────────────────────────────────────────────────────────────
# Determine data source
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(df_input)} rows from upload")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        df_input = None
elif os.path.exists("test_data.csv"):
    df_input = pd.read_csv("test_data.csv")
    st.sidebar.info("Using built-in test_data.csv")
else:
    df_input = None

# Parse X / y
if df_input is not None and "target" in df_input.columns:
    X_input = df_input[FEATURES].copy()
    y_input = df_input["target"].copy()
    data_ready = True
elif df_input is not None:
    # No target column — prediction only
    X_input = df_input[FEATURES].copy() if all(f in df_input.columns for f in FEATURES) else None
    y_input = None
    data_ready = X_input is not None
else:
    X_input = None
    y_input = None
    data_ready = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
        <h1>🫀 Heart Disease Classification</h1>
        <p>End-to-end ML classification app · 5 models · 6 evaluation metrics · Interactive analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card"><p>Dataset</p><h2>UCI</h2><p>Heart Disease · 920 rows · 13 features</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><p>Models</p><h2>5</h2><p>LR · DT · kNN · NB · RF</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><p>Metrics</p><h2>6</h2><p>Accuracy · AUC · Precision · Recall · F1 · MCC</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">About the Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    The **UCI Heart Disease Dataset** is a classic benchmark for binary classification in medical ML.
    It merges four regional studies (Cleveland, VA, Switzerland, Hungary) into 920 patient records with 13 clinical features:

    | Feature | Description |
    |---|---|
    | age | Age in years |
    | sex | Sex (1 = male, 0 = female) |
    | cp | Chest pain type (0–3) |
    | trestbps | Resting blood pressure (mm Hg) |
    | chol | Serum cholesterol (mg/dl) |
    | fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
    | restecg | Resting ECG results (0–2) |
    | thalach | Maximum heart rate achieved |
    | exang | Exercise-induced angina (1 = yes) |
    | oldpeak | ST depression induced by exercise |
    | slope | Slope of peak exercise ST segment |
    | ca | Number of major vessels (0–3) |
    | thal | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |
    | **target** | **0 = No Disease, 1 = Disease** |
    """)

    st.markdown('<div class="section-header">Models Implemented</div>', unsafe_allow_html=True)
    for name, obs in MODEL_OBSERVATIONS.items():
        with st.expander(f"📌 {name}"):
            st.write(obs)

    if data_ready:
        st.markdown('<div class="section-header">Quick Prediction Preview</div>', unsafe_allow_html=True)
        Xs = scaler.transform(X_input.head(5))
        preds = selected_model.predict(Xs)
        preview = X_input.head(5).copy()
        if y_input is not None:
            preview["Actual"] = y_input.values[:5]
        preview["Predicted"] = preds
        preview["Predicted_Label"] = preview["Predicted"].map({0: "No Disease ✅", 1: "Disease ⚠️"})
        st.dataframe(preview, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Analysis":
    st.markdown(f"""
    <div class="hero">
        <h1>📊 {selected_model_name}</h1>
        <p>Detailed evaluation metrics, confusion matrix, ROC curve and feature analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready or y_input is None:
        st.warning("⚠️ Please upload a CSV file with a `target` column to see full evaluation metrics.")
        st.info("You can use the built-in `test_data.csv` from the repository.")
    else:
        metrics, y_pred, y_prob = compute_metrics(selected_model, scaler, X_input, y_input)

        # ── Evaluation Metrics (Required feature 3) ───────────────────────────
        st.markdown('<div class="section-header">📐 Evaluation Metrics</div>', unsafe_allow_html=True)
        cols = st.columns(6)
        metric_icons = {"Accuracy": "🎯", "AUC": "📈", "Precision": "🔬", "Recall": "🔭", "F1": "⚖️", "MCC": "🧲"}
        for col, (metric, value) in zip(cols, metrics.items()):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <p>{metric_icons[metric]} {metric}</p>
                    <h2>{value:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Confusion Matrix + ROC (Required feature 4) ──────────────────────
        st.markdown('<div class="section-header">📊 Confusion Matrix & ROC Curve</div>', unsafe_allow_html=True)
        col_cm, col_roc = st.columns(2)
        with col_cm:
            fig_cm = plot_confusion_matrix(y_input, y_pred, selected_model_name)
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        with col_roc:
            fig_roc = plot_roc_curve(y_input, y_prob, selected_model_name)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        # ── Classification Report ─────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_input, y_pred, target_names=["No Disease", "Disease"])
        st.code(report, language="text")

        # ── Feature Importance ────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔍 Feature Analysis</div>', unsafe_allow_html=True)
        fig_fi = plot_feature_importance(selected_model, selected_model_name)
        if fig_fi:
            st.pyplot(fig_fi)
            plt.close(fig_fi)
        else:
            st.info(f"Feature importance is not available for {selected_model_name}.")

        # ── Model Observation ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">💡 Model Observations</div>', unsafe_allow_html=True)
        st.info(MODEL_OBSERVATIONS[selected_model_name])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: All Models Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 All Models Comparison":
    st.markdown("""
    <div class="hero">
        <h1>📈 All Models Comparison</h1>
        <p>Side-by-side evaluation of all 5 classification models</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready or y_input is None:
        st.warning("⚠️ Upload a CSV with a `target` column to compare all models.")
    else:
        all_models = load_all_models()
        all_results = {}
        for name, mdl in all_models.items():
            m, _, _ = compute_metrics(mdl, scaler, X_input, y_input)
            all_results[name] = m

        results_df = pd.DataFrame(all_results).T
        results_df.index.name = "ML Model Name"

        # ── Comparison Table ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Metrics Comparison Table</div>', unsafe_allow_html=True)

        # Highlight best in each column
        def highlight_max(s):
            is_max = s == s.max()
            return ["background-color: rgba(129,140,248,0.3); font-weight:bold;" if v else "" for v in is_max]

        styled = (
            results_df.style
            .apply(highlight_max, axis=0)
            .format("{:.4f}")
            .set_table_styles([
                {"selector": "th", "props": [("background", "#302b63"), ("color", "white")]},
                {"selector": "td", "props": [("color", "white")]},
            ])
        )
        st.dataframe(styled, use_container_width=True)

        # Winner
        winner = results_df["F1"].idxmax()
        st.markdown(f"🏆 **Overall Winner (Best F1):** <span class='winner-badge'>{winner}</span>", unsafe_allow_html=True)

        # ── Bar Chart Comparison ──────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Visual Comparison</div>', unsafe_allow_html=True)
        fig_comp = plot_model_comparison(results_df)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        # ── Observations Table ────────────────────────────────────────────────
        st.markdown('<div class="section-header">💡 Model Observations</div>', unsafe_allow_html=True)
        obs_data = [(name, obs) for name, obs in MODEL_OBSERVATIONS.items()]
        obs_df = pd.DataFrame(obs_data, columns=["ML Model Name", "Observation about model performance"])
        obs_df = obs_df.set_index("ML Model Name")
        st.table(obs_df)

        # ── Individual Confusion Matrices ─────────────────────────────────────
        st.markdown('<div class="section-header">🔲 All Confusion Matrices</div>', unsafe_allow_html=True)
        Xs = scaler.transform(X_input)
        cm_cols = st.columns(3)
        for idx, (name, mdl) in enumerate(all_models.items()):
            y_pred_m = mdl.predict(Xs)
            fig_c = plot_confusion_matrix(y_input, y_pred_m, name)
            cm_cols[idx % 3].pyplot(fig_c)
            plt.close(fig_c)
