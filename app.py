import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc
)

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data/employee_attrition.csv"
MODEL_DIR = Path("models")
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


# ----------------------------------------------------------------------------
# DATA / MODEL LOADING (cached so it only runs once per session)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def _load_pretrained(df: pd.DataFrame, selected_features, algorithm, max_depth):
    """Use artifacts saved by train.py, but only when the sidebar selections match
    exactly what train.py trained on (full default feature set, Decision Tree, depth 5).
    Otherwise return None so the caller trains live instead."""
    required = ["model.pkl", "encoders.pkl", "feature_names.pkl"]
    if not all((MODEL_DIR / f).exists() for f in required):
        return None
    if algorithm != "Decision Tree" or max_depth != 5:
        return None
    try:
        with open(MODEL_DIR / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODEL_DIR / "encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open(MODEL_DIR / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except Exception:
        return None

    if list(feature_names) != list(selected_features):
        return None

    work = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    for col, le in encoders.items():
        work[col] = le.transform(work[col])
    X = work[feature_names]
    y = work["Attrition"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return {
        "model": model, "encoders": encoders, "feature_names": list(feature_names),
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob,
        "metrics": metrics, "source": "pretrained",
    }


@st.cache_resource
def train_model(df: pd.DataFrame, selected_features: tuple, algorithm: str,
                 max_depth: int = 5, n_estimators: int = 200):
    selected_features = list(selected_features)

    work = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

    encoders = {}
    for col in work.select_dtypes(include="object").columns:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col])
        encoders[col] = le

    X = work[selected_features]
    y = work["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if algorithm == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
        )
    else:
        model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return {
        "model": model,
        "encoders": encoders,
        "feature_names": selected_features,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "source": "live",
    }


df_raw = load_data()

# Full pool of usable features (everything except the always-dropped ID-like
# columns and the target itself).
ALL_FEATURES = [c for c in df_raw.columns if c not in DROP_COLS + ["Attrition"]]

# ----------------------------------------------------------------------------
# SIDEBAR: model configuration
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Configuration")

algorithm = st.sidebar.selectbox("Algorithm", ["Decision Tree", "Random Forest"])

if algorithm == "Random Forest":
    n_estimators = st.sidebar.slider("Number of trees (n_estimators)", 50, 500, 200, step=50)
else:
    n_estimators = 200  # unused for Decision Tree, kept for a stable cache key

max_depth = st.sidebar.slider("Max tree depth", 2, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Feature Selection")
select_all = st.sidebar.checkbox("Select all features", value=True)
if select_all:
    selected_features = ALL_FEATURES
else:
    selected_features = st.sidebar.multiselect(
        "Choose features to train on", options=ALL_FEATURES, default=ALL_FEATURES[:8]
    )

if len(selected_features) < 2:
    st.sidebar.error("Select at least 2 features to train a model.")
    st.stop()

pretrained = _load_pretrained(df_raw, selected_features, algorithm, max_depth)
if pretrained is not None:
    bundle = pretrained
    st.sidebar.success("Using model trained by train.py")
else:
    bundle = train_model(
        df_raw, tuple(selected_features), algorithm,
        max_depth=max_depth, n_estimators=n_estimators,
    )
    st.sidebar.info(f"Trained live: {algorithm}, depth={max_depth}"
                     + (f", trees={n_estimators}" if algorithm == "Random Forest" else "")
                     + f", {len(selected_features)} features")

st.sidebar.markdown("---")
st.sidebar.markdown("### Filter data (EDA tab)")
dept_filter = st.sidebar.multiselect(
    "Department", options=sorted(df_raw["Department"].unique()), default=None
)
gender_filter = st.sidebar.multiselect(
    "Gender", options=sorted(df_raw["Gender"].unique()), default=None
)

df = df_raw.copy()
if dept_filter:
    df = df[df["Department"].isin(dept_filter)]
if gender_filter:
    df = df[df["Gender"].isin(gender_filter)]

st.sidebar.markdown("---")
st.sidebar.caption("IBM HR Analytics Employee Attrition dataset.")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("📊 Employee Attrition Analytics Dashboard")
st.caption("Explore workforce attrition patterns, model performance, and predict attrition risk for individual employees.")

tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ["🏠 Overview", "🔍 Explore Data", "🤖 Model Performance", "🎯 Predict Attrition"]
)

# ----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------------------------
with tab_overview:
    total_emp = len(df)
    attr_rate = (df["Attrition"] == "Yes").mean() * 100
    avg_age = df["Age"].mean()
    avg_income = df["MonthlyIncome"].mean()
    avg_tenure = df["YearsAtCompany"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees", f"{total_emp:,}")
    c2.metric("Attrition Rate", f"{attr_rate:.1f}%")
    c3.metric("Avg. Age", f"{avg_age:.1f} yrs")
    c4.metric("Avg. Monthly Income", f"${avg_income:,.0f}")
    c5.metric("Avg. Tenure", f"{avg_tenure:.1f} yrs")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            df, names="Attrition", title="Attrition Split",
            color="Attrition", color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dept_attr = (
            df.groupby("Department")["Attrition"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="AttritionRate")
        )
        fig = px.bar(
            dept_attr, x="Department", y="AttritionRate",
            title="Attrition Rate by Department (%)", text_auto=".1f",
            color="AttritionRate", color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        ot_attr = (
            df.groupby("OverTime")["Attrition"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="AttritionRate")
        )
        fig = px.bar(
            ot_attr, x="OverTime", y="AttritionRate", color="OverTime",
            title="Attrition Rate: OverTime vs No OverTime (%)", text_auto=".1f",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            df, x="Age", color="Attrition", barmode="overlay", nbins=25,
            title="Age Distribution by Attrition",
            color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: EXPLORE DATA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Interactive Exploration")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["EmployeeCount", "EmployeeNumber", "StandardHours"]]

    colx, coly = st.columns(2)
    with colx:
        x_axis = st.selectbox("X-axis (numeric)", numeric_cols, index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0)
    with coly:
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        cat_cols = [c for c in cat_cols if c != "Attrition"]
        split_by = st.selectbox("Split by (category)", cat_cols, index=cat_cols.index("JobRole") if "JobRole" in cat_cols else 0)

    fig = px.box(
        df, x=split_by, y=x_axis, color="Attrition",
        title=f"{x_axis} by {split_by} and Attrition",
        color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        fig = px.histogram(
            df, x="JobSatisfaction", color="Attrition", barmode="group",
            title="Job Satisfaction vs Attrition",
            color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        fig = px.histogram(
            df, x="WorkLifeBalance", color="Attrition", barmode="group",
            title="Work-Life Balance vs Attrition",
            color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Correlation Heatmap (numeric features)")
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Raw Data")
    st.dataframe(df, use_container_width=True, height=350)

# ----------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# ----------------------------------------------------------------------------
with tab_model:
    m = bundle["metrics"]
    st.subheader(f"{algorithm} Performance")
    st.caption(
        f"Trained on 80/20 split using {len(bundle['feature_names'])} feature(s), "
        f"max_depth={max_depth}"
        + (f", n_estimators={n_estimators}" if algorithm == "Random Forest" else "")
        + ". Adjust in the sidebar."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
    c2.metric("Precision", f"{m['precision']*100:.1f}%")
    c3.metric("Recall", f"{m['recall']*100:.1f}%")
    c4.metric("F1 Score", f"{m['f1']*100:.1f}%")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(bundle["y_test"], bundle["y_pred"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No", "Yes"], y=["No", "Yes"],
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fpr, tpr, _ = roc_curve(bundle["y_test"], bundle["y_prob"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={m['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random"))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        precision_c, recall_c, _ = precision_recall_curve(bundle["y_test"], bundle["y_prob"])
        pr_auc = auc(recall_c, precision_c)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall_c, y=precision_c, name=f"PR (AUC={pr_auc:.3f})"))
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fi = pd.Series(bundle["model"].feature_importances_, index=bundle["feature_names"])
        fi = fi.sort_values(ascending=False).head(15)
        fig = px.bar(
            fi[::-1], orientation="h",
            title="Top Feature Importances",
            labels={"value": "Importance", "index": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4: PREDICT (form built dynamically from the currently selected features)
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict Attrition Risk for an Employee")
    st.caption(
        f"Form below reflects the {len(bundle['feature_names'])} feature(s) currently "
        "selected in the sidebar, used by the trained model."
    )

    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_names = bundle["feature_names"]

    with st.form("predict_form"):
        cols = st.columns(3)
        raw_input = {}

        for i, col in enumerate(feature_names):
            target_col = cols[i % 3]
            with target_col:
                if col in encoders:  # categorical
                    options = sorted(df_raw[col].unique().tolist())
                    default_idx = 0
                    raw_input[col] = st.selectbox(col, options, index=default_idx, key=f"in_{col}")
                else:  # numeric
                    col_min = int(df_raw[col].min())
                    col_max = int(df_raw[col].max())
                    col_default = int(df_raw[col].median())
                    if col_min == col_max:
                        col_max = col_min + 1
                    raw_input[col] = st.slider(col, col_min, col_max, col_default, key=f"in_{col}")

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([raw_input])
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        input_df = input_df[feature_names]
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        res1, res2 = st.columns([1, 2])
        with res1:
            if pred == 1:
                st.error(f"⚠️ Likely to Leave\n\nAttrition Probability: **{prob*100:.1f}%**")
            else:
                st.success(f"✅ Likely to Stay\n\nAttrition Probability: **{prob*100:.1f}%**")
        with res2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Attrition Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#E63946" if pred == 1 else "#2E86AB"},
                    "steps": [
                        {"range": [0, 33], "color": "#d4edda"},
                        {"range": [33, 66], "color": "#fff3cd"},
                        {"range": [66, 100], "color": "#f8d7da"},
                    ],
                },
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
