import pickle
from datetime import datetime
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
from fpdf import FPDF

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

# The columns the bundled IBM HR dataset ships with (minus drop cols/target).
# Used only as a reference to tell the user which "standard" features are
# missing when they upload their own file — everything still works without them.
EXPECTED_FEATURES = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


def has_col(df, col):
    return col in df.columns


def impute_missing(work: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill missing values (numeric -> median, categorical -> mode) and report what was filled."""
    report = {}
    for col in work.columns:
        n_missing = int(work[col].isna().sum())
        if n_missing > 0:
            if work[col].dtype == object:
                fill_val = work[col].mode(dropna=True)
                fill_val = fill_val.iloc[0] if len(fill_val) else "Unknown"
            else:
                fill_val = work[col].median()
            work[col] = work[col].fillna(fill_val)
            report[col] = n_missing
    return work, report


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)


# ----------------------------------------------------------------------------
# MODEL TRAINING (cached)
# ----------------------------------------------------------------------------
def _load_pretrained(df: pd.DataFrame, selected_features, algorithm, max_depth):
    """Use artifacts saved by train.py, but only when the sidebar selections match
    exactly what train.py trained on (full default feature set, Decision Tree, depth 5,
    default bundled dataset)."""
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
    try:
        for col, le in encoders.items():
            work[col] = le.transform(work[col])
    except Exception:
        return None
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
        "metrics": metrics, "source": "pretrained", "missing_report": {},
    }


@st.cache_resource
def train_model(df: pd.DataFrame, selected_features: tuple, algorithm: str,
                 max_depth: int = 5, n_estimators: int = 200):
    selected_features = list(selected_features)

    work = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    work, missing_report = impute_missing(work)

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
        "missing_report": missing_report,
    }


@st.cache_data
def compute_feature_ranking(df: pd.DataFrame, feature_pool: tuple):
    """Quick baseline Decision Tree over every available feature, used only to
    rank features by importance for the sidebar's feature-selection UI."""
    work = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    work, _ = impute_missing(work)
    for col in work.select_dtypes(include="object").columns:
        work[col] = LabelEncoder().fit_transform(work[col])
    X = work[list(feature_pool)]
    y = work["Attrition"]
    baseline = DecisionTreeClassifier(random_state=42, max_depth=6)
    baseline.fit(X, y)
    importances = pd.Series(baseline.feature_importances_, index=feature_pool)
    return importances.sort_values(ascending=False)


# ----------------------------------------------------------------------------
# PDF REPORT
# ----------------------------------------------------------------------------
def _pdf_safe(text: str) -> str:
    """fpdf2's default core font (Helvetica) only supports latin-1. Replace common
    unicode punctuation and drop anything else it can't encode, so arbitrary
    column names / uploaded file names never crash report generation."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(data_source, df, algorithm, max_depth, n_estimators,
                         missing_expected, missing_report, bundle, ranked_importance):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Employee Attrition Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    def section(title):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _pdf_safe(title), ln=True)
        pdf.set_font("Helvetica", "", 10)

    section("Dataset Summary")
    pdf.cell(0, 6, _pdf_safe(f"Source: {data_source}"), ln=True)
    pdf.cell(0, 6, f"Rows: {len(df)}    Columns: {df.shape[1]}", ln=True)
    if "Attrition" in df.columns:
        attr_rate = (df["Attrition"] == "Yes").mean() * 100
        pdf.cell(0, 6, f"Attrition rate: {attr_rate:.1f}%", ln=True)

    if missing_expected:
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, _pdf_safe("Standard features not present in this dataset (skipped): "
                              + ", ".join(missing_expected)))
        pdf.set_font("Helvetica", "", 10)

    if missing_report:
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Missing values filled (median/mode imputation):", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for col, cnt in missing_report.items():
            pct = cnt / len(df) * 100
            pdf.cell(0, 5, _pdf_safe(f"  - {col}: {cnt} missing ({pct:.1f}%)"), ln=True)

    section("Model Configuration")
    pdf.cell(0, 6, f"Algorithm: {algorithm}", ln=True)
    pdf.cell(0, 6, f"Max depth: {max_depth}", ln=True)
    if algorithm == "Random Forest":
        pdf.cell(0, 6, f"Number of trees: {n_estimators}", ln=True)
    pdf.cell(0, 6, f"Features used: {len(bundle['feature_names'])}", ln=True)

    section("Performance Metrics (on held-out test split)")
    m = bundle["metrics"]
    pdf.cell(0, 6, f"Accuracy:  {m['accuracy']*100:.2f}%", ln=True)
    pdf.cell(0, 6, f"Precision: {m['precision']*100:.2f}%", ln=True)
    pdf.cell(0, 6, f"Recall:    {m['recall']*100:.2f}%", ln=True)
    pdf.cell(0, 6, f"F1 Score:  {m['f1']*100:.2f}%", ln=True)
    pdf.cell(0, 6, f"ROC-AUC:   {m['roc_auc']:.3f}", ln=True)

    cm = confusion_matrix(bundle["y_test"], bundle["y_pred"])
    section("Confusion Matrix")
    pdf.cell(0, 6, f"True Negatives:  {cm[0][0]}    False Positives: {cm[0][1]}", ln=True)
    pdf.cell(0, 6, f"False Negatives: {cm[1][0]}    True Positives:  {cm[1][1]}", ln=True)

    section("Feature Importance Ranking (all evaluated features)")
    total = ranked_importance.sum() or 1
    for rank, (feat, imp) in enumerate(ranked_importance.items(), start=1):
        used = "*" if feat in bundle["feature_names"] else " "
        line = f"{rank:>2}. [{used}] {feat} - {imp/total*100:.1f}%"
        pdf.cell(0, 5, _pdf_safe(line), ln=True)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "[*] = included in the currently trained model", ln=True)

    return bytes(pdf.output())


# ----------------------------------------------------------------------------
# SIDEBAR: dataset
# ----------------------------------------------------------------------------
st.sidebar.title("📁 Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload your own CSV (optional)", type=["csv"],
    help="Must include an 'Attrition' column (Yes/No target). Any of the standard "
         "columns can be missing — the app will just work with what's available.",
)

missing_expected = []
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        if "Attrition" not in user_df.columns:
            st.sidebar.error("Uploaded file has no 'Attrition' column — falling back to the bundled dataset.")
            df_raw = load_default_data()
            data_source = "bundled dataset (upload rejected: no 'Attrition' column)"
        else:
            df_raw = user_df
            data_source = f"uploaded file: {uploaded_file.name}"
            missing_expected = [c for c in EXPECTED_FEATURES if c not in df_raw.columns]
            if missing_expected:
                st.sidebar.warning(
                    f"{len(missing_expected)} standard feature(s) not in this file — "
                    "they're simply excluded from selection:\n\n"
                    + ", ".join(missing_expected)
                )
            extra_cols = [c for c in df_raw.columns if c not in EXPECTED_FEATURES + ["Attrition"] + DROP_COLS]
            if extra_cols:
                st.sidebar.success(f"{len(extra_cols)} extra column(s) found and available: " + ", ".join(extra_cols))
    except Exception as e:
        st.sidebar.error(f"Couldn't read that file ({e}) — using the bundled dataset instead.")
        df_raw = load_default_data()
        data_source = "bundled dataset (upload failed to parse)"
else:
    df_raw = load_default_data()
    data_source = "bundled dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv)"

ALL_FEATURES = [c for c in df_raw.columns if c not in DROP_COLS + ["Attrition"]]
if len(ALL_FEATURES) < 2:
    st.error("This dataset doesn't have enough usable feature columns to train on.")
    st.stop()

total_missing_cells = int(df_raw[ALL_FEATURES + ["Attrition"]].isna().sum().sum())
if total_missing_cells > 0:
    st.sidebar.info(f"{total_missing_cells} missing cell(s) detected — will be auto-imputed "
                     "(median for numeric, mode for categorical) before training.")

# ----------------------------------------------------------------------------
# SIDEBAR: model configuration
# ----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Model Configuration")

algorithm = st.sidebar.selectbox("Algorithm", ["Decision Tree", "Random Forest"])

if algorithm == "Random Forest":
    n_estimators = st.sidebar.slider("Number of trees (n_estimators)", 50, 500, 200, step=50)
else:
    n_estimators = 200  # unused for Decision Tree, kept for a stable cache key

max_depth = st.sidebar.slider("Max tree depth", 2, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Feature Selection (ranked by importance)")

ranked_importance = compute_feature_ranking(df_raw, tuple(ALL_FEATURES))
ranked_features = ranked_importance.index.tolist()
_total_imp = ranked_importance.sum() or 1
feature_labels = {
    f: f"{i+1}. {f} ({ranked_importance[f]/_total_imp*100:.1f}%)"
    for i, f in enumerate(ranked_features)
}
label_to_feature = {v: k for k, v in feature_labels.items()}

select_all = st.sidebar.checkbox("Select all features", value=True)
if select_all:
    selected_features = ranked_features
else:
    default_labels = [feature_labels[f] for f in ranked_features[:8]]
    chosen_labels = st.sidebar.multiselect(
        "Choose features to train on (most important first)",
        options=[feature_labels[f] for f in ranked_features],
        default=default_labels,
    )
    selected_features = [label_to_feature[l] for l in chosen_labels]

with st.sidebar.expander("View full importance ranking"):
    st.dataframe(
        (ranked_importance / _total_imp * 100).round(1).rename("Importance %").reset_index().rename(columns={"index": "Feature"}),
        use_container_width=True, height=300,
    )

if len(selected_features) < 2:
    st.sidebar.error("Select at least 2 features to train a model.")
    st.stop()

pretrained = _load_pretrained(df_raw, selected_features, algorithm, max_depth) if uploaded_file is None else None
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
    "Department", options=sorted(df_raw["Department"].unique()) if has_col(df_raw, "Department") else [],
    default=None, disabled=not has_col(df_raw, "Department"),
)
gender_filter = st.sidebar.multiselect(
    "Gender", options=sorted(df_raw["Gender"].unique()) if has_col(df_raw, "Gender") else [],
    default=None, disabled=not has_col(df_raw, "Gender"),
)

df = df_raw.copy()
if dept_filter:
    df = df[df["Department"].isin(dept_filter)]
if gender_filter:
    df = df[df["Gender"].isin(gender_filter)]

st.sidebar.markdown("---")
st.sidebar.caption(data_source)

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

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Employees", f"{total_emp:,}")
    kpi_cols[1].metric("Attrition Rate", f"{attr_rate:.1f}%")
    if has_col(df, "Age"):
        kpi_cols[2].metric("Avg. Age", f"{df['Age'].mean():.1f} yrs")
    if has_col(df, "MonthlyIncome"):
        kpi_cols[3].metric("Avg. Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")
    if has_col(df, "YearsAtCompany"):
        kpi_cols[4].metric("Avg. Tenure", f"{df['YearsAtCompany'].mean():.1f} yrs")

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
        if has_col(df, "Department"):
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
        else:
            st.info("No 'Department' column in this dataset.")

    col3, col4 = st.columns(2)
    with col3:
        if has_col(df, "OverTime"):
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
        else:
            st.info("No 'OverTime' column in this dataset.")

    with col4:
        if has_col(df, "Age"):
            fig = px.histogram(
                df, x="Age", color="Attrition", barmode="overlay", nbins=25,
                title="Age Distribution by Attrition",
                color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'Age' column in this dataset.")

    if total_missing_cells > 0:
        st.markdown("---")
        st.markdown("#### Data Quality — Missing Values")
        miss_counts = df_raw[ALL_FEATURES + ["Attrition"]].isna().sum()
        miss_counts = miss_counts[miss_counts > 0].sort_values(ascending=False)
        fig = px.bar(miss_counts, title="Missing Values by Column", labels={"value": "Missing count", "index": "Column"})
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: EXPLORE DATA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Interactive Exploration")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["EmployeeCount", "EmployeeNumber", "StandardHours"]]
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Attrition"]

    if numeric_cols and cat_cols:
        colx, coly = st.columns(2)
        with colx:
            x_axis = st.selectbox("X-axis (numeric)", numeric_cols, index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0)
        with coly:
            split_by = st.selectbox("Split by (category)", cat_cols, index=cat_cols.index("JobRole") if "JobRole" in cat_cols else 0)

        fig = px.box(
            df, x=split_by, y=x_axis, color="Attrition",
            title=f"{x_axis} by {split_by} and Attrition",
            color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least one numeric and one categorical column for this chart.")

    st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        if has_col(df, "JobSatisfaction"):
            fig = px.histogram(
                df, x="JobSatisfaction", color="Attrition", barmode="group",
                title="Job Satisfaction vs Attrition",
                color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'JobSatisfaction' column in this dataset.")
    with col6:
        if has_col(df, "WorkLifeBalance"):
            fig = px.histogram(
                df, x="WorkLifeBalance", color="Attrition", barmode="group",
                title="Work-Life Balance vs Attrition",
                color_discrete_map={"No": "#2E86AB", "Yes": "#E63946"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'WorkLifeBalance' column in this dataset.")

    if len(numeric_cols) >= 2:
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
            title="Top Feature Importances (current model)",
            labels={"value": "Importance", "index": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if bundle.get("missing_report"):
        st.markdown("---")
        st.markdown("#### Missing Values Imputed Before Training")
        miss_df = pd.DataFrame(
            [(c, n, f"{n/len(df_raw)*100:.1f}%") for c, n in bundle["missing_report"].items()],
            columns=["Column", "Missing Count", "% of Rows"],
        )
        st.dataframe(miss_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📄 Download Report")
    st.caption("A PDF summary of the dataset, model configuration, performance metrics, and full feature importance ranking.")
    pdf_bytes = generate_pdf_report(
        data_source, df_raw, algorithm, max_depth, n_estimators,
        missing_expected, bundle.get("missing_report", {}), bundle, ranked_importance,
    )
    st.download_button(
        "📥 Download PDF Report", data=pdf_bytes,
        file_name=f"attrition_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf", use_container_width=True,
    )

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
                    options = sorted(df_raw[col].dropna().unique().tolist())
                    raw_input[col] = st.selectbox(col, options, index=0, key=f"in_{col}")
                else:  # numeric
                    series = df_raw[col].dropna()
                    col_min = int(series.min())
                    col_max = int(series.max())
                    col_default = int(series.median())
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
