import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1f4e79;}
    .metric-card {background: #f0f4ff; border-radius: 10px; padding: 1rem; text-align: center;}
    .risk-high {color: #c0392b; font-weight: 700; font-size: 1.4rem;}
    .risk-low  {color: #27ae60; font-weight: 700; font-size: 1.4rem;}
    .section-title {font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ─── Load & Prepare Data ────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded=None):
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        # Try default path
        try:
            df = pd.read_csv("Credit Risk Data.csv")
        except FileNotFoundError:
            return None
    return df.dropna()


@st.cache_resource
def train_model(df, model_type="Random Forest"):
    df_enc = pd.get_dummies(df, drop_first=True)
    X = df_enc.drop("loan_status", axis=1)
    y = df_enc["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if model_type == "Random Forest":
        model = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=100)
    else:
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, X_train, X_test, y_train, y_test, y_pred, y_prob, X.columns.tolist()


# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-front-side.png", width=80)
st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])

st.sidebar.markdown("---")
nav = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA", "🤖 Model Performance", "🔮 Predict Loan Risk"],
)

# ─── Load Data ──────────────────────────────────────────────────────────────
df = load_data(uploaded_file)

if df is None:
    st.markdown('<p class="main-header">💳 Credit Risk Predictor</p>', unsafe_allow_html=True)
    st.warning("⚠️ No dataset found. Please upload `Credit_Risk_Data.csv` using the sidebar.")
    st.stop()

model, X_train, X_test, y_train, y_test, y_pred, y_prob, feature_cols = train_model(df, model_choice)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Overview":
    st.markdown('<p class="main-header">💳 Credit Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown("Explore the dataset, evaluate model performance, and predict individual loan risk.")

    col1, col2, col3, col4 = st.columns(4)
    default_rate = df["loan_status"].mean() * 100
    col1.metric("📋 Total Records", f"{len(df):,}")
    col2.metric("📌 Features", len(df.columns) - 1)
    col3.metric("⚠️ Default Rate", f"{default_rate:.1f}%")
    col4.metric("✅ Model Accuracy", f"{accuracy_score(y_test, y_pred)*100:.1f}%")

    st.markdown("---")
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
elif nav == "📊 EDA":
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Loan status distribution
    with col1:
        st.subheader("Loan Status Distribution")
        counts = df["loan_status"].value_counts().rename({0: "Non-Default", 1: "Default"})
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=["#27ae60", "#c0392b"], startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Default vs Non-Default")
        st.pyplot(fig)
        plt.close()

    # Loan intent breakdown
    with col2:
        st.subheader("Loan Intent Breakdown")
        fig, ax = plt.subplots(figsize=(5, 4))
        intent_counts = df["loan_intent"].value_counts()
        intent_counts.plot(kind="barh", ax=ax, color="#4a90d9")
        ax.set_xlabel("Count")
        ax.set_title("Loan Intent")
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    # Age distribution
    with col3:
        st.subheader("Age Distribution by Loan Status")
        fig, ax = plt.subplots(figsize=(5, 4))
        for status, label, color in [(0, "Non-Default", "#27ae60"), (1, "Default", "#c0392b")]:
            df[df["loan_status"] == status]["person_age"].hist(
                bins=25, ax=ax, alpha=0.6, label=label, color=color
            )
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Age by Loan Status")
        st.pyplot(fig)
        plt.close()

    # Interest rate vs loan amount
    with col4:
        st.subheader("Interest Rate vs Loan Amount")
        fig, ax = plt.subplots(figsize=(5, 4))
        sample = df.sample(min(1000, len(df)), random_state=42)
        scatter = ax.scatter(
            sample["loan_amnt"], sample["loan_int_rate"],
            c=sample["loan_status"], cmap="RdYlGn_r",
            alpha=0.6, s=15
        )
        plt.colorbar(scatter, ax=ax, label="Default (1=Yes)")
        ax.set_xlabel("Loan Amount ($)")
        ax.set_ylabel("Interest Rate (%)")
        ax.set_title("Loan Amount vs Interest Rate")
        st.pyplot(fig)
        plt.close()

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, square=True)
    st.pyplot(fig)
    plt.close()

    # Home ownership
    st.subheader("Default Rate by Home Ownership")
    ho = df.groupby("person_home_ownership")["loan_status"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3))
    ho.plot(kind="bar", ax=ax, color=["#c0392b", "#e67e22", "#f1c40f", "#27ae60"])
    ax.set_ylabel("Default Rate")
    ax.set_title("Default Rate by Home Ownership")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif nav == "🤖 Model Performance":
    st.markdown(f'<p class="main-header">🤖 {model_choice} — Model Performance</p>',
                unsafe_allow_html=True)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Non-Default", "Default"],
                                   output_dict=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("ROC-AUC", f"{auc:.4f}")
    col3.metric("Precision (Default)", f"{report['Default']['precision']:.2f}")
    col4.metric("Recall (Default)", f"{report['Default']['recall']:.2f}")

    col_a, col_b = st.columns(2)

    # Confusion Matrix
    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Non-Default", "Default"],
                    yticklabels=["Non-Default", "Default"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close()

    # ROC Curve
    with col_b:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#4a90d9", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # Feature Importance (Random Forest only)
    if model_choice == "Random Forest":
        st.subheader("Feature Importances")
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        importances.tail(15).plot(kind="barh", ax=ax, color="#4a90d9")
        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        plt.close()

    # Classification Report Table
    st.subheader("Classification Report")
    rep_df = pd.DataFrame(report).T.round(3)
    st.dataframe(rep_df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
elif nav == "🔮 Predict Loan Risk":
    st.markdown('<p class="main-header">🔮 Predict Loan Risk</p>', unsafe_allow_html=True)
    st.markdown("Fill in the applicant details below to get an instant credit risk prediction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Personal Info**")
            person_age = st.number_input("Age", 18, 100, 30)
            person_income = st.number_input("Annual Income ($)", 5000, 1_000_000, 55000, step=1000)
            person_home_ownership = st.selectbox(
                "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
            )
            person_emp_length = st.number_input("Employment Length (years)", 0, 60, 5)

        with col2:
            st.markdown("**💼 Loan Details**")
            loan_intent = st.selectbox(
                "Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
            )
            loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
            loan_amnt = st.number_input("Loan Amount ($)", 500, 100_000, 10000, step=500)
            loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.1)

        with col3:
            st.markdown("**📋 Credit Info**")
            loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.20, step=0.01)
            cb_default = st.selectbox("Previous Default on File", ["N", "Y"])
            cb_cred_hist = st.number_input("Credit History Length (years)", 1, 40, 5)

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        # Build input row
        input_dict = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_default,
            "cb_person_cred_hist_length": cb_cred_hist,
        }
        input_df = pd.DataFrame([input_dict])

        # Align with training encoding
        full_df = pd.get_dummies(df.drop("loan_status", axis=1), drop_first=True)
        input_enc = pd.get_dummies(input_df, drop_first=True)
        input_enc = input_enc.reindex(columns=full_df.columns, fill_value=0)

        prob = model.predict_proba(input_enc)[0][1]
        pred = model.predict(input_enc)[0]

        st.markdown("---")
        st.subheader("📋 Prediction Result")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Default Probability", f"{prob*100:.1f}%")
        col_r2.metric("Risk Level", "HIGH RISK 🔴" if pred == 1 else "LOW RISK 🟢")
        col_r3.metric("Recommendation", "⛔ Decline" if pred == 1 else "✅ Approve")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 2.5))
        bar_color = "#c0392b" if prob > 0.5 else "#27ae60"
        ax.barh(["Risk Score"], [prob], color=bar_color, height=0.4)
        ax.barh(["Risk Score"], [1 - prob], left=prob, color="#ecf0f1", height=0.4)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Default Probability: {prob*100:.1f}%")
        ax.set_xlabel("Probability")
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)
        plt.close()

        if pred == 1:
            st.error("⚠️ **High Credit Risk** — This applicant is likely to default based on the model. Consider reviewing loan terms or declining.")
        else:
            st.success("✅ **Low Credit Risk** — This applicant is unlikely to default. Loan can be considered for approval.")
