import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============ PATHS & CONFIG ============

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide",
)


# ============ LOAD DATA & TRAIN MODEL (CACHED) ============

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Ensure correct types
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train churn prediction model and return pipeline + feature columns + metrics."""
    target = "ChurnLabel"
    y = df[target]

    feature_cols = [c for c in df.columns if c not in [target, "customerID"]]
    X = df[feature_cols]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return model, feature_cols, numeric_features, categorical_features, metrics


df = load_data()
model, feature_cols, num_feats, cat_feats, model_metrics = train_model(df)


# ============ SIDEBAR: THEME & FILTERS & NAV ============

st.sidebar.title("⚙️ Controls")

# Theme toggle (changes chart style)
theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
plot_template = "plotly_dark" if theme_choice == "Dark" else "plotly_white"

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "🔍 Navigate",
    ["🏠 Home", "📈 Insights", "🤖 Predict Churn", "ℹ️ About"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")

# Multi-select filters
gender_filter = st.sidebar.multiselect(
    "Gender", sorted(df["gender"].dropna().unique()), default=sorted(df["gender"].dropna().unique())
)

senior_filter = st.sidebar.multiselect(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    default=[0, 1],
)

contract_filter = st.sidebar.multiselect(
    "Contract Type", sorted(df["Contract"].dropna().unique()), default=sorted(df["Contract"].dropna().unique())
)

internet_filter = st.sidebar.multiselect(
    "Internet Service", sorted(df["InternetService"].dropna().unique()),
    default=sorted(df["InternetService"].dropna().unique())
)

payment_filter = st.sidebar.multiselect(
    "Payment Method", sorted(df["PaymentMethod"].dropna().unique()),
    default=sorted(df["PaymentMethod"].dropna().unique())
)

# Range filters
tenure_min, tenure_max = int(df["tenure"].min()), int(df["tenure"].max())
tenure_range = st.sidebar.slider("Tenure (months)", tenure_min, tenure_max, (tenure_min, tenure_max))

mc_min, mc_max = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())
monthly_range = st.sidebar.slider(
    "Monthly Charges",
    float(np.floor(mc_min)),
    float(np.ceil(mc_max)),
    (float(np.floor(mc_min)), float(np.ceil(mc_max))),
)


def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()

    if gender_filter:
        d = d[d["gender"].isin(gender_filter)]
    if senior_filter:
        d = d[d["SeniorCitizen"].isin(senior_filter)]
    if contract_filter:
        d = d[d["Contract"].isin(contract_filter)]
    if internet_filter:
        d = d[d["InternetService"].isin(internet_filter)]
    if payment_filter:
        d = d[d["PaymentMethod"].isin(payment_filter)]

    d = d[(d["tenure"] >= tenure_range[0]) & (d["tenure"] <= tenure_range[1])]
    d = d[(d["MonthlyCharges"] >= monthly_range[0]) & (d["MonthlyCharges"] <= monthly_range[1])]

    return d


filtered_df = apply_filters(df)


# ============ REUSABLE KPIs ============

def render_kpis(d: pd.DataFrame):
    total_customers = len(d)
    churn_rate = d["ChurnLabel"].mean() * 100 if len(d) > 0 else 0.0
    avg_monthly = d["MonthlyCharges"].mean() if len(d) > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", total_customers)
    c2.metric("Churn Rate (%)", f"{churn_rate:.2f}")
    c3.metric("Avg Monthly Charge", f"${avg_monthly:.2f}")
    c4.metric("Model F1 Score", f"{model_metrics['f1']:.2f}")


# ============ PAGE: HOME ============

if page == "🏠 Home":
    st.title("📊 Telco Customer Churn — Enterprise Analytics Dashboard")

    st.caption("Filters applied on left sidebar. All statistics & visuals below reflect the filtered data.")

    render_kpis(filtered_df)

    st.markdown("---")
    st.subheader("📌 Sample Customer Records")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    # Download button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name="filtered_telco_churn_customers.csv",
        mime="text/csv",
    )


# ============ PAGE: INSIGHTS ============

elif page == "📈 Insights":
    st.title("📈 Churn Insights")

    if filtered_df.empty:
        st.warning("No data to display. Please relax filters in the sidebar.")
    else:
        render_kpis(filtered_df)
        st.markdown("---")

        # Churn rate by contract
        st.subheader("🔹 Churn Rate by Contract Type")
        contract_stats = (
            filtered_df.groupby("Contract")["ChurnLabel"]
            .mean()
            .reset_index()
            .rename(columns={"ChurnLabel": "ChurnRate"})
        )
        fig1 = px.bar(
            contract_stats,
            x="Contract",
            y="ChurnRate",
            labels={"ChurnRate": "Churn Rate"},
            text=(contract_stats["ChurnRate"] * 100).round(1).astype(str) + "%",
            template=plot_template,
        )
        fig1.update_traces(textposition="outside")
        fig1.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)

        # Churn by tenure
        st.subheader("🔹 Churn by Tenure")
        fig2 = px.histogram(
            filtered_df,
            x="tenure",
            color="Churn",
            barmode="group",
            nbins=40,
            template=plot_template,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Monthly charges vs churn
        st.subheader("🔹 Monthly Charges vs Churn")
        fig3 = px.box(
            filtered_df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            template=plot_template,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ============ PAGE: PREDICT CHURN ============

elif page == "🤖 Predict Churn":
    st.title("🤖 Churn Prediction — What-if Analysis")

    st.write("Fill in the customer details below and click **Predict** to see whether the customer is likely to churn.")

    # Use a base row to ensure all columns exist
    base = df.iloc[[0]].copy()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", sorted(df["Partner"].dropna().unique()))

        with col2:
            dependents = st.selectbox("Dependents", sorted(df["Dependents"].dropna().unique()))
            tenure_val = st.slider("Tenure (months)", int(df["tenure"].min()), int(df["tenure"].max()), 12)
            contract = st.selectbox("Contract", sorted(df["Contract"].dropna().unique()))

        with col3:
            internet = st.selectbox("Internet Service", sorted(df["InternetService"].dropna().unique()))
            payment = st.selectbox("Payment Method", sorted(df["PaymentMethod"].dropna().unique()))
            monthly_val = st.number_input(
                "Monthly Charges",
                float(df["MonthlyCharges"].min()),
                float(df["MonthlyCharges"].max()),
                float(df["MonthlyCharges"].mean()),
            )

        total_val = st.number_input(
            "Total Charges",
            float(df["TotalCharges"].min()),
            float(df["TotalCharges"].max()),
            float(df["TotalCharges"].mean()),
        )

        submitted = st.form_submit_button("🔍 Predict Churn")

    if submitted:
        # Update base row with user choices
        base["gender"] = gender
        base["SeniorCitizen"] = senior
        base["Partner"] = partner
        base["Dependents"] = dependents
        base["tenure"] = tenure_val
        base["Contract"] = contract
        base["InternetService"] = internet
        base["PaymentMethod"] = payment
        base["MonthlyCharges"] = monthly_val
        base["TotalCharges"] = total_val

        X_input = base[feature_cols]

        pred_proba = model.predict_proba(X_input)[0][1]
        pred_label = model.predict(X_input)[0]

        st.markdown("---")
        col_left, col_right = st.columns([2, 1])

        with col_left:
            if pred_label == 1:
                st.error(f"⚠ The customer is **LIKELY TO CHURN**.\n\nEstimated probability: **{pred_proba:.2%}**")
            else:
                st.success(f"✅ The customer is **NOT LIKELY TO CHURN**.\n\nEstimated probability: **{pred_proba:.2%}**")

        with col_right:
            st.metric("Churn Probability", f"{pred_proba:.2%}")
            st.metric("Model Accuracy", f"{model_metrics['accuracy']:.2f}")
            st.metric("Model F1 Score", f"{model_metrics['f1']:.2f}")


# ============ PAGE: ABOUT ============

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.write(
        """
        This is an **enterprise-style Customer Churn Analytics & Prediction system** built using:

        - **Python, Pandas, NumPy**
        - **Scikit-learn** for machine learning
        - **Streamlit** for the interactive web dashboard
        - **Plotly** for interactive visualizations
        - A real-world **Telco Customer Churn** dataset

        ### Key Capabilities
        - Global filters to slice & dice customer segments
        - KPIs: total customers, churn rate, average revenue
        - Interactive insights on churn vs contract, tenure, and charges
        - Machine learning model (Random Forest) to predict churn probability
        - What-if analysis via simulated customer profiles
        - One-click download of filtered customer data

        This project demonstrates an end-to-end data pipeline from **cleaned dataset → ML model → business dashboard**,
        similar to what is deployed in real analytics teams in industry.
        """
    )

    st.markdown("---")
    st.subheader("📊 Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📈 Feature Summary (Numeric)")
    st.write(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())
