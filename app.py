import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from src.ml_utils import load_and_profile, preprocess, train_models, get_feature_importance

# Page config
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Dashboard")
st.markdown("Upload a CSV dataset, explore it, and train ML models — all in one place.")

# ── 1. Upload ──────────────────────────────────────────────────────────────────
st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded — {df.shape[0]} rows, {df.shape[1]} columns")

    # ── 2. Data Profile ────────────────────────────────────────────────────────
    st.header("2. Data Profile")
    profile = load_and_profile(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", profile["rows"])
    col2.metric("Columns", profile["columns"])
    col3.metric("Missing Values", profile["missing"])

    with st.expander("Show raw data"):
        st.dataframe(df.head(50))

    with st.expander("Show column types"):
        st.json(profile["dtypes"])

    # ── 3. Visualizations ──────────────────────────────────────────────────────
    st.header("3. Explore Your Data")

    if profile["numeric_cols"]:
        st.subheader("Distributions")
        selected_col = st.selectbox("Select a column to plot", profile["numeric_cols"])
        fig = px.histogram(df, x=selected_col, nbins=30, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        fig2, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df[profile["numeric_cols"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax
        )
        st.pyplot(fig2)

    # ── 4. Train Models ────────────────────────────────────────────────────────
    st.header("4. Train ML Models")
    st.markdown("Select a **target column** to train classification models.")

    target_col = st.selectbox("Select target column", df.columns.tolist())

    if st.button("🚀 Train Models"):
        with st.spinner("Training in progress..."):
            try:
                X_train, X_test, y_train, y_test = preprocess(df, target_col)
                results = train_models(X_train, X_test, y_train, y_test)

                # ── Results ───────────────────────────────────────────────────
                st.subheader("Model Accuracy")
                cols = st.columns(3)
                for i, (name, result) in enumerate(results.items()):
                    cols[i].metric(name, f"{result['accuracy']}%")

                # ── Confusion Matrices ────────────────────────────────────────
                st.subheader("Confusion Matrices")
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for i, (name, result) in enumerate(results.items()):
                    sns.heatmap(
                        result["confusion_matrix"],
                        annot=True, fmt="d",
                        cmap="Blues", ax=axes[i]
                    )
                    axes[i].set_title(name)
                    axes[i].set_xlabel("Predicted")
                    axes[i].set_ylabel("Actual")
                st.pyplot(fig)

                # ── Feature Importance ────────────────────────────────────────
                st.subheader("Feature Importance")
                for name, result in results.items():
                    importance = get_feature_importance(
                        result["model"], X_train.columns
                    )
                    if importance is not None:
                        fig3 = px.bar(
                            importance.head(15),
                            orientation="h",
                            title=f"{name} — Top Features",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig3, use_container_width=True)

            except Exception as e:
                st.error(f"Error during training: {e}")

else:
    st.info("👆 Upload a CSV file to get started.")