import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------
#  Utility functions
# ---------------------------------
@st.cache_data
def load_data(uploaded_file=None, default_path="sp500_60d_clean.csv"):
    """Load CSV either from upload or from default file."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
    else:
        df = pd.read_csv(default_path, parse_dates=["date"])

    if "date" in df.columns:
        df = df.set_index("date")

    return df


def prepare_features(df):
    """Select required features and drop missing values."""
    features = ["ret1", "ema_10", "z_close_20", "vix", "vix_z_20"]
    required_cols = features + ["close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        return None, None, None

    df_clean = df.dropna(subset=required_cols).copy()
    X = df_clean[features]
    y = df_clean["close"]
    return df_clean, X, y


def time_split(X, y, train_ratio=0.8):
    """Simple time-based train/test split (no shuffling)."""
    n = len(X)
    split_idx = int(n * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ---------------------------------
#  Streamlit app
# ---------------------------------
def main():
    st.set_page_config(
        page_title="Sprint-3 Behavioural Finance Dashboard",
        layout="wide"
    )

    st.title("Sprint-3: Behavioural Finance Model Dashboard")
    st.write(
        "This dashboard loads S&P500-style data, trains **Linear Regression** "
        "and **Random Forest** models, and visualises predictions together "
        "with behavioural & technical indicators."
    )

    # Sidebar: data upload and options
    st.sidebar.header("Data & Settings")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (optional). If empty, uses 'sp500_60d_clean.csv' in the app folder.",
        type=["csv"],
    )

    train_ratio = st.sidebar.slider(
        "Train ratio (time-based split)",
        min_value=0.6,
        max_value=0.9,
        value=0.8,
        step=0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Columns expected in the dataset:")
    st.sidebar.write("`date, close, ret1, ema_10, z_close_20, vix, vix_z_20`")

    # Load data
    df = load_data(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Prepare features
    df_clean, X, y = prepare_features(df)
    if df_clean is None:
        st.stop()  # cannot continue without required columns

    st.subheader("Summary Statistics (Key Variables)")
    st.dataframe(df_clean[["ret1", "ema_10", "z_close_20", "vix", "vix_z_20", "close"]].describe())

    # Train / test split
    X_train, X_test, y_train, y_test = time_split(X, y, train_ratio=train_ratio)

    # ---------------------------------
    #  Models: Linear Regression & RF
    # ---------------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Metrics
    lr_mae, lr_rmse, lr_r2 = compute_metrics(y_test, lr_pred)
    rf_mae, rf_rmse, rf_r2 = compute_metrics(y_test, rf_pred)

    st.subheader("Model Performance (Test Period)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear Regression**")
        st.metric("MAE", f"{lr_mae:,.2f}")
        st.metric("RMSE", f"{lr_rmse:,.2f}")
        st.metric("R²", f"{lr_r2:.3f}")

    with col2:
        st.markdown("**Random Forest**")
        st.metric("MAE", f"{rf_mae:,.2f}")
        st.metric("RMSE", f"{rf_rmse:,.2f}")
        st.metric("R²", f"{rf_r2:.3f}")

    # ---------------------------------
    #  Figure 1: Actual vs Predicted
    # ---------------------------------
    st.subheader("Actual vs Predicted S&P 500 Close (Test Period)")

    results = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Linear Regression": lr_pred,
            "Random Forest": rf_pred,
        },
        index=y_test.index,
    )

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(results.index, results["Actual"], label="Actual", marker="o")
    ax1.plot(results.index, results["Linear Regression"], label="Linear Regression", marker="x")
    ax1.plot(results.index, results["Random Forest"], label="Random Forest", marker="s")
    ax1.set_title("Actual vs Predicted S&P 500 Close (Test Period)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)

    # ---------------------------------
    #  Figure 2: Normalized Behavioural Indicators
    # ---------------------------------
    st.subheader("Normalized Behavioural & Technical Indicators (0–1 Scale)")

    behav_cols = [c for c in ["vix", "vix_z_20", "z_close_20", "ema_10", "ret1"] if c in df_clean.columns]

    if behav_cols:
        df_norm = df_clean[behav_cols].copy()
        for col in behav_cols:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        for col in behav_cols:
            ax2.plot(df_norm.index, df_norm[col], label=col)

        ax2.set_title("Normalized Behavioural & Technical Indicators (0–1 Scale)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Normalized Value")
        ax2.grid(True)
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("No behavioural indicator columns found to plot.")

    st.markdown("---")
    st.caption("Sprint-3 Behavioural Finance Dashboard – Linear Regression vs Random Forest")


if __name__ == "__main__":
    main()
