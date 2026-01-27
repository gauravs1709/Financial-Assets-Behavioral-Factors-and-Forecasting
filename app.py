import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

# ----------------------------------
# Streamlit page setup
# ----------------------------------
st.set_page_config(
    page_title="Sprint 3 – Behavioural Finance Dashboard",
    layout="wide"
)

st.title("Sprint 3 – Behavioural Finance Model Dashboard")
st.write(
    "This dashboard visualises short-term market behaviour using "
    "behavioural finance indicators and compares Linear Regression vs Random Forest."
)

# ----------------------------------
# Helper functions
# ----------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=["date"])
    df = df.set_index("date")
    return df

def compute_models(df, feature_cols, target_col="close", test_size=0.2):
    # Drop rows with missing feature/target
    df = df.dropna(subset=feature_cols + [target_col]).copy()

    X = df[feature_cols]
    y = df[target_col]

    # Time-based split (no shuffle)
    split = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train models
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Metrics
    def metrics(true, pred, name):
        mae = mean_absolute_error(true, pred)
        rmse = mean_squared_error(true, pred) ** 0.5
        r2 = r2_score(true, pred)
        return {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

    lr_metrics = metrics(y_test, lr_pred, "Linear Regression")
    rf_metrics = metrics(y_test, rf_pred, "Random Forest")

    # Build a DataFrame for plotting (test period only)
    results = pd.DataFrame({
        "Actual": y_test,
        "LR_Pred": lr_pred,
        "RF_Pred": rf_pred
    }, index=y_test.index)

    return lr_metrics, rf_metrics, results, df


# ----------------------------------
# Sidebar – data upload & options
# ----------------------------------
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload cleaned 60-day CSV (optional)", type=["csv"],
    help="If not uploaded, the app will use sp500_60d_clean.csv from the project folder."
)

test_size = st.sidebar.slider(
    "Test size (fraction of data for testing)",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

# Default features based on your Sprint-2 work
default_features = ["ret1", "ema_10", "z_close_20", "vix", "vix_z_20"]

# ----------------------------------
# Load data
# ----------------------------------
if uploaded_file is not None:
    st.success("Using uploaded dataset.")
    df = load_data(uploaded_file)
else:
    st.info("No file uploaded. Using default 'sp500_60d_clean.csv'.")
    df = load_data("sp500_60d_clean.csv")

# Check which of the default features actually exist
available_features = [c for c in default_features if c in df.columns]

if not available_features:
    st.error(
        "None of the expected behavioural features were found in the dataset. "
        "Expected some of: " + ", ".join(default_features)
    )
    st.stop()

st.sidebar.write("Behavioural / technical features used:")
for f in available_features:
    st.sidebar.write(f"- {f}")

# ----------------------------------
# Main – Data overview
# ----------------------------------
st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("**Sample of data (last 5 rows):**")
    st.dataframe(df.tail())

with col2:
    st.write("**Summary statistics (key variables):**")
    show_cols = list(set(available_features + ["close"]))
    st.dataframe(df[show_cols].describe().T)

st.markdown("---")

# ----------------------------------
# Model training & results
# ----------------------------------
st.subheader("Model Training & Evaluation")

lr_metrics, rf_metrics, results_df, df_full = compute_models(
    df, available_features, target_col="close", test_size=test_size
)

metrics_df = pd.DataFrame([lr_metrics, rf_metrics])
st.write("**Model performance (test set):**")
st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}))

st.markdown(
    "*Linear Regression generally performs better on this short-term 60-day dataset, "
    "while Random Forest struggles due to limited data and mostly linear relationships.*"
)

# ----------------------------------
# Plot – Actual vs Predictions (test period)
# ----------------------------------
st.subheader("Actual vs Predicted Prices (Test Period)")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results_df.index, y=results_df["Actual"],
    mode="lines+markers", name="Actual"
))
fig.add_trace(go.Scatter(
    x=results_df.index, y=results_df["LR_Pred"],
    mode="lines+markers", name="Linear Regression"
))
fig.add_trace(go.Scatter(
    x=results_df.index, y=results_df["RF_Pred"],
    mode="lines+markers", name="Random Forest"
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="S&P 500 Close Price",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "Hover over the chart to see the exact values for Actual, Linear Regression, "
    "and Random Forest for each day in the test period."
)

# ----------------------------------
# Behavioural indicators view
# ----------------------------------
st.subheader("Behavioural Finance Indicators")

behav_cols = [c for c in ["vix", "vix_z_20", "z_close_20", "ema_10", "ret1"] if c in df.columns]
if behav_cols:
    st.write("**Behavioural and technical indicators over time:**")
    fig2 = go.Figure()
    for c in behav_cols:
        fig2.add_trace(go.Scatter(
            x=df.index, y=df[c],
            mode="lines", name=c
        ))
    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Indicator value (normalised where applicable)",
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No behavioural indicator columns found to plot.")

st.markdown("---")
st.caption("Sprint 3 – Behavioural Finance Dashboard (S&P 500, 60-day window)")
