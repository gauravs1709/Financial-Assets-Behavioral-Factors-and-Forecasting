"""
sprint4_final_models.py

Uses clean merged dataset:
    sp500_60d_with_sentiment_clean.csv

Models:
    - Linear Regression (behaviour only)
    - Random Forest (behaviour only)
    - Linear Regression (behaviour + sentiment)
    - Random Forest (behaviour + sentiment)

Outputs:
    - Metrics (MAE, RMSE, R2)
    - 5 charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================
# 1. LOAD CLEAN DATASET
# ======================================================

df = pd.read_csv("sp500_60d_with_sentiment_clean.csv", parse_dates=["date"])

# Sort by date to be safe
df = df.sort_values("date").reset_index(drop=True)

print("Columns in dataset:")
print(df.columns)

target_col = "close"

# Behavioural / technical features
behaviour_features = ["ret1", "ema_10", "z_close_20", "vix", "vix_z_20"]

# Sentiment features
sentiment_features = [
    "finbert_pos",
    "finbert_neg",
    "tb_polarity",
    "tb_subjectivity",
    "headlines_count",
]

# Keep only rows where all required columns are non-null
all_features = behaviour_features + sentiment_features + [target_col]
df = df.dropna(subset=all_features)

print("\nRows available after dropping NaNs:", len(df))

# ======================================================
# 2. TRAIN / TEST SPLIT (TIME-BASED)
# ======================================================

def time_split(df_subset, feature_cols, target_col="close", test_size=0.2):
    X = df_subset[feature_cols]
    y = df_subset[target_col]

    split_idx = int(len(df_subset) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test

# same X_train/X_test indices for all models
X_train_behav, X_test_behav, y_train, y_test = time_split(
    df, behaviour_features, target_col
)
X_train_full, X_test_full, _, _ = time_split(
    df, behaviour_features + sentiment_features, target_col
)

test_dates = df["date"].iloc[int(len(df) * 0.8):]  # last 20% dates

# ======================================================
# 3. TRAIN & EVALUATE MODELS
# ======================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, label):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"\n=== {label} ===")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))
    return pred, {"name": label, "MAE": mae, "RMSE": rmse, "R2": r2}

# 1) LR – behaviour
lr_base = LinearRegression()
lr_base_pred, lr_base_metrics = evaluate_model(
    lr_base,
    X_train_behav,
    y_train,
    X_test_behav,
    y_test,
    "Linear Regression – Behaviour Only",
)

# 2) RF – behaviour
rf_base = RandomForestRegressor(n_estimators=200, random_state=42)
rf_base_pred, rf_base_metrics = evaluate_model(
    rf_base,
    X_train_behav,
    y_train,
    X_test_behav,
    y_test,
    "Random Forest – Behaviour Only",
)

# 3) LR – behaviour + sentiment
lr_full = LinearRegression()
lr_full_pred, lr_full_metrics = evaluate_model(
    lr_full,
    X_train_full,
    y_train,
    X_test_full,
    y_test,
    "Linear Regression – Behaviour + Sentiment",
)

# 4) RF – behaviour + sentiment
rf_full = RandomForestRegressor(n_estimators=200, random_state=42)
rf_full_pred, rf_full_metrics = evaluate_model(
    rf_full,
    X_train_full,
    y_train,
    X_test_full,
    y_test,
    "Random Forest – Behaviour + Sentiment",
)

# Collect metrics for bar chart
metrics_list = [lr_base_metrics, rf_base_metrics, lr_full_metrics, rf_full_metrics]

# ======================================================
# 4. CHART 1 – ACTUAL VS PRED (LINEAR REGRESSION)
# ======================================================

plt.figure(figsize=(10, 5))
plt.plot(test_dates, y_test.values, marker="o", label="Actual")
plt.plot(test_dates, lr_base_pred, marker="x", label="LR – Behaviour Only")
plt.plot(test_dates, lr_full_pred, marker="s", label="LR – Behaviour + Sentiment")
plt.title("Actual vs Predicted Close – Linear Regression")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================
# 5. CHART 2 – ACTUAL VS PRED (RANDOM FOREST)
# ======================================================

plt.figure(figsize=(10, 5))
plt.plot(test_dates, y_test.values, marker="o", label="Actual")
plt.plot(test_dates, rf_base_pred, marker="x", label="RF – Behaviour Only")
plt.plot(test_dates, rf_full_pred, marker="s", label="RF – Behaviour + Sentiment")
plt.title("Actual vs Predicted Close – Random Forest")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================
# 6. CHART 3 – R2 COMPARISON BAR CHART
# ======================================================

model_names = [m["name"] for m in metrics_list]
r2_values = [m["R2"] for m in metrics_list]

plt.figure(figsize=(10, 5))
bars = plt.bar(model_names, r2_values)
plt.title("Model R² Comparison")
plt.ylabel("R²")
plt.xticks(rotation=20, ha="right")

# label bars
for bar, val in zip(bars, r2_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

# ======================================================
# 7. CHART 4 – NORMALISED BEHAVIOURAL INDICATORS
# ======================================================

behav_df = df[["date"] + behaviour_features].copy()

# min-max normalise
for col in behaviour_features:
    c = behav_df[col]
    behav_df[col] = (c - c.min()) / (c.max() - c.min() + 1e-9)

plt.figure(figsize=(12, 6))
for col in behaviour_features:
    plt.plot(behav_df["date"], behav_df[col], label=col)

plt.title("Normalized Behavioural & Technical Indicators (0–1 Scale)")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================
# 8. CHART 5 – SENTIMENT INDICATORS OVER TIME
# ======================================================

sent_df = df[["date", "finbert_pos", "finbert_neg", "tb_polarity"]].copy()

plt.figure(figsize=(12, 6))
plt.plot(sent_df["date"], sent_df["finbert_pos"], marker="o", label="FinBERT Positive")
plt.plot(sent_df["date"], sent_df["finbert_neg"], marker="o", label="FinBERT Negative")
plt.plot(sent_df["date"], sent_df["tb_polarity"], marker="o", label="TextBlob Polarity")

plt.title("Daily Sentiment Indicators (FinBERT + TextBlob)")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.legend()
plt.tight_layout()
plt.show()

