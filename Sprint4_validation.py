import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. LOAD DATASETS
# ===============================

train_df = pd.read_csv("sp500_60d_clean.csv")
valid_df = pd.read_csv("sp500_60d_with_sentiment_clean.csv")

# Drop missing values
train_df = train_df.dropna()
valid_df = valid_df.dropna()

target = "close"

features_behaviour = [
    "ema_10", "ret1", "vix", "z_close_20"
]

features_with_sentiment = features_behaviour + [
    "finbert_pos", "finbert_neg", "tb_polarity"
]

# ===============================
# 2. SPLIT DATA
# ===============================

X_train = train_df[features_behaviour]
y_train = train_df[target]

X_valid = valid_df[features_with_sentiment]
y_valid = valid_df[target]

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid[features_behaviour])

# ===============================
# 3. MODELS
# ===============================

lr = LinearRegression()
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# ===============================
# 4. TRAIN MODELS
# ===============================

lr.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

# ===============================
# 5. VALIDATION
# ===============================

lr_pred = lr.predict(X_valid_scaled)
rf_pred = rf.predict(X_valid_scaled)

def metrics(name, y, pred):
    print(f"\n=== {name} ===")
    print("MAE:", mean_absolute_error(y, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, pred)))
    print("R2:", r2_score(y, pred))

metrics("Linear Regression (Validation)", y_valid, lr_pred)
metrics("Random Forest (Validation)", y_valid, rf_pred)

# ===============================
# 6. VISUALISATION
# ===============================

plt.figure(figsize=(10,5))
plt.plot(y_valid.values, label="Actual", linewidth=2)
plt.plot(lr_pred, label="Linear Regression")
plt.plot(rf_pred, label="Random Forest")
plt.legend()
plt.title("Model Validation: Actual vs Predictions")
plt.xlabel("Time")
plt.ylabel("S&P 500 Close")
plt.tight_layout()
plt.savefig("Sprint4_Model_Comparison.png")
plt.close()

print("\n✅ Sprint-4 graph saved as Sprint4_Model_Comparison.png")
