import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("sp500_60d_clean.csv", parse_dates=["date"])
df = df.set_index("date")

# 2. Summary statistics
print("\n===== SUMMARY STATISTICS =====")
print(df.describe())

# 3. Select features (behavioural finance indicators)
features = ["ret1", "ema_10", "z_close_20", "vix", "vix_z_20"]
df = df.dropna(subset=features)

X = df[features]
y = df["close"]

# 4. Train-test split (time-based)
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 6. Random Forest Model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 7. Accuracy Metrics
def metrics(name, true, pred):
    mse = mean_squared_error(true, pred)
    rmse = mse ** 0.5
    print(f"\n=== {name} ===")
    print("MAE:", mean_absolute_error(true, pred))
    print("RMSE:", rmse)
    print("R2:", r2_score(true, pred))

metrics("Linear Regression", y_test, lr_pred)
metrics("Random Forest", y_test, rf_pred)

# 8. Plot comparison
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(lr_pred, label="Linear Regression")
plt.plot(rf_pred, label="Random Forest")
plt.legend()
plt.title("Model Comparison: Actual vs Predicted")
plt.show()
