import pandas as pd
import numpy as np
import yfinance as yf

# -------------------------
# 1. DOWNLOAD RAW MARKET DATA
# -------------------------

sp500 = yf.download("^GSPC", period="60d", interval="1d", auto_adjust=True, progress=False)
vix = yf.download("^VIX", period="60d", interval="1d", auto_adjust=True, progress=False)

# Rename price columns
sp500 = sp500.rename(columns={
    "Open": "open_raw",
    "High": "high_raw",
    "Low": "low_raw",
    "Close": "close_raw",
    "Volume": "volume_raw"
})

# Merge VIX fear index
sp500["vix"] = vix["Close"]

# -------------------------
# 2. ADD BEHAVIOURAL FINANCE FEATURES
# -------------------------

# 1. Momentum (Herding)
sp500["ret1"] = sp500["close_raw"].pct_change()

# 2. Trend-following (EMA = herding)
sp500["ema_10"] = sp500["close_raw"].ewm(span=10, adjust=False).mean()

# 3. Overreaction (Z-score)
sp500["z_close_20"] = (
    sp500["close_raw"] - sp500["close_raw"].rolling(20).mean()
) / sp500["close_raw"].rolling(20).std()

# 4. Fear measure (VIX Z-score)
sp500["vix_z_20"] = (
    sp500["vix"] - sp500["vix"].rolling(20).mean()
) / sp500["vix"].rolling(20).std()

# Use close_raw as final “close” column for models
sp500["close"] = sp500["close_raw"]

# -------------------------
# 3. CLEANING
# -------------------------
sp500 = sp500.dropna()
sp500 = sp500.reset_index()  # exposes "Date" column
sp500 = sp500.rename(columns={"Date": "date"})

# -------------------------
# 4. SAVE CLEAN DATASET
# -------------------------
sp500.to_csv("sp500_60d_clean.csv", index=False)

print("✅ Clean dataset created with behavioural features!")
print("Rows:", len(sp500), "Columns:", list(sp500.columns))
