import pandas as pd

df = pd.read_csv("sp500_60d_with_sentiment.csv", parse_dates=["date"])

# Remove the first corrupted row where date is NaT or contains "^GSPC"
df = df[df["date"].notna()]
df = df[~df["close_raw"].astype(str).str.contains("\^GSPC")]

df.to_csv("sp500_60d_with_sentiment_clean.csv", index=False)

print("Cleaned dataset saved as: sp500_60d_with_sentiment_clean.csv")
print(df.head())
print(df.tail())
