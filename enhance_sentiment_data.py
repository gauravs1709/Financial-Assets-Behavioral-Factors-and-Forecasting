import pandas as pd
import numpy as np

# Load your price dataset
df = pd.read_csv("sp500_60d_clean.csv", parse_dates=["date"])

# Create synthetic sentiment for all available dates
np.random.seed(42)

df["finbert_pos"] = np.random.uniform(0.05, 0.30, len(df))
df["finbert_neg"] = np.random.uniform(0.05, 0.35, len(df))
df["tb_polarity"] = np.random.uniform(-0.2, 0.3, len(df))
df["tb_subjectivity"] = np.random.uniform(0.2, 0.8, len(df))
df["headlines_count"] = np.random.randint(5, 35, len(df))

# Normalize sentiment so pos + neg approx match probability style
total = df["finbert_pos"] + df["finbert_neg"]
df["finbert_pos"] = df["finbert_pos"] / total
df["finbert_neg"] = df["finbert_neg"] / total

# Save enhanced dataset
df.to_csv("sp500_60d_with_sentiment.csv", index=False)

print("Enhanced sentiment dataset created successfully!")
print(df.head())
