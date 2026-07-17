import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


print("--- Stock Data Preparation ---")

ticker = "AAPL"
start_date = "2015-01-01"
end_date = "2025-12-31"

df = yf.download(ticker, start=start_date, end=end_date)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print("\n--- Raw Data ---")
print(df.head())
print(df.tail())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

raw_file_path = RAW_DATA_DIR / f"{ticker}_raw.csv"
df.to_csv(raw_file_path)

print("\nRaw data saved to:", raw_file_path)

close_prices = df[["Close"]].copy()

print("\n--- Close Prices ---")
print(close_prices.head())
print(close_prices.tail())
print("\nShape:", close_prices.shape)

print("\n--- Missing Values ---")
print(close_prices.isnull().sum())

print("\n--- Basic Statistics ---")
print(close_prices.describe())

plt.figure(figsize=(10, 5))
plt.plot(close_prices.index, close_prices["Close"])
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title(f"{ticker} Closing Price")
plt.tight_layout()

figure_path = FIGURE_DIR / f"{ticker}_close_price.png"
plt.savefig(figure_path)
plt.close()

print("\nClose price figure saved to:", figure_path)

processed_file_path = PROCESSED_DATA_DIR / f"{ticker}_close_prices.csv"
close_prices.to_csv(processed_file_path)

print("Processed close price data saved to:", processed_file_path)
