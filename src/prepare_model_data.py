import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


print("--- Prepare Model Data ---")

ticker = "AAPL"

input_file_path = PROCESSED_DATA_DIR / f"{ticker}_close_prices.csv"

df = pd.read_csv(input_file_path)

print("\n--- Loaded Close Price Data ---")
print(df.head())
print(df.tail())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

if "Date" in df.columns and "Close" in df.columns:
    close_df = df[["Date", "Close"]].copy()
else:
  
    close_df = df.copy()
    close_df.columns = ["Date", "Close"]

close_df["Date"] = pd.to_datetime(close_df["Date"])
close_df["Close"] = pd.to_numeric(close_df["Close"], errors="coerce")

close_df = close_df.dropna()

print("\n--- Clean Close Data ---")
print(close_df.head())
print(close_df.tail())
print("\nShape:", close_df.shape)
print("\nMissing values:")
print(close_df.isnull().sum())
close_values = close_df[["Close"]].values

print("\n--- Close Values as NumPy Array ---")
print(close_values[:5])
print("Shape:", close_values.shape)
train_ratio = 0.8
train_size = int(len(close_values) * train_ratio)

train_data = close_values[:train_size]
test_data = close_values[train_size:]

print("\n--- Chronological Train/Test Split ---")
print("Total samples:", len(close_values))
print("Train samples:", len(train_data))
print("Test samples:", len(test_data))

print("\nFirst train value:", train_data[0])
print("Last train value:", train_data[-1])
print("First test value:", test_data[0])
print("Last test value:", test_data[-1])
scaler = MinMaxScaler(feature_range=(-1, 1))

train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

print("\n--- Scaled Data ---")
print("Train scaled min:", train_scaled.min())
print("Train scaled max:", train_scaled.max())
print("Test scaled min:", test_scaled.min())
print("Test scaled max:", test_scaled.max())

print("\nFirst 5 scaled train values:")
print(train_scaled[:5])
train_scaled_path = PROCESSED_DATA_DIR / f"{ticker}_train_scaled.npy"
test_scaled_path = PROCESSED_DATA_DIR / f"{ticker}_test_scaled.npy"

np.save(train_scaled_path, train_scaled)
np.save(test_scaled_path, test_scaled)

clean_close_path = PROCESSED_DATA_DIR / f"{ticker}_clean_close_prices.csv"
close_df.to_csv(clean_close_path, index=False)

print("\n--- Saved Files ---")
print("Clean close prices saved to:", clean_close_path)
print("Train scaled data saved to:", train_scaled_path)
print("Test scaled data saved to:", test_scaled_path)