import numpy as np

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

print("--- Create Sliding Window Sequences ---")

ticker = "AAPL"
lookback = 20

train_scaled_path = PROCESSED_DATA_DIR / f"{ticker}_train_scaled.npy"
test_scaled_path = PROCESSED_DATA_DIR / f"{ticker}_test_scaled.npy"

train_scaled = np.load(train_scaled_path)
test_scaled = np.load(test_scaled_path)

print("\n--- Loaded Scaled Data ---")
print("Train scaled shape:", train_scaled.shape)
print("Test scaled shape:", test_scaled.shape)


def create_sequences(data, lookback):
    X = []
    y = []

    for i in range(len(data) - lookback):
        input_sequence = data[i:i + lookback]
        target_value = data[i + lookback]

        X.append(input_sequence)
        y.append(target_value)

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled, lookback)
X_test, y_test = create_sequences(test_scaled, lookback)

print("\n--- Created Sequences ---")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\n--- Example Training Sample ---")
print("First X_train sample:")
print(X_train[0])
print("First y_train target:")
print(y_train[0])

print("\n--- Shape Meaning ---")
print("X_train shape means: samples, lookback days, features")
print("For LSTM/GRU, input shape should be:")
print("(number of samples, sequence length, number of features)")


X_train_path = PROCESSED_DATA_DIR / f"{ticker}_X_train.npy"
y_train_path = PROCESSED_DATA_DIR / f"{ticker}_y_train.npy"
X_test_path = PROCESSED_DATA_DIR / f"{ticker}_X_test.npy"
y_test_path = PROCESSED_DATA_DIR / f"{ticker}_y_test.npy"

np.save(X_train_path, X_train)
np.save(y_train_path, y_train)
np.save(X_test_path, X_test)
np.save(y_test_path, y_test)

print("\n--- Saved Files ---")
print("X_train saved to:", X_train_path)
print("y_train saved to:", y_train_path)
print("X_test saved to:", X_test_path)
print("y_test saved to:", y_test_path)

print("\n--- Final Project Logic ---")
print(f"Lookback window: {lookback} days")
print("Input: previous 20 scaled closing prices")
print("Target: next scaled closing price")
print("These arrays are now ready to be converted into PyTorch tensors.")