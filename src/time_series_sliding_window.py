import numpy as np


def create_sliding_windows(data, lookback):
    X = []
    y = []

    for i in range(len(data) - lookback):
        input_sequence = data[i:i + lookback]
        target_value = data[i + lookback]

        X.append(input_sequence)
        y.append(target_value)

    return np.array(X), np.array(y)


print("--- Time Series Data ---")

prices = np.array([100, 102, 101, 105, 108, 110, 109, 112, 115, 117])

print("Prices:", prices)
print("Number of prices:", len(prices))


print("\n--- Sliding Window ---")

lookback = 3

X, y = create_sliding_windows(prices, lookback)

print("X:")
print(X)

print("\ny:")
print(y)

print("\nX shape:", X.shape)
print("y shape:", y.shape)


print("\n--- Readable Input-Target Pairs ---")

for i in range(len(X)):
    print(f"Input: {X[i]} -> Target: {y[i]}")


print("\n--- Chronological Train/Test Split ---")

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

print("X_train:")
print(X_train)

print("\ny_train:")
print(y_train)

print("\nX_test:")
print(X_test)

print("\ny_test:")
print(y_test)


print("\n--- Final Project Logic ---")
print("In the final project:")
print("Input = previous 20 closing prices")
print("Target = next day's closing price")
print("The same sliding window logic will be used for LSTM and GRU models.")