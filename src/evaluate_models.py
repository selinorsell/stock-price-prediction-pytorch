from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("--- Evaluate LSTM and GRU Models ---")

ticker = "AAPL"
lookback = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        last_time_step = lstm_output[:, -1, :]
        prediction = self.output_layer(last_time_step)
        return prediction


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_output, _ = self.gru(x)
        last_time_step = gru_output[:, -1, :]
        prediction = self.output_layer(last_time_step)
        return prediction


X_test = np.load(PROCESSED_DATA_DIR / f"{ticker}_X_test.npy")
y_test = np.load(PROCESSED_DATA_DIR / f"{ticker}_y_test.npy")

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

print("\n--- Loaded Test Data ---")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


input_size = 1
hidden_size = 32
num_layers = 2
output_size = 1

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
gru_model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)

lstm_model.load_state_dict(
    torch.load(MODEL_DIR / f"{ticker}_lstm_model.pth", map_location=device)
)

gru_model.load_state_dict(
    torch.load(MODEL_DIR / f"{ticker}_gru_model.pth", map_location=device)
)

lstm_model.eval()
gru_model.eval()

with torch.no_grad():
    lstm_predictions_scaled = lstm_model(X_test_tensor).cpu().numpy()
    gru_predictions_scaled = gru_model(X_test_tensor).cpu().numpy()


lstm_mse_scaled = mean_squared_error(y_test, lstm_predictions_scaled)
gru_mse_scaled = mean_squared_error(y_test, gru_predictions_scaled)

lstm_rmse_scaled = np.sqrt(lstm_mse_scaled)
gru_rmse_scaled = np.sqrt(gru_mse_scaled)

print("\n--- Scaled Evaluation Results ---")
print("LSTM MSE:", lstm_mse_scaled)
print("LSTM RMSE:", lstm_rmse_scaled)
print("GRU MSE:", gru_mse_scaled)
print("GRU RMSE:", gru_rmse_scaled)


clean_close_path = PROCESSED_DATA_DIR / f"{ticker}_clean_close_prices.csv"
close_df = pd.read_csv(clean_close_path)

close_values = close_df[["Close"]].values

train_ratio = 0.8
train_size = int(len(close_values) * train_ratio)

train_data = close_values[:train_size]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_data)

y_test_original = scaler.inverse_transform(y_test)
lstm_predictions_original = scaler.inverse_transform(lstm_predictions_scaled)
gru_predictions_original = scaler.inverse_transform(gru_predictions_scaled)

lstm_mse_original = mean_squared_error(y_test_original, lstm_predictions_original)
gru_mse_original = mean_squared_error(y_test_original, gru_predictions_original)

lstm_rmse_original = np.sqrt(lstm_mse_original)
gru_rmse_original = np.sqrt(gru_mse_original)

print("\n--- Original Price Evaluation Results ---")
print("LSTM MSE:", lstm_mse_original)
print("LSTM RMSE:", lstm_rmse_original)
print("GRU MSE:", gru_mse_original)
print("GRU RMSE:", gru_rmse_original)


comparison_df = pd.DataFrame({
    "Model": ["LSTM", "GRU"],
    "Scaled_MSE": [lstm_mse_scaled, gru_mse_scaled],
    "Scaled_RMSE": [lstm_rmse_scaled, gru_rmse_scaled],
    "Original_MSE": [lstm_mse_original, gru_mse_original],
    "Original_RMSE": [lstm_rmse_original, gru_rmse_original]
})

comparison_path = METRICS_DIR / f"{ticker}_model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\n--- Model Comparison ---")
print(comparison_df)
print("\nComparison saved to:", comparison_path)


plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label="Actual Price")
plt.plot(lstm_predictions_original, label="LSTM Prediction")
plt.plot(gru_predictions_original, label="GRU Prediction")
plt.xlabel("Test Time Step")
plt.ylabel("Close Price")
plt.title(f"{ticker} Actual vs Predicted Close Price")
plt.legend()
plt.tight_layout()

plot_path = FIGURE_DIR / f"{ticker}_actual_vs_predicted.png"
plt.savefig(plot_path)
plt.close()

print("Prediction plot saved to:", plot_path)


if lstm_rmse_original < gru_rmse_original:
    better_model = "LSTM"
elif gru_rmse_original < lstm_rmse_original:
    better_model = "GRU"
else:
    better_model = "Both models performed equally"

print("\n--- Conclusion ---")
print("Better model based on original RMSE:", better_model)