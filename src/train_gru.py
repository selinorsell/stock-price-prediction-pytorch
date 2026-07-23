import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("--- Train GRU Model ---")

ticker = "AAPL"

X_train = np.load(PROCESSED_DATA_DIR / f"{ticker}_X_train.npy")
y_train = np.load(PROCESSED_DATA_DIR / f"{ticker}_y_train.npy")

print("\n--- Loaded Training Data ---")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)


class StockPriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


train_dataset = StockPriceDataset(X_train_tensor, y_train_tensor)

batch_size = 32

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False
)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

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


input_size = 1
hidden_size = 32
num_layers = 2
output_size = 1
learning_rate = 0.001
epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n--- Device ---")
print("Using device:", device)

model = GRUModel(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size
).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("\n--- Model ---")
print(model)

print("\n--- Training Started ---")

start_time = time.time()

training_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_X)

        loss = loss_function(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(train_loader)
    training_losses.append(average_epoch_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss:.6f}")

end_time = time.time()
training_time = end_time - start_time

print("\n--- Training Finished ---")
print(f"Training time: {training_time:.2f} seconds")

model_path = MODEL_DIR / f"{ticker}_gru_model.pth"
torch.save(model.state_dict(), model_path)

print("GRU model saved to:", model_path)

losses_path = METRICS_DIR / f"{ticker}_gru_training_losses.npy"
np.save(losses_path, np.array(training_losses))

print("Training losses saved to:", losses_path)

time_path = METRICS_DIR / f"{ticker}_gru_training_time.txt"
with open(time_path, "w", encoding="utf-8") as file:
    file.write(str(training_time))

print("Training time saved to:", time_path)

print("\n--- Final Training Result ---")
print("Final training loss:", training_losses[-1])