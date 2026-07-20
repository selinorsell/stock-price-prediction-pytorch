import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

print("--- PyTorch Dataset and DataLoader ---")

ticker = "AAPL"

X_train_path = PROCESSED_DATA_DIR / f"{ticker}_X_train.npy"
y_train_path = PROCESSED_DATA_DIR / f"{ticker}_y_train.npy"
X_test_path = PROCESSED_DATA_DIR / f"{ticker}_X_test.npy"
y_test_path = PROCESSED_DATA_DIR / f"{ticker}_y_test.npy"


X_train = np.load(X_train_path)
y_train = np.load(y_train_path)
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

print("\n--- Loaded NumPy Arrays ---")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("\n--- Converted to PyTorch Tensors ---")
print("X_train tensor shape:", X_train_tensor.shape)
print("y_train tensor shape:", y_train_tensor.shape)
print("X_test tensor shape:", X_test_tensor.shape)
print("y_test tensor shape:", y_test_tensor.shape)


class StockPriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


train_dataset = StockPriceDataset(X_train_tensor, y_train_tensor)
test_dataset = StockPriceDataset(X_test_tensor, y_test_tensor)

print("\n--- Dataset Sizes ---")
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))


batch_size = 32

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("\n--- DataLoader Created ---")
print("Batch size:", batch_size)
print("Number of train batches:", len(train_loader))
print("Number of test batches:", len(test_loader))


print("\n--- First Batch Example ---")

for batch_X, batch_y in train_loader:
    print("batch_X shape:", batch_X.shape)
    print("batch_y shape:", batch_y.shape)
    break


print("\n--- Shape Meaning ---")
print("batch_X shape = batch size, sequence length, number of features")
print("For our project, this should look like:")
print("(32, 20, 1)")
print("This is the input format expected by LSTM and GRU models.")