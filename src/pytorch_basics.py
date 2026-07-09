import torch
import torch.nn as nn
import torch.optim as optim

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("\n--- Tensor Basics ---")

prices = torch.tensor([100.0, 102.0, 101.0, 105.0, 108.0])

print("Prices:", prices)
print("Shape:", prices.shape)
print("Data type:", prices.dtype)
print("\n--- 2D Tensor ---")

stock_data = torch.tensor([
    [100.0, 1000.0],
    [102.0, 1200.0],
    [101.0, 900.0],
    [105.0, 1500.0]
])

print(stock_data)
print("Shape:", stock_data.shape)
print("\n--- Feature and Target Tensors ---")

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [45.0],
    [55.0],
    [65.0],
    [75.0],
    [85.0]
])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\n--- Simple Linear Model ---")


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        prediction = self.linear(x)
        return prediction


model = SimpleLinearModel()

print(model)
print("\n--- Loss Function and Optimizer ---")

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Loss function:", loss_function)
print("Optimizer:", optimizer)
print("\n--- Training Loop ---")

epochs = 100

for epoch in range(epochs):
    predictions = model(X)

    loss = loss_function(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        print("\n--- Prediction After Training ---")

new_hours = torch.tensor([[6.0]])

predicted_score = model(new_hours)

print("Predicted score for 6 study hours:", predicted_score.item())