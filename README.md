# Stock Price Prediction with PyTorch

Microsoft AI Innovation Summer Program project.

## Project Goal

This project compares **LSTM** and **GRU** recurrent neural network models for
one-step-ahead stock closing-price prediction.

The project covers:

- stock data collection and exploratory analysis
- chronological train/test splitting
- scaling without data leakage
- sliding-window sequence generation
- LSTM and GRU model training in PyTorch
- evaluation with MSE and RMSE
- comparison of prediction quality and training time
- discussion of the limitations of stock-price forecasting

## Planned Repository Structure

```text
stock-price-prediction-pytorch/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── results/
│   ├── figures/
│   └── metrics/
├── docs/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name stock-ml --display-name "Python (stock-ml)"
```

## Run

Open the project in VS Code, open the notebook in `notebooks/`, and select the
kernel named **Python (stock-ml)**.

## Models

- LSTM
- GRU

Both models will initially use the same input data, lookback window, hidden
dimension, number of layers, optimizer, and number of epochs to make the
comparison fair.

## Evaluation

The final comparison will report:

- Test MSE
- Test RMSE
- Training time
- Actual vs. predicted price plots

## Important Limitation

This is an educational time-series forecasting project, not financial advice.
Historical price patterns alone do not reliably capture all information that
moves financial markets.
