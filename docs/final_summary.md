\# Final Project Summary



\## Project Title



Stock Price Prediction with PyTorch



\## Program



Microsoft AI Innovation Summer Program



\## Objective



The objective of this project is to build a time-series forecasting pipeline using PyTorch and compare two recurrent neural network models, LSTM and GRU, for stock price prediction.



The project uses historical Apple (AAPL) stock closing prices. The models use the previous 20 days of closing prices to predict the next day's closing price.



\## Dataset



The dataset was downloaded using the `yfinance` library.



\- Stock ticker: AAPL

\- Date range: 2015-01-01 to 2025-12-31

\- Main feature: Close price

\- Train/test split: 80% training, 20% testing

\- Scaling method: MinMaxScaler with range `\[-1, 1]`

\- Sequence length: 20 days



\## Models



Two models were implemented and compared:



\- LSTM

\- GRU



Both models used the same main hyperparameters:



\- Input size: 1

\- Hidden size: 32

\- Number of layers: 2

\- Output size: 1

\- Learning rate: 0.001

\- Epochs: 50

\- Batch size: 32

\- Loss function: MSELoss

\- Optimizer: Adam



\## Results



| Model | Scaled MSE | Scaled RMSE | Original MSE | Original RMSE |

|---|---:|---:|---:|---:|

| LSTM | 0.0109 | 0.1043 | 81.5696 | 9.0316 |

| GRU | 0.0348 | 0.1865 | 260.7215 | 16.1469 |



The LSTM model performed better in this experiment because it achieved a lower Original RMSE.



\## Main Output



The project includes:



\- Data download and preparation scripts

\- Sequence creation for time-series forecasting

\- PyTorch Dataset and DataLoader

\- LSTM training script

\- GRU training script

\- Model evaluation script

\- Actual vs predicted plot

\- README documentation

\- Project log



\## Conclusion



This project demonstrates the full workflow of a deep learning time-series forecasting project, from data collection to model comparison.



The LSTM model performed better than the GRU model in this experiment. However, stock price prediction is a complex task. Historical closing prices alone do not include all market factors, such as news, company announcements, investor sentiment, and macroeconomic events.



Therefore, this project should be interpreted as an educational machine learning project rather than a real financial prediction system.

