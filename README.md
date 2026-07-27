\# Stock Price Prediction with PyTorch



This project was developed for the Microsoft AI Innovation Summer Program.



The goal of the project is to build and compare two recurrent neural network models, LSTM and GRU, for stock price prediction using historical Apple (AAPL) closing price data.



\## Project Objective



The project focuses on time-series regression. The models use the previous 20 days of scaled closing prices to predict the next day's closing price.



Input:



```text

Previous 20 closing prices

```



Target:



```text

Next day's closing price

```



\## Dataset



The dataset was collected using the `yfinance` Python library.



\- Ticker: AAPL

\- Date range: 2015-01-01 to 2025-12-31

\- Main feature used: Close price

\- Data type: daily stock price time series



The raw data includes Open, High, Low, Close, and Volume values. For the first version of this project, only the Close price was used.



\## Project Workflow



```text

Download stock data

→ Select Close price

→ Clean data

→ Chronological train/test split

→ MinMax scaling

→ Sliding window sequence creation

→ PyTorch Dataset/DataLoader

→ Train LSTM model

→ Train GRU model

→ Evaluate and compare models

```



\## Repository Structure



```text

stock-price-prediction-pytorch/

├── data/

│   ├── raw/

│   └── processed/

├── models/

├── notebooks/

├── results/

│   ├── figures/

│   └── metrics/

├── src/

│   ├── stock\_data\_preparation.py

│   ├── prepare\_model\_data.py

│   ├── create\_sequences.py

│   ├── dataset\_loader.py

│   ├── train\_lstm.py

│   ├── train\_gru.py

│   └── evaluate\_models.py

├── README.md

├── PROJECT\_LOG.md

└── requirements.txt

```



\## Methods



\### Data Preparation



The stock data was downloaded using `yfinance`. After downloading the data, the Close price column was selected and saved as the main time-series feature.



The data was split chronologically:



\- Older 80% of the data: training set

\- Newer 20% of the data: test set



The data was not shuffled because stock prices are time-series data. Shuffling would break the time order and could cause data leakage.



\### Scaling



The Close price values were scaled using `MinMaxScaler` with the range `\[-1, 1]`.



The scaler was fitted only on the training data:



```python

scaler.fit\_transform(train\_data)

scaler.transform(test\_data)

```



This prevents information from the test period from leaking into the training process.



\### Sliding Window



A lookback window of 20 days was used.



Example:



```text

\[day 1, day 2, ..., day 20] → day 21

\[day 2, day 3, ..., day 21] → day 22

```



This creates the input format required by LSTM and GRU models:



```text

(samples, sequence length, features)

```



In this project:



```text

(samples, 20, 1)

```



\## Models



Two recurrent neural network models were implemented in PyTorch.



\### LSTM



The LSTM model uses:



\- input size: 1

\- hidden size: 32

\- number of layers: 2

\- output size: 1

\- optimizer: Adam

\- loss function: MSELoss

\- epochs: 50



\### GRU



The GRU model uses the same hyperparameters as the LSTM model to make the comparison fair.



\## Results



The models were evaluated using MSE and RMSE.



| Model | Scaled MSE | Scaled RMSE | Original MSE | Original RMSE |

|---|---:|---:|---:|---:|

| LSTM | 0.0109 | 0.1043 | 81.5696 | 9.0316 |

| GRU | 0.0348 | 0.1865 | 260.7215 | 16.1469 |



The `Original\_RMSE` value is easier to interpret because it is measured in the original stock price scale.



In this experiment, the LSTM model achieved a lower error than the GRU model. The LSTM model had an Original RMSE of approximately 9.03, while the GRU model had an Original RMSE of approximately 16.15.



\## Actual vs Predicted Plot



The following plot compares the actual AAPL closing prices with LSTM and GRU predictions on the test set.



!\[Actual vs Predicted](results/figures/AAPL\_actual\_vs\_predicted.png)



\## Interpretation



The model with the lower Original RMSE performed better on the test set.



In this project, the better-performing model was:



```text

LSTM

```



The LSTM model produced predictions that were closer to the actual AAPL closing prices compared to the GRU model in this experiment.



However, stock price prediction is a difficult problem. Historical closing prices alone do not include all factors that affect the market, such as news, macroeconomic events, company announcements, investor sentiment, and unexpected shocks.



Therefore, this project should be understood as an educational time-series forecasting project, not as a real financial prediction tool.



\## How to Run the Project



Create and activate a virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

pip install -r requirements.txt

```



Run the project pipeline:



```powershell

python src\\stock\_data\_preparation.py

python src\\prepare\_model\_data.py

python src\\create\_sequences.py

python src\\dataset\_loader.py

python src\\train\_lstm.py

python src\\train\_gru.py

python src\\evaluate\_models.py

```



\## What I Learned



Through this project, I learned:



\- basic machine learning concepts

\- time-series data preparation

\- chronological train/test splitting

\- data leakage prevention

\- scaling with MinMaxScaler

\- sliding window sequence creation

\- PyTorch Dataset and DataLoader usage

\- LSTM and GRU model implementation

\- model evaluation with MSE and RMSE

\- comparing model predictions visually and numerically



\## Limitations and Future Work



This project only uses historical Close prices. In future work, the model could be improved by using more features such as:



\- Open, High, Low, and Volume

\- technical indicators

\- multiple stocks

\- longer hyperparameter tuning

\- validation set

\- dropout or regularization

\- more advanced time-series models

