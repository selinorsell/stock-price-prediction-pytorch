\# Project Log



\## Project Name



Stock Price Prediction with PyTorch



\## Program



Microsoft AI Innovation Summer Program



\## Project Goal



The goal of this project is to build a time-series stock price prediction pipeline using PyTorch and compare two recurrent neural network models: LSTM and GRU.



The project uses historical Apple (AAPL) stock closing prices. The models learn from the previous 20 days of closing prices and try to predict the next day's closing price.



\---



\## Week 1: Environment Setup and Machine Learning Basics



During the first week, I focused on setting up the development environment and reviewing the basic concepts required for the project.



Completed tasks:



\- Created the project folder structure

\- Created and activated a Python virtual environment

\- Installed the required Python libraries

\- Checked the Python, NumPy, Pandas, Matplotlib, scikit-learn, and PyTorch installations

\- Reviewed basic machine learning concepts

\- Practiced NumPy and Pandas operations

\- Studied simple regression logic before moving into deep learning



Main learning outcomes:



\- Understanding the difference between features and targets

\- Understanding train/test split

\- Understanding the importance of preprocessing

\- Getting comfortable with Python tools used in ML projects



\---



\## Week 2: Data Collection and Data Preparation



During the second week, I worked on collecting and preparing the stock price data.



Completed tasks:



\- Downloaded historical AAPL stock data using `yfinance`

\- Saved raw stock data

\- Selected the Close price as the main feature

\- Cleaned the Close price time series

\- Split the dataset chronologically into training and test sets

\- Applied MinMax scaling with the range `\[-1, 1]`

\- Saved processed training and test arrays



Important implementation detail:



The scaler was fitted only on the training data and then applied to the test data.



```python

train\_scaled = scaler.fit\_transform(train\_data)

test\_scaled = scaler.transform(test\_data)

```



This helped avoid data leakage from the test set into the training process.



Main learning outcomes:



\- Understanding why time-series data should not be randomly shuffled

\- Understanding chronological train/test splitting

\- Understanding data leakage

\- Learning how scaling affects neural network training



\---



\## Week 3: Sequence Creation and PyTorch DataLoader



During the third week, I prepared the data for recurrent neural networks.



Completed tasks:



\- Created sliding window sequences

\- Used a lookback window of 20 days

\- Converted data into the shape required by recurrent models

\- Created a custom PyTorch Dataset class

\- Created PyTorch DataLoaders for model training



Sliding window logic:



```text

\[day 1, day 2, ..., day 20] → day 21

\[day 2, day 3, ..., day 21] → day 22

```



Final input shape:



```text

(samples, 20, 1)

```



Main learning outcomes:



\- Understanding how time-series data becomes supervised learning data

\- Understanding sequence length

\- Understanding PyTorch Dataset and DataLoader

\- Learning why LSTM and GRU expect 3D input



\---



\## Week 4: Model Training and Evaluation



During the final development week, I implemented and compared LSTM and GRU models.



Completed tasks:



\- Built an LSTM model in PyTorch

\- Trained the LSTM model

\- Built a GRU model in PyTorch

\- Trained the GRU model

\- Evaluated both models on the test set

\- Calculated MSE and RMSE

\- Converted scaled predictions back to the original price scale

\- Created an actual vs predicted price plot

\- Compared LSTM and GRU results



Model hyperparameters:



| Parameter | Value |

|---|---:|

| Input size | 1 |

| Hidden size | 32 |

| Number of layers | 2 |

| Output size | 1 |

| Learning rate | 0.001 |

| Epochs | 50 |

| Batch size | 32 |



Evaluation results:



| Model | Scaled MSE | Scaled RMSE | Original MSE | Original RMSE |

|---|---:|---:|---:|---:|

| LSTM | 0.0109 | 0.1043 | 81.5696 | 9.0316 |

| GRU | 0.0348 | 0.1865 | 260.7215 | 16.1469 |



In this experiment, the LSTM model performed better than the GRU model because it achieved a lower Original RMSE.



Main learning outcomes:



\- Implementing LSTM and GRU models in PyTorch

\- Understanding recurrent neural networks for time-series forecasting

\- Evaluating regression models using MSE and RMSE

\- Comparing models numerically and visually

\- Understanding the limitations of stock price prediction



\---



\## Final Project Output



The final project includes:



\- Data collection script

\- Data preprocessing script

\- Sequence creation script

\- PyTorch Dataset/DataLoader script

\- LSTM training script

\- GRU training script

\- Model evaluation script

\- Actual vs predicted plot

\- README documentation

\- Project log



\---



\## Reflection



This project helped me understand the full machine learning project pipeline from data collection to model evaluation.



At the beginning of the project, I focused on understanding the basics of machine learning, data preparation, and PyTorch. Later, I learned how raw time-series data can be transformed into a supervised learning problem using sliding windows.



The most important part of the project was understanding that model performance does not only depend on the neural network architecture. Data preparation, scaling, train/test splitting, and evaluation methods are also critical.



The LSTM model performed better in this experiment, but the project also showed that stock price prediction is a difficult task. Historical closing prices alone are not enough to capture all real market movements.



Therefore, I see this project as an educational deep learning and time-series forecasting project rather than a financial prediction tool.



\---



\## Possible Future Improvements



Future versions of this project could include:



\- Open, High, Low, and Volume features

\- Technical indicators

\- Validation set

\- Dropout or regularization

\- Hyperparameter tuning

\- More stocks

\- Longer training experiments

\- Comparison with non-neural baseline models

\- Transformer-based time-series models

