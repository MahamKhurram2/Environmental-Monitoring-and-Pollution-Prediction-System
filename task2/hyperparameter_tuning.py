import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.statsmodels
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Load cleaned merged data
file_path = 'data/cleaned_merged_data.csv'
data = pd.read_csv(file_path)
data['Index'] = range(len(data))  # Create a sequential index
data.set_index('Index', inplace=True)

# Set target
# Use "temperature" as the target column from your CSV
target = 'temperature'  # Ensure column name matches your dataset
if target not in data.columns:
    raise ValueError(f"The dataset must contain the target column '{target}'.")

# Split data
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# ARIMA Hyperparameter Tuning
print("--- Starting ARIMA Hyperparameter Tuning ---")
p_values = range(0, 3)  # Reduced range for smaller data
d_values = range(0, 2)
q_values = range(0, 3)
best_score, best_cfg = float("inf"), None

mlflow.set_experiment("ARIMA_Hyperparameter_Tuning")

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                with mlflow.start_run(run_name=f"ARIMA_{order}"):
                    mlflow.log_param("p", p)
                    mlflow.log_param("d", d)
                    mlflow.log_param("q", q)

                    model = ARIMA(train[target], order=order)
                    model_fit = model.fit()

                    forecast = model_fit.forecast(steps=len(test))

                    rmse = np.sqrt(mean_squared_error(test[target], forecast))
                    mlflow.log_metric("RMSE", rmse)

                    mlflow.statsmodels.log_model(model_fit, f"arima_model_{order}")

                    print(f"ARIMA{order} RMSE={rmse}")

                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
            except Exception as e:
                print(f"ARIMA{order} failed with error: {e}")
                continue

print(f"Best ARIMA{best_cfg} RMSE={best_score}")

# ------------------------------------------------------
# LSTM Hyperparameter Tuning
# ------------------------------------------------------
print("--- Starting LSTM Hyperparameter Tuning ---")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data[target] = scaler.fit_transform(data[[target]])

# Create dataset
def create_dataset(series, time_step=1):
    X, Y = [], []
    for i in range(len(series) - time_step - 1):
        a = series[i:(i + time_step)]
        X.append(a)
        Y.append(series[i + time_step])
    return np.array(X), np.array(Y)

time_step = 5
series = data[target].values
X, Y = create_dataset(series, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Define hyperparameter ranges
epochs_list = [10, 20]
batch_size_list = [16, 32]

best_rmse, best_params = float("inf"), None

mlflow.set_experiment("LSTM_Hyperparameter_Tuning")

for epochs in epochs_list:
    for batch_size in batch_size_list:
        with mlflow.start_run(run_name=f"LSTM_epochs{epochs}_batch{batch_size}"):
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                LSTM(50, return_sequences=False),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

            rmse = np.sqrt(mean_squared_error(Y_test_inv, predictions))
            mae = mean_absolute_error(Y_test_inv, predictions)

            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)

            mlflow.tensorflow.log_model(model, f"lstm_model_epochs{epochs}_batch{batch_size}")

            print(f"LSTM Epochs={epochs}, Batch Size={batch_size} => RMSE={rmse}, MAE={mae}")

            if rmse < best_rmse:
                best_rmse, best_params = rmse, (epochs, batch_size)

print(f"Best LSTM Params: Epochs={best_params[0]}, Batch Size={best_params[1]} with RMSE={best_rmse}")