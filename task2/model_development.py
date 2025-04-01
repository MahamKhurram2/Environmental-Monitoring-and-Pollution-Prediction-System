import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
import mlflow.tensorflow
import mlflow.sklearn
import joblib

# Load cleaned data
data = pd.read_csv('data/cleaned_merged_data.csv')

# Ensure the target column exists
target = 'temperature'  # Update to match your column name
if target not in data.columns:
    raise ValueError(f"The dataset must contain the target column '{target}'.")

# Use the DataFrame index as a sequential "time" variable
data['Index'] = range(len(data))  # Create a sequential index
data.set_index('Index', inplace=True)

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

print(f"Training Data Shape: {train.shape}")
print(f"Testing Data Shape: {test.shape}")

# --- MLflow Tracking ---
mlflow.set_experiment("Pollution Trend Prediction")

# --- ARIMA Model ---
print("\n--- ARIMA Model ---")
with mlflow.start_run(run_name="ARIMA Model"):
    try:
        # Fit the ARIMA model
        model_arima = ARIMA(train[target], order=(5, 1, 0))
        model_arima_fit = model_arima.fit()

        # Forecast
        forecast_arima = model_arima_fit.forecast(steps=len(test))

        # Calculate metrics
        rmse_arima = np.sqrt(mean_squared_error(test[target], forecast_arima))
        mae_arima = mean_absolute_error(test[target], forecast_arima)

        print(f"ARIMA RMSE: {rmse_arima}")
        print(f"ARIMA MAE: {mae_arima}")

        # Log parameters, metrics, and artifacts
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("order", (5, 1, 0))
        mlflow.log_metric("rmse", rmse_arima)
        mlflow.log_metric("mae", mae_arima)

        # Plot ARIMA Predictions
        plt.figure(figsize=(14, 5))
        plt.plot(test.index, test[target], label='Actual Temperature')
        plt.plot(test.index, forecast_arima, label='ARIMA Forecast')
        plt.title('ARIMA Forecast vs Actual Temperature')
        plt.xlabel('Index')
        plt.ylabel('Temperature')
        plt.legend()
        plt.savefig("arima_forecast.png")
        mlflow.log_artifact("arima_forecast.png")
        plt.show()

    except Exception as e:
        print(f"ARIMA Model Error: {e}")

# --- Save the Best ARIMA Model ---
print("\n--- Saving Best ARIMA Model ---")
try:
    best_arima_model = ARIMA(train[target], order=(5, 1, 0)).fit()
    joblib.dump(best_arima_model, "arima.pkl")
    print(f"Best ARIMA model saved as 'arima.pkl' with order (5, 1, 0)")
except Exception as e:
    print(f"Error saving ARIMA model: {e}")

# --- LSTM Model ---
print("\n--- LSTM Model ---")
def create_dataset(series, time_step=1):
    X, Y = [], []
    for i in range(len(series) - time_step - 1):
        a = series[i:(i + time_step)]
        X.append(a)
        Y.append(series[i + time_step])
    return np.array(X), np.array(Y)

with mlflow.start_run(run_name="LSTM Model"):
    try:
        # Prepare data for LSTM
        series = data[target].values

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        series_scaled = scaler.fit_transform(series.reshape(-1, 1))

        # Create datasets
        time_step = 4
        X, Y = create_dataset(series_scaled, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]

        # Build LSTM model
        model_lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50, return_sequences=False),
            Dense(1)
        ])

        # Compile the model
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')

        # Enable MLflow autologging for TensorFlow
        mlflow.tensorflow.autolog()

        # Train the model
        model_lstm.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)

        # Predict
        predictions = model_lstm.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Inverse transform Y_test
        Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # Calculate metrics
        rmse_lstm = np.sqrt(mean_squared_error(Y_test_inv, predictions))
        mae_lstm = mean_absolute_error(Y_test_inv, predictions)

        print(f"LSTM RMSE: {rmse_lstm}")
        print(f"LSTM MAE: {mae_lstm}")

        # Log metrics and model
        mlflow.log_metric("rmse", rmse_lstm)
        mlflow.log_metric("mae", mae_lstm)
        model_lstm.save("lstm_model.h5")
        mlflow.log_artifact("lstm_model.h5")

    except Exception as e:
        print(f"LSTM Model Error: {e}")
