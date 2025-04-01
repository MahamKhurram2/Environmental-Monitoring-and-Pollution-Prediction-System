import pandas as pd
import matplotlib.pyplot as plt

# Evaluation based on MLflow outputs

# ARIMA Evaluation
print("\n--- ARIMA Evaluation ---")
# Display ARIMA RMSE and MAE from MLflow
print("ARIMA Model: Best RMSE = 0.0 (as per MLflow logs)")

# Show ARIMA forecast plot
arima_forecast_img = "arima_forecast.png"  # Use the image you uploaded
plt.imshow(plt.imread(arima_forecast_img))
plt.axis('off')
plt.title("ARIMA Forecast vs Actual")
plt.show()

# LSTM Evaluation
print("\n--- LSTM Evaluation ---")
# Display LSTM RMSE and MAE from MLflow
print("LSTM Model: Best RMSE = 7.629394538355427e-08 (as per MLflow logs)")

# Show LSTM evaluation details from the MLflow logs or other visual outputs
lstm_summary_img = "LSTMA.PNG"  # Use your uploaded LSTM image
plt.imshow(plt.imread(lstm_summary_img))
plt.axis('off')
plt.title("LSTM Metrics Overview")
plt.show()

# Final Model Selection
print("\n--- Model Selection ---")
if 0.0 < 7.629394538355427e-08:  # Comparing RMSE values
    print("Selected Model: ARIMA due to better RMSE.")
else:
    print("Selected Model: LSTM due to better RMSE.")
