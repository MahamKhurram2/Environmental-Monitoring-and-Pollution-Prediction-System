import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
data = pd.read_csv('data/results.csv')

# Extract actual data and predictions
# Assuming 'Live Data' and 'Predictions' columns contain the respective data
actuals = [eval(live).get("main", {}).get("temp") for live in data["Live Data"]]  # Extracting 'temp' from 'Live Data'
predictions = [sum(eval(pred)) / len(eval(pred)) for pred in data["Predictions"]]  # Calculating average of predictions

# Analyze live data and predictions
print("Sample Data:")
print(data.head())

# Average predictions
average_predictions = sum(predictions) / len(predictions)
print("Average Predictions:", average_predictions)

# Check trends in predictions
plt.figure(figsize=(10, 5))
plt.plot(actuals, label="Actual Temperatures", marker='o')
plt.plot(predictions, label="Predicted Values", marker='x')
plt.legend()
plt.title("Actual vs Predicted Trends")
plt.xlabel("Index")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.show()
