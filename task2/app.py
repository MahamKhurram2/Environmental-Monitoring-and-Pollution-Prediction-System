from flask import Flask, request, render_template
from prometheus_flask_exporter import PrometheusMetrics
import joblib

# Load the saved ARIMA model
model_path = "arima.pkl"  # Ensure this file exists
try:
    arima_model = joblib.load(model_path)
    print("ARIMA model loaded successfully.")
except FileNotFoundError:
    raise Exception(f"Model file '{model_path}' not found. Ensure the ARIMA model is saved.")

# Initialize Flask app
app = Flask(__name__)

# Enable Prometheus metrics
metrics = PrometheusMetrics(app)
metrics.info("app_info", "ARIMA Prediction API", version="1.0.0")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        steps = int(request.form.get("steps", 0))
        if steps <= 0:
            return render_template('index.html', error="'steps' must be a positive integer.")
        
        # Make predictions
        forecast = arima_model.forecast(steps=steps)
        forecast = [round(val, 2) for val in forecast.tolist()]  # Round to 2 decimals
        
        return render_template('index.html', steps=steps, forecast=forecast)
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
