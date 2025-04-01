from flask import Flask, request, jsonify, render_template
from prometheus_flask_exporter import PrometheusMetrics

# Initialize Flask app
app = Flask(__name__)

# Attach Prometheus Metrics
metrics = PrometheusMetrics(app)

# Expose default metrics and a custom metric
metrics.info("app_info", "Test Flask App with Prometheus Metrics", version="1.0")

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Test Flask App</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                margin-top: 50px;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                color: white;
                background-color: #4CAF50;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Welcome to the Test Flask App!</h1>
        <p>Use the button below to visit the Prometheus Metrics page.</p>
        <a href="/metrics">
            <button>View Metrics</button>
        </a>
    </body>
    </html>
    """

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        payload = request.json
        steps = payload.get("steps", 1)
    else:
        steps = int(request.args.get("steps", 1))

    if steps <= 0:
        return jsonify({"error": "'steps' must be a positive integer."}), 400

    forecast = [7.08] * steps  # Dummy prediction logic
    return jsonify({
        "steps": steps,
        "forecast": forecast
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
