import os
import requests
import csv
import time
import schedule
from dotenv import load_dotenv  # For loading environment variables from .env

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("OPENWEATHER_API_KEY")  # Ensure your API key is set in the .env file or environment
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
CITY = os.getenv("CITY", "London")
DATA_DIR = os.getenv("DATA_DIR", "data")
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://localhost:5001/predict")
FETCH_INTERVAL_SECONDS = int(os.getenv("FETCH_INTERVAL_SECONDS", 600))  # Default: Fetch every 10 minutes

# Function to fetch live data from OpenWeather API
def fetch_live_data():
    if not API_KEY:
        print("Error: API_KEY is not set. Please set it as an environment variable.")
        return None

    try:
        params = {
            "q": CITY,
            "appid": API_KEY,
            "units": "metric"  # Get data in Celsius
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            print("Fetched Live Data:", data)
            return data
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None

# Function to validate predictions using Flask API
def validate_predictions():
    live_data = fetch_live_data()
    if live_data:
        payload = {"steps": 5}  # Adjust payload structure if necessary
        try:
            response = requests.post(FLASK_API_URL, json=payload)  # Using POST
            if response.status_code == 200:
                predictions = response.json().get("forecast", [])
                print("Model Predictions:", predictions)
                store_results(live_data, predictions)
            else:
                print(f"Prediction Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error sending prediction request: {e}")


# Function to store results in a CSV file
def store_results(live_data, predictions):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, "results.csv")
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write header if file is empty
        if os.stat(file_path).st_size == 0:
            writer.writerow(["Timestamp", "Live Data", "Predictions"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), live_data, predictions])
        print(f"Results stored in {file_path}")

# Schedule data fetching and validation
def job():
    print("Running scheduled job...")
    validate_predictions()

# Schedule the job to run every specified interval
schedule.every(FETCH_INTERVAL_SECONDS).seconds.do(job)

if __name__ == "__main__":
    print("Starting live data fetcher...")
    job()  # Run the first job immediately
    while True:
        schedule.run_pending()
        time.sleep(1)
