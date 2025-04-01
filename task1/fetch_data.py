import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/"
DATA_DIR = "data"

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Fetch weather data
def fetch_weather(city="London"):
    try:
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(BASE_URL + "weather", params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

# Fetch air pollution data
def fetch_pollution(lat, lon):
    try:
        params = {"lat": lat, "lon": lon, "appid": API_KEY}
        response = requests.get(BASE_URL + "air_pollution", params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "aqi": data["list"][0]["main"]["aqi"],  # Air Quality Index
            "pm2_5": data["list"][0]["components"]["pm2_5"],  # PM2.5 concentration
            "pm10": data["list"][0]["components"]["pm10"],  # PM10 concentration
        }
    except Exception as e:
        print(f"Error fetching pollution data: {e}")
        return None

def save_to_csv(data, file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.DataFrame([data])
    # Append to CSV or create a new file if it doesn't exist
    df.to_csv(file_path, index=False, mode="a", header=not os.path.exists(file_path))

if __name__ == "__main__":
    # Example usage: Fetching data for London
    city = "London"
    latitude = 51.5074
    longitude = -0.1278

    # Fetch weather data
    weather_data = fetch_weather(city)
    if weather_data:
        save_to_csv(weather_data, "weather.csv")

    # Fetch air pollution data
    pollution_data = fetch_pollution(latitude, longitude)
    if pollution_data:
        save_to_csv(pollution_data, "pollution.csv")

    print("Data fetched and saved successfully!")
