
# Environmental Monitoring and Pollution Prediction System

## Project Overview
This project implements an **MLOps pipeline** to monitor environmental data (such as air quality and weather conditions) and predict pollution trends. The system utilizes **DVC (Data Version Control)** for managing data, **MLflow** for tracking and training models, and **Grafana & Prometheus** for monitoring the deployed system.

## Features
- **Data Collection & Versioning**: Fetches real-time weather and pollution data from public APIs and manages it using DVC.
- **Pollution Trend Prediction**: Uses machine learning models (ARIMA, LSTMs) to predict air quality trends.
- **Live Monitoring & Optimization**: Implements real-time monitoring using **Grafana** and **Prometheus**.

## Tasks Breakdown
### Task 1: Managing Environmental Data with DVC
- **Live Data Collection**: Fetches data from APIs such as OpenWeatherMap, AirVisual (IQAir), and EPA AirNow.
- **DVC Integration**: Version control is implemented for collected datasets.
- **Remote Storage Setup**: Configured storage with Google Drive or GitHub.
- **Automated Fetching**: Data fetching script runs at scheduled intervals.
- **Version Updates**: Uses `dvc add`, `dvc commit`, and `dvc push` to update datasets.

✅ **Completed**: Data fetching, version control, automation.

### Task 2: Pollution Trend Prediction with MLflow
- **Preprocessing**: Cleaned and prepared collected environmental data.
- **Model Training**: Implemented ARIMA & LSTMs for AQI prediction.
- **Hyperparameter Tuning**: Optimized models using Grid Search.
- **Experiment Tracking**: Tracked model performance with MLflow.
- **Model Deployment**: Deployed API using Flask/FastAPI.

✅ **Completed**: Model training, optimization, deployment.

### Task 3: Monitoring and Live Testing
- **Real-time Monitoring**: Implemented **Grafana** and **Prometheus** dashboards.
- **Live Data Testing**: Validated model predictions using real-time data.
- **Optimization**: Analyzed and improved pipeline performance.
- 
## 📬 Author

**[Maham Khurram]**  
👨‍💻 GitHub: https://github.com/MahamKhurram2/Environmental-Monitoring-and-Pollution-Prediction-System.git
