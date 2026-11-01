# Real-Time Motor Health Assessment and Fault Prediction Using IoT and Machine Learning

## Project Overview
This project aims to assess the health condition of a DC motor using vibration and temperature parameters with Machine Learning.

The system uses a trained ML model to classify motor health into five categories:
- Normal
- Watch
- Warning
- Critical
- Failure

A Streamlit dashboard allows users to enter sensor values, view predictions, monitor readings, and download prediction history.

## Technologies Used
- Python
- Machine Learning
- Scikit-learn
- Pandas and NumPy
- Streamlit
- Matplotlib
- Joblib

## Project Structure
- `app.py` – Streamlit dashboard
- `train_model.py` – Model training
- `predict_motor.py` – Motor health prediction
- `evaluate_model.py` – Model evaluation
- `inspect_dataset.py` – Dataset inspection
- `predicta_dataset_v2.csv` – Dataset
- `motor_model.joblib` – Trained ML model
- `requirements.txt` – Required Python libraries

## Installation and Execution

### 1. Clone the Repository
```bash
git clone https://github.com/koteswarithummalapudi/Motor_Predictive_Maintenance.git
```

### 2. Open the Project Folder
```bash
cd Motor_Predictive_Maintenance
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
python -m streamlit run app.py
```

## Current Implementation
The current version demonstrates motor health prediction using manually entered vibration and temperature values through a Streamlit dashboard.

Hardware sensor integration with ESP32 and real-time data collection are planned as the next development stage.

## Project Objective
To develop a low-cost motor health monitoring prototype that can identify abnormal operating conditions and support predictive maintenance.