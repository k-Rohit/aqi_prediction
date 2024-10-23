# # Streamlit Web App

# import streamlit as st
# import pandas as pd
# import pickle
# import numpy as np

# # Load the trained model and scaler
# MODEL_FILE = "../models/aqi_model.pkl"
# SCALER_FILE = "../models/scaler.pkl"

# # Load the model
# with open(MODEL_FILE, 'rb') as file:
#     model = pickle.load(file)

# # Load the scaler
# with open(SCALER_FILE, 'rb') as file:
#     scaler = pickle.load(file)

# # Function to make predictions
# def predict_aqi(features):
#     # Scale the features
#     features_scaled = scaler.transform(np.array(features).reshape(1, -1))
#     # Make prediction
#     prediction = model.predict(features_scaled)
#     return prediction[0]

# # Streamlit app layout
# st.title("Air Quality Index Prediction")
# st.write("Enter the following features to get AQI prediction:")

# # Input fields for features
# co = st.number_input("CO (in µg/m³)", min_value=0.0)
# no = st.number_input("NO (in µg/m³)", min_value=0.0)
# no2 = st.number_input("NO2 (in µg/m³)", min_value=0.0)
# o3 = st.number_input("O3 (in µg/m³)", min_value=0.0)
# so2 = st.number_input("SO2 (in µg/m³)", min_value=0.0)
# pm2_5 = st.number_input("PM2.5 (in µg/m³)", min_value=0.0)
# pm10 = st.number_input("PM10 (in µg/m³)", min_value=0.0)
# nh3 = st.number_input("NH3 (in µg/m³)", min_value=0.0)

# # Button to predict AQI
# if st.button("Predict AQI"):
#     features = [co, no, no2, o3, so2, pm2_5, pm10, nh3]
#     prediction = predict_aqi(features)
#     st.success(f"The predicted AQI is: {prediction:.2f}")



import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests


from datetime import datetime, timedelta

# Assuming you have already defined the IST_OFFSET and fetch_current_aqi function
IST_OFFSET = timedelta(hours=5, minutes=30)

# Load the trained model and scaler
MODEL_FILE = "../models/aqi_model.pkl"
SCALER_FILE = "../models/scaler.pkl"

# Load the model
with open(MODEL_FILE, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open(SCALER_FILE, 'rb') as file:
    scaler = pickle.load(file)

# Function to fetch current AQI data from OpenWeather API
def fetch_current_aqi(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        components = data['list'][0]['components']
        aqi = data['list'][0]['main']['aqi']
        dt = data['list'][0]['dt']
        return {
            "Date": dt,
            "co": components["co"],
            "no": components["no"],
            "no2": components["no2"],
            "o3": components["o3"],
            "so2": components["so2"],
            "pm2_5": components["pm2_5"],
            "pm10": components["pm10"],
            "nh3": components["nh3"],
            "aqi": aqi
        }
    else:
        st.error("Error fetching data from OpenWeather API.")
        return None

# Function to make predictions
def predict_aqi(features):
    # Scale the features
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    # Make prediction
    prediction = model.predict(features_scaled)
    return prediction[0]

# Streamlit app layout
st.title("Air Quality Index Prediction")

# User inputs for API key and location
api_key = st.text_input("Enter your OpenWeather API Key:", type="password")
lat = st.number_input("Enter Latitude:", format="%.6f", value=28.6139)  # Default: Delhi
lon = st.number_input("Enter Longitude:", format="%.6f", value=77.2090)  # Default: Delhi

# Button to fetch current AQI data
# Fetch current AQI and predict
if st.button("Fetch Current AQI Data and Predict"):
    if api_key:
        current_data = fetch_current_aqi(api_key, lat, lon)
        if current_data:
            # Convert the current timestamp from UTC to IST
            date_utc = datetime.utcfromtimestamp(current_data['Date'])
            date_ist = date_utc + IST_OFFSET  # Add IST offset to UTC time
            date = date_ist.strftime('%Y-%m-%d %H:%M:%S')

            # Prepare additional features from the date
            day_of_week = date_ist.weekday()  # 0=Monday, 6=Sunday
            month = date_ist.month
            hour = date_ist.hour

            st.write("Current AQI Data:")
            st.json(current_data)

            # Prepare features for prediction
            features = [
                current_data['co'],
                current_data['no'],
                current_data['no2'],
                current_data['o3'],
                current_data['so2'],
                current_data['pm2_5'],
                current_data['pm10'],
                current_data['nh3'],
                day_of_week,
                month,
                hour
            ]
            
            # Make prediction
            prediction = predict_aqi(features)
            st.success(f"The predicted AQI is: {prediction:.2f}")
        else:
            st.error("Failed to fetch current AQI data.")
    else:
        st.warning("Please enter a valid OpenWeather API Key.")