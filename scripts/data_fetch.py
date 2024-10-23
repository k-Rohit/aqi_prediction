# Script to fetch data from OpenWeather API


import requests
import os
import time
import json
from datetime import datetime, timedelta

# Your OpenWeather API Key (replace with your actual key)
API_KEY = "e8775684fb652e1382fcaf077305e749"

# Latitude and Longitude for the location (example: Patna, India)
LATITUDE = "25.5941"
LONGITUDE = "85.1376"

# Base URL for the Air Pollution API
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# Output folder for raw data
OUTPUT_FOLDER = "data/raw"

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def fetch_aqi_data(start, end):
    """
    Fetch AQI data from OpenWeather API for the specified start and end timestamps.
    :param start: Start time in Unix timestamp format
    :param end: End time in Unix timestamp format
    :return: JSON response with AQI data
    """
    url = f"{BASE_URL}?lat={LATITUDE}&lon={LONGITUDE}&start={start}&end={end}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

def save_data(data, day):
    """
    Save the fetched AQI data to a JSON file.
    :param data: JSON data to save
    :param day: The specific day for which data is being saved
    """
    filename = f"{OUTPUT_FOLDER}/aqi_{day}.json"
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data for {day} saved to {filename}")

def main():
    """
    Main function to fetch AQI data for the last 365 days and save it to disk.
    """
    # Get the current timestamp
    end_date = datetime.now()
    
    # Iterate through the last 365 days
    for i in range(365):
        # Get the start and end time for each day (midnight to midnight)
        start_date = end_date - timedelta(days=1)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Fetch AQI data for this day
        print(f"Fetching data for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        data = fetch_aqi_data(start_timestamp, end_timestamp)
        
        # Save the data if it's not None
        if data:
            save_data(data, end_date.strftime('%Y-%m-%d'))
        
        # Move to the previous day
        end_date = start_date
        
        # Sleep to avoid exceeding API rate limits (if any)
        time.sleep(1)

if __name__ == "__main__":
    main()
