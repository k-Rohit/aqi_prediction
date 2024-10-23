import os
import json
import pandas as pd
from datetime import datetime, timedelta

RAW_DATA_FOLDER = "data/raw"
PROCESSED_DATA_FILE = "data/processed/processed_data.csv"
IST_OFFSET = timedelta(hours=5, minutes=30)

def load_data():
    """
    Load all JSON files from the raw data folder and combine them into a single DataFrame.
    :return: Pandas DataFrame with the combined AQI data.
    """
    # Create a list to collect rows of data
    data = []

    # Loop through each file in the raw data folder
    for file_name in os.listdir(RAW_DATA_FOLDER):
        if file_name.endswith('.json'):
            file_path = os.path.join(RAW_DATA_FOLDER, file_name)
            with open(file_path, 'r') as file:
                json_data = json.load(file)

                # Extracting pollutants values from each entry in the list
                for entry in json_data['list']:
                    timestamp = entry['dt']
                    # Convert the timestamp from UTC to IST
                    date_utc = datetime.utcfromtimestamp(timestamp)
                    date_ist = date_utc + IST_OFFSET  # Add IST offset to UTC time
                    date = date_ist.strftime('%Y-%m-%d %H:%M:%S')

                    # Extract pollutants and AQI values
                    co = entry["components"]["co"]
                    no = entry["components"]["no"]
                    no2 = entry["components"]["no2"]
                    o3 = entry["components"]["o3"]
                    so2 = entry["components"]["so2"]
                    pm2_5 = entry["components"]["pm2_5"]
                    pm10 = entry["components"]["pm10"]
                    nh3 = entry["components"]["nh3"]
                    aqi = entry["main"]["aqi"]

                    # Append the row to the list
                    data.append([date, co, no, no2, o3, so2, pm2_5, pm10, nh3, aqi])

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=['Date', 'Co', 'No', 'No2', 'O3', 'So2', 'Pm2_5', 'Pm10', 'NH3', 'AQI'])

    return df

def create_features(df: pd.DataFrame):
    """
    Create additional features from the data, such as day of the week, month, etc.
    :param df: DataFrame containing the AQI data.
    :return: DataFrame with additional features.
    """
    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract date-related features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Hour'] = df['Date'].dt.hour

    return df

def save_data(df):
    """
    Save the processed DataFrame to a CSV file.
    :param df: Processed DataFrame.
    """
    df.drop('Date',axis=1, inplace=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_FILE}")

def main():
    """
    Main function to load the raw data, generate features, and save the processed data.
    """
    # Load the raw AQI data
    print("Loading raw AQI data...")
    df = load_data()

    # Create additional features
    print("Creating additional features...")
    df = create_features(df)

    # Handle missing values (if any) by filling with the mean value
    df.fillna(df.mean(), inplace=True)

    # Save the processed data to disk
    print("Saving processed data...")
    save_data(df)

if __name__ == "__main__":
    main()
