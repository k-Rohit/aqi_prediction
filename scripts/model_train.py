import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

# File paths
PROCESSED_DATA_FILE = "data/processed/processed_data.csv"
MODEL_FILE = "models/aqi_model.pkl"
SCALER_FILE = "models/scaler.pkl"  # To save the scaler

def load_processed_data():
    """
    Load the processed AQI data from the CSV file.
    :return: Pandas DataFrame with processed AQI data.
    """
    df = pd.read_csv(PROCESSED_DATA_FILE)
    return df

def scale_features(X_train, X_test):
    """
    Scale the features using StandardScaler.
    :param X_train: Training features.
    :param X_test: Testing features.
    :return: Scaled training and testing features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler to a file
    with open(SCALER_FILE, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"Scaler saved to {SCALER_FILE}")
    
    return X_train_scaled, X_test_scaled

def train_model(df):
    """
    Train a machine learning model to predict AQI.
    :param df: Processed DataFrame.
    :return: Trained model and test data for evaluation.
    """
    # Define features and target variable
    X = df.drop(columns=['AQI'])  # Features (all columns except AQI)
    y = df['AQI']  # Target (AQI)
    
    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Initialize and train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict AQI values for the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation:\nMean Squared Error: {mse:.2f}\nR² Score: {r2:.2f}")
    
    return model

def save_model(model):
    """
    Save the trained model to a file using pickle.
    :param model: Trained model.
    """
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {MODEL_FILE}")

def main():
    """
    Main function to load data, train the model, and save the trained model.
    """
    print("Loading processed AQI data...")
    df = load_processed_data()

    print("Training the model...")
    model = train_model(df)
    
    print("Saving the trained model...")
    save_model(model)

if __name__ == "__main__":
    main()
