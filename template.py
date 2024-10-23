import os

directories = [
    ".github/workflows",   
    "data/raw",            # Raw data
    "data/processed",      # Processed data
    "models",              # Trained models
    "scripts",             # Python scripts for data fetching, training, etc.
    "app",                 # Streamlit web app
    ".dvc",                # DVC configuration folder
    "mlflow"               # MLflow experiment tracking folder (if using)
]

files = {
    ".github/workflows/ci.yml": "# CI/CD Pipeline for Air Quality Prediction",
    "scripts/data_fetch.py": "# Script to fetch data from OpenWeather API",
    "scripts/feature_generation.py": "# Script for feature engineering",
    "scripts/model_train.py": "# Model training script",
    "scripts/model_predict.py": "# Script to make predictions using the model",
    "app/streamlit_app.py": "# Streamlit Web App",
    "requirements.txt": "# Python dependencies\n\nscikit-learn\npandas\nrequests\nstreamlit\njoblib\nmlflow\nazureml-sdk\ndvc",
    ".gitignore": "# Git ignore file\n\n__pycache__/\ndata/\nmodels/\n.dvc/\nmlruns/\n",
    "README.md": "# Project overview and instructions",
    "dvc.yaml": "# DVC pipeline definition"
}

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Directory created: {directory}")

for file_path, file_content in files.items():
    with open(file_path, 'w') as file:
        file.write(file_content)
    print(f"File created: {file_path}")

print("Project structure created successfully.")
