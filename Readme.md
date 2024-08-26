# Airbnb Price Prediction API

This Flask application provides an API endpoint to obtain prediction results from a trained Random Forest model. The model predicts prices based on an Airbnb dataset.

## Description

The application exposes a REST API endpoint `/predict` that accepts a CSV file containing the data. The data is cleaned, processed, and then passed to the Random Forest model to generate price predictions.

## Setup Instructions

To set up the environment and run the application, follow these steps:

1. **Ensure you have Conda installed**: This application requires the Conda Python interpreter.

2. **Create and activate the Conda environment**:
    ```bash
    conda create --name airbnb-prediction python=3.8
    conda activate airbnb-prediction
    ```

3. **Install the required libraries**:
    ```bash
    conda install pandas numpy matplotlib seaborn joblib scikit-learn xgboost
    ```

4. **Run the Flask application**:
    ```bash
    python Predict.py
    ```

## Using the API

To get predictions from the API:

1. **Send a POST request to `/predict`** with a CSV file containing the data.

   Example using `curl`:
   ```bash
   curl -X POST -F "file=@path/to/your/file.csv" http://<your-server-address>:<port>/predict

2. Response: The response will be a JSON object with the key predictions, containing the predicted prices.

## Error Handling

- **400 Bad Request**: Indicates missing file, invalid CSV format, or data encoding issues.
- **500 Internal Server Error**: Indicates issues with making predictions.



