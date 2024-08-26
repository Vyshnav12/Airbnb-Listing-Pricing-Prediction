# Data Analysis and Model Training

## Data Cleaning and Preprocessing

### Data Cleaning

- The dataset was cleaned to remove any erroneous or irrelevant entries.
- Unnecessary columns were removed to focus on relevant features.

### Outlier Detection

- Outliers were identified and removed using the Tukey test.
- The Tukey test uses the Interquartile Range (IQR) to detect outliers. The formula for outlier detection is:

  \[
  \text{Outlier} \text{ if } (x < Q1 - 1.5 \times IQR) \text{ or } (x > Q3 + 1.5 \times IQR)
  \]

  Where:
  - \( Q1 \) = First quartile (25th percentile)
  - \( Q3 \) = Third quartile (75th percentile)
  - \( IQR \) = Interquartile Range (\( Q3 - Q1 \))

### Custom Plotting Function

- A custom plotting function was created to visualize the data and analyze the results of the machine learning models.

## Machine Learning Models

Three models were trained and evaluated on the cleaned dataset:

1. **Linear Regression**
2. **Random Forest**
3. **XGBoost**

### Evaluation Metrics

- **R2 Score**: Indicates how well the model explains the variance in the data.
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.

- Plots comparing predictions to actual values were created to visualize model performance.

### Best Model

- The **Random Forest** model demonstrated the best performance and was saved as a `.pkl` file using `joblib`. This model is used in the Flask application to make predictions through the API.


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



