from flask import Flask, request, jsonify, abort
import pandas as pd
import joblib
import io

app = Flask(__name__)

# Load the trained model and other necessary files
model = joblib.load("rf_model.pkl")
feature_names = joblib.load('feature_names.pkl')
neighbourhood_encoder = joblib.load('neighbourhood_encoder.pkl')

# Define data cleaning function
def clean_data(df):
    # Remove unwanted columns
    columns_to_drop = ['name','id','host_name','host_id', 'price', 'neighbourhood_group', 'last_review']  
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    # Read the uploaded file
    file = request.files.get('file')
    if file is None:
        abort(400, description='No file uploaded')
    
    # Convert the file to a DataFrame
    try:
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    except Exception as e:
        abort(400, description='Invalid CSV file')

    # Clean the data
    df = clean_data(df)

    # Apply pd.get_dummies to 'room_type'
    df = pd.get_dummies(df, columns=['room_type'])

    # Apply LabelEncoder to 'neighbourhood'
    try:
        df['neighbourhood'] = neighbourhood_encoder.transform(df['neighbourhood'])
    except Exception as e:
        abort(400, description="Error encoding 'neighbourhood'. Possibly an unseen category.")

    # Ensure DataFrame contains only the expected features
    try:
        df = df[feature_names]
    except KeyError as e:
        abort(400, description="DataFrame missing expected features: " + str(e))

    # Make predictions
    try:
        predictions = model.predict(df)
    except Exception as e:
        abort(500, description="Error making predictions: " + str(e))

    return jsonify({'predictions': predictions.tolist()})

# Custom error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': str(error.description)}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
