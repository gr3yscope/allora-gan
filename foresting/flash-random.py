import numpy as np
import sqlite3
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from app_config import DATABASE_PATH

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Fetch data from the database
def load_data(token_name):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT price FROM prices 
            WHERE token=?
            ORDER BY block_height ASC
        """, (token_name,))
        result = cursor.fetchall()
    return np.array([x[0] for x in result]).reshape(-1, 1)

# Prepare data for Random Forest (modifying the data preparation)
def prepare_data_for_rf(data, look_back, prediction_horizon):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, Y = [], []
    for i in range(len(scaled_data) - look_back - prediction_horizon):
        X.append(scaled_data[i:(i + look_back), 0])
        Y.append(scaled_data[i + look_back + prediction_horizon - 1, 0])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, scaler

# Create and train Random Forest model with hyperparameter tuning
def train_and_save_rf_model(token_name, look_back, prediction_horizon):
    try:
        print(f"Training Random Forest model for {token_name} with a {prediction_horizon}-minute horizon.")
        
        data = load_data(token_name)
        X, Y, scaler = prepare_data_for_rf(data, look_back, prediction_horizon)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
        
        # Random Forest model with hyperparameter tuning
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),  # Ensure scaling is done in the pipeline
            ('rf', RandomForestRegressor(random_state=42))
        ])

        # Hyperparameter tuning using RandomizedSearchCV
        param_dist = {
            'rf__n_estimators': [100, 200, 300, 400],
            'rf__max_depth': [None, 10, 20, 30],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__bootstrap': [True, False]
        }

        # RandomizedSearchCV to find the best parameters
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=30, cv=3, scoring='neg_mean_squared_error', n_jobs=3, random_state=42, verbose=2)
        random_search.fit(X_train, Y_train)

        best_model = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")

        # Making predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

        # Save the Random Forest model in .pkl format
        model_path = f'models/{token_name.lower()}_rf_model_{prediction_horizon}m.pkl'
        joblib.dump(best_model, model_path)

        # Save the scaler in .pkl format
        scaler_path = f'models/{token_name.lower()}_scaler_{prediction_horizon}m.pkl'
        joblib.dump(scaler, scaler_path)

        print(f"Model and scaler for {token_name} ({prediction_horizon}-minute prediction) saved to {model_path} and {scaler_path}")
    
    except Exception as e:
        print(f"Error occurred while training Random Forest model for {token_name}: {e}")

# General training function to train Random Forest and save as .pkl
def train_and_save_model(token_name, look_back, prediction_horizon):
    train_and_save_rf_model(token_name, look_back, prediction_horizon)

# Define different time horizons for model training
time_horizons = {
    '10m': (10, 10),  # LOOK_BACK=10, PREDICTION_HORIZON=10
    '20m': (10, 20),  # LOOK_BACK=10, PREDICTION_HORIZON=20
}

# Training for each token and time horizon
for token in ['ETH', 'ARB', 'BTC', 'SOL', 'BNB']:
    for horizon_name, (look_back, prediction_horizon) in time_horizons.items():
        train_and_save_model(f"{token}USD".lower(), look_back, prediction_horizon)




----------------------------------------------------------------------------

import logging
logging.basicConfig(level=logging.INFO)
from flask import Flask, Response
import sqlite3
import os
import numpy as np
import joblib
import functools
import json  # Needed for manually creating JSON responses
from app_config import DATABASE_PATH

app = Flask(__name__)

# Constants from environment variables
API_PORT = int(os.environ.get('API_PORT', 8000))
LOOK_BACK = int(os.environ.get('LOOK_BACK', 10))  # Default to 10 if not set
PREDICTION_STEPS = int(os.environ.get('PREDICTION_STEPS', 10))  # Default to 10 if not set

# HTTP Response Codes
HTTP_RESPONSE_CODE_200 = 200
HTTP_RESPONSE_CODE_404 = 404
HTTP_RESPONSE_CODE_500 = 500

# Load Random Forest model and scaler
def load_model_and_scaler(token_name, prediction_horizon):
    model_path = f'app/models/{token_name.lower()}_rf_model_{prediction_horizon}m.pkl'
    scaler_path = f'app/models/{token_name.lower()}_scaler_{prediction_horizon}m.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Cache predictions to improve performance
@functools.lru_cache(maxsize=128)
def cached_prediction(token_name, prediction_horizon):
    model, scaler = load_model_and_scaler(token_name, prediction_horizon)
    
    if model is None or scaler is None:
        return None
    
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT price FROM prices 
            WHERE token=?
            ORDER BY block_height DESC 
            LIMIT ?
        """, (token_name, LOOK_BACK))
        result = cursor.fetchall()
    
    if not result or len(result) == 0:
        return None
    
    # Reverse the result to chronological order
    prices = np.array([x[0] for x in reversed(result)]).reshape(-1, 1)
    
    # Preprocess data
    scaled_data = scaler.transform(prices)
    
    # Prepare the data in the format expected by the Random Forest model
    recent_data = scaled_data.reshape(1, -1)  # Reshape to 2D for Random Forest (1 sample, many features)
    
    # Make a single prediction for the next step
    pred = model.predict(recent_data)
    
    # Inverse scaling to get actual price
    prediction = scaler.inverse_transform(pred.reshape(-1, 1))
    
    return prediction[0][0]

@app.route('/', methods=['GET'])
async def health():
    return "Hello, World, I'm alive!"

@app.route('/inference/<token>', methods=['GET'])
async def get_inference(token):
    logging.info(f"Received inference request for {token}")
    if not token:
        response = json.dumps({"error": "Token is required"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = f"{token}USD".lower()

    try:
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, cached_prediction, token_name, PREDICTION_STEPS)

        if prediction is None:
            response = json.dumps({"error": "No data found or model unavailable for the specified token"})
            return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')

        print(f"{token} inference: {prediction}")
        return Response(str(prediction), status=HTTP_RESPONSE_CODE_200, mimetype='text/plain')

    except Exception as e:
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/truth/<token>/<block_height>', methods=['GET'])
async def get_price(token, block_height):
    # Directly interact with SQLite database
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT block_height, price 
            FROM prices 
            WHERE token=? AND block_height <= ? 
            ORDER BY ABS(block_height - ?) 
            LIMIT 1
        """, (token.lower(), block_height, block_height))
        result = cursor.fetchone()

    if result:
        # Only return the price in the "body" field as per your desired format
        response = json.dumps(result[1])  # This will only return the price, not the entire object
        return Response(response, status=HTTP_RESPONSE_CODE_200, mimetype='application/json')
    else:
        response = json.dumps({'error': 'No price data found for the specified token and block_height'})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=API_PORT)







