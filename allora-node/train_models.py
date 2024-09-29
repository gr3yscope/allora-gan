from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import joblib
import sqlite3
import numpy as np
import os
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

# R² Score function
def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# Prepare data for CNN + LSTM
def prepare_data_for_cnn_lstm(data, look_back, prediction_horizon):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, Y = [], []
    for i in range(len(scaled_data) - look_back - prediction_horizon):
        X.append(scaled_data[i:(i + look_back), 0])
        Y.append(scaled_data[i + look_back + prediction_horizon - 1, 0])
    X = np.array(X)
    Y = np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for CNN + LSTM
    return X, Y, scaler

# Create CNN + LSTM model
def create_cnn_lstm_model(look_back, dropout_rate=0.3):
    model = Sequential()
    model.add(Input(shape=(look_back, 1)))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1))
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-5), loss='mean_squared_error', metrics=[r2_score])
    return model

# Generalized training function
def train_and_save_model(token_name, look_back, prediction_horizon):
    try:
        print(f"Training model for {token_name} with a {prediction_horizon}-minute horizon.")
        
        data = load_data(token_name)
        X, Y, scaler = prepare_data_for_cnn_lstm(data, look_back, prediction_horizon)
        
        model = create_cnn_lstm_model(look_back)
        
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model_path = f'models/{token_name.lower()}_model_{prediction_horizon}m.keras'
        checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, mode='min')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

        model.fit(X, Y, epochs=12, batch_size=2, validation_split=0.2, verbose=2, callbacks=[reduce_lr, early_stopping, checkpoint, tensorboard])
        
        scaler_path = f'models/{token_name.lower()}_scaler_{prediction_horizon}m.pkl'
        joblib.dump(scaler, scaler_path)
        
        print(f"Model and scaler for {token_name} ({prediction_horizon}-minute prediction) saved to {model_path} and {scaler_path}")
    
    except Exception as e:
        print(f"Error occurred while training model for {token_name}: {e}")

# Train for different time horizons (without 24h horizon)
time_horizons = {
    '10m': (10, 10),    # LOOK_BACK=10, PREDICTION_HORIZON=10
    '20m': (10, 20),    # LOOK_BACK=10, PREDICTION_HORIZON=20
}

for token in ['ETH', 'ARB', 'BTC', 'SOL', 'BNB']:
    for horizon_name, (look_back, prediction_horizon) in time_horizons.items():
        train_and_save_model(f"{token}USD".lower(), look_back, prediction_horizon)
