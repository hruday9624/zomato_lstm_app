# -*- coding: utf-8 -*-
"""zomato_lstm_app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bQ1s71dnxynljz0GUp0uB1eJFbWHu8mA
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import math

# Streamlit Web App Setup
st.title("Zomato Stock Price Prediction using LSTM")
st.write("This app predicts Zomato stock prices using a Bidirectional LSTM model with confidence intervals.")

# Step 1: Import Data Using yfinance
symbol = "ZOMATO.NS"  # Zomato ticker symbol on NSE
data = yf.download(symbol, start="2021-01-01", end="2024-10-08")

# Step 2: Data Preprocessing
# Focusing on 'Close' price for time series forecasting
data = data[['Close']]
data = data.dropna()  # Drop any rows with missing values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Data Splitting
training_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - training_size
train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :]

# Step 4: Create Sequences
def create_sequences(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 5: Build and Train the LSTM Model
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape=(time_step, 1))))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Add EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Step 6: Evaluate the Model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual values
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform([y_test])

# Calculate Evaluation Metrics
train_rmse = math.sqrt(mean_squared_error(y_train_actual[0], train_predict[:, 0]))
test_rmse = math.sqrt(mean_squared_error(y_test_actual[0], test_predict[:, 0]))
train_mae = mean_absolute_error(y_train_actual[0], train_predict[:, 0])
test_mae = mean_absolute_error(y_test_actual[0], test_predict[:, 0])
train_mape = np.mean(np.abs((y_train_actual[0] - train_predict[:, 0]) / y_train_actual[0])) * 100
test_mape = np.mean(np.abs((y_test_actual[0] - test_predict[:, 0]) / y_test_actual[0])) * 100

st.write(f"Train RMSE: {train_rmse}")
st.write(f"Test RMSE: {test_rmse}")
st.write(f"Train MAE: {train_mae}")
st.write(f"Test MAE: {test_mae}")
st.write(f"Train MAPE: {train_mape}%")
st.write(f"Test MAPE: {test_mape}%")

# Step 7: Calculate Confidence Intervals
# Calculate residuals
test_residuals = y_test_actual[0] - test_predict[:, 0]
std_dev = np.std(test_residuals)
Z = 1.96  # For a 95% confidence level

# Calculate the confidence interval bounds
lower_bound = test_predict[:, 0] - Z * std_dev
upper_bound = test_predict[:, 0] + Z * std_dev

# Plotting with Streamlit
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(data.index[time_step:training_size], y_train_actual[0], label='Train Actual')
ax.plot(data.index[time_step:training_size], train_predict[:, 0], label='Train Prediction')
ax.plot(data.index[training_size + time_step:], y_test_actual[0], label='Test Actual')
ax.plot(data.index[training_size + time_step:], test_predict[:, 0], label='Test Prediction')
ax.fill_between(data.index[training_size + time_step:], lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('Zomato Stock Price Prediction using Bidirectional LSTM with Confidence Interval')
ax.legend()
st.pyplot(fig)

# Step 8: Predict Future Values
def predict_future(model, last_sequence, n_days, scaler):
    future_predictions = []
    current_sequence = last_sequence
    for _ in range(n_days):
        prediction = model.predict(current_sequence.reshape(1, time_step, 1))
        future_predictions.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], prediction, axis=0)
    return np.array(future_predictions)

# Get the last sequence from the test data to predict future values
last_sequence = test_data[-time_step:]

# Predict for 7, 14, and 30 days into the future
future_7_days = predict_future(model, last_sequence, 7, scaler)
future_14_days = predict_future(model, last_sequence, 14, scaler)
future_30_days = predict_future(model, last_sequence, 30, scaler)

# Inverse transform to get actual values
future_7_days_actual = scaler.inverse_transform(future_7_days.reshape(-1, 1))
future_14_days_actual = scaler.inverse_transform(future_14_days.reshape(-1, 1))
future_30_days_actual = scaler.inverse_transform(future_30_days.reshape(-1, 1))

# Display future predictions with Streamlit
st.write("Future 7 Days Prediction:")
st.write(future_7_days_actual)

st.write("Future 14 Days Prediction:")
st.write(future_14_days_actual)

st.write("Future 30 Days Prediction:")
st.write(future_30_days_actual)

# Plot future predictions
fig_future, ax_future = plt.subplots(figsize=(14, 5))
ax_future.plot(range(1, 8), future_7_days_actual, marker='o', label='7 Days Prediction')
ax_future.plot(range(1, 15), future_14_days_actual, marker='o', label='14 Days Prediction')
ax_future.plot(range(1, 31), future_30_days_actual, marker='o', label='30 Days Prediction')
ax_future.set_xlabel('Days into the Future')
ax_future.set_ylabel('Predicted Close Price')
ax_future.set_title('Future Stock Price Predictions for Zomato')
ax_future.legend()
st.pyplot(fig_future)