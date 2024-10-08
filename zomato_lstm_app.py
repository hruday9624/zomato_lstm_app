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
st.title("Stock Price Prediction using LSTM")
st.write("This app predicts stock prices using a Bidirectional LSTM model with confidence intervals. Simply enter the company name and select the forecast period.")

# User Input for Company Name and Forecast Period
company_name = st.text_input("Enter the company name:")
forecast_period = st.selectbox("Select forecast period (days):", [7, 14, 30])

# Mapping company names to ticker symbols (example mapping)
company_to_symbol = {
    "Zomato": "ZOMATO.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Google": "GOOGL"
}

# Get the ticker symbol for the entered company name
ticker_symbol = company_to_symbol.get(company_name, None)

if ticker_symbol:
    # Step 1: Import Data Using yfinance
    data = yf.download(ticker_symbol, start="2021-01-01", end="2024-10-08")

    # Step 2: Data Preprocessing
    # Focusing on 'Close' price for time series forecasting
    data = data[['Close']]
    data = data.dropna()  # Drop any rows with missing values

    # Plot the complete historical data
    if not data.empty:
        st.write(f"Historical Data for {company_name}:")
        fig_hist, ax_hist = plt.subplots(figsize=(14, 5))
        ax_hist.plot(data.index, data['Close'], label='Historical Close Price')
        ax_hist.set_xlabel('Date')
        ax_hist.set_ylabel('Close Price')
        ax_hist.set_title(f'Historical Stock Price for {company_name}')
        ax_hist.legend()
        st.pyplot(fig_hist)

    # Calculate a 30-day moving average and overlay on the historical data plot
    data['30 Day MA'] = data['Close'].rolling(window=30).mean()
    fig_hist, ax_hist = plt.subplots(figsize=(14, 5))
    ax_hist.plot(data.index, data['Close'], label='Historical Close Price')
    ax_hist.plot(data.index, data['30 Day MA'], label='30 Day Moving Average', color='orange')
    ax_hist.set_xlabel('Date')
    ax_hist.set_ylabel('Close Price')
    ax_hist.set_title(f'Historical Stock Price for {company_name} with Moving Average')
    ax_hist.legend()
    st.pyplot(fig_hist)

    # Dummy code for adding a news feed section (replace with actual API usage)
    st.write(f"Latest News for {company_name}:")
    news_feed = [
        "Company announces new product line.",
        "Stock price surges after earnings report.",
        "Market analysts upgrade rating."
    ]
    for news in news_feed:
        st.write("- " + news)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

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
    ax.set_title(f'{company_name} Stock Price Prediction using Bidirectional LSTM with Confidence Interval')
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

    # Predict for the selected forecast period
    future_predictions = predict_future(model, last_sequence, forecast_period, scaler)

    # Inverse transform to get actual values
    future_predictions_actual = scaler.inverse_transform(future_predictions.reshape(-1, 1))

    # Display future predictions with Streamlit
    st.write(f"Future {forecast_period} Days Prediction:")
    st.write(future_predictions_actual)

    # Plot future predictions
    fig_future, ax_future = plt.subplots(figsize=(14, 5))
    ax_future.plot(range(1, forecast_period + 1), future_predictions_actual, marker='o', label=f'{forecast_period} Days Prediction')
    ax_future.set_xlabel('Days into the Future')
    ax_future.set_ylabel('Predicted Close Price')
    ax_future.set_title(f'Future Stock Price Predictions for {company_name}')
    ax_future.legend()
    st.pyplot(fig_future)

    # Create a DataFrame from future predictions for download
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_period)
    future_predictions_df = pd.DataFrame(future_predictions_actual, index=future_dates, columns=["Predicted Close Price"])

    # Provide a download button for the predictions
    st.write("Download the Future Predictions as CSV:")
    csv = future_predictions_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name=f'{company_name}_future_predictions.csv',
        mime='text/csv',
    )

else:
    if company_name:
        st.write("Company not found. Please enter a valid company name.")
