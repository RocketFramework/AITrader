import json
import time
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from FetchCompanies import rank_companies
from ModelPlotter import ModelPlotter
from BackTest import BackTest
from sklearn.model_selection import train_test_split

# Function to calculate RSI - Relative Strength Index
def calculate_rsi(data, window=7):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD Histogram
def calculate_macd(data):
    macd = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd, signal, hist

# Function to calculate Relative Volume
def calculate_relative_volume(data, window=20):
    avg_volume = data['volume'].rolling(window=window).mean()
    rel_volume = data['volume'] / avg_volume
    return rel_volume

# Function to calculate Price Change Percentage
def calculate_price_change_pct(data, interval=1):
    price_change_pct = data['close'].pct_change(periods=interval) * 100
    return price_change_pct

# Function to detect candlestick patterns
def detect_candlestick_patterns(data):
    bullish_engulfing = (
        (data['open'].shift(1) > data['close'].shift(1)) &
        (data['close'] > data['open']) &
        (data['close'] > data['open'].shift(1)) &
        (data['open'] < data['close'].shift(1))
    )
    
    bearish_engulfing = (
        (data['open'].shift(1) < data['close'].shift(1)) &
        (data['close'] < data['open']) &
        (data['close'] < data['open'].shift(1)) &
        (data['open'] > data['close'].shift(1))
    )
    
    return bullish_engulfing.astype(int), bearish_engulfing.astype(int)


# Functions for RSI, MACD, Relative Volume, and Price Change remain the same as your original code.
def get_stock_data(symbol, exchange, interval=Interval.in_daily, n_bars=5000):
    tv = TvDatafeed(username="None", password="None")  # Replace with your credentials
    try:
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        if data is not None:
            data.dropna(inplace=True)
            win_size = 5
            
            # Add technical indicators
            data['RSI'] = calculate_rsi(data)
            rolling_close = data['close'].rolling(window=win_size)
            data['SMA'] = rolling_close.mean()
            rolling_std = rolling_close.std()
            data['BB_upper'] = data['SMA'] + 2 * rolling_std
            data['BB_lower'] = data['SMA'] - 2 * rolling_std
            data['MACD'], data['Signal_line'], data['MACD_hist'] = calculate_macd(data)
            data['Rel_Volume'] = calculate_relative_volume(data)
            data['Price_Change_Pct'] = calculate_price_change_pct(data)
            data['Bullish_Engulfing'], data['Bearish_Engulfing'] = detect_candlestick_patterns(data)
            
            # Define a simplistic buy/sell signal
            data['Signal'] = data['RSI'].apply(lambda x: 1 if x < 35 else (-1 if x > 70 else 0))

            
            return data
        else:
            print(f"No data returned for {symbol} on {exchange}.")
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")

# Function to create a neural network model
def create_neural_network(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train a deep learning model
def train_neural_network(X, y):
    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create the neural network
    model = create_neural_network(X_train.shape[1])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, checkpoint]
    )

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model, scaler

if __name__ == "__main__":
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    existing_symbols = {entry["symbol"] for entry in config["symbols"]}
    top_40 = []
    try:
        top_40 = rank_companies(existing_symbols)
        if top_40 is not None:
            print(top_40)
    except Exception as e:
        print(f"Error occurred: {e}")

    interval = getattr(Interval, config["interval"], Interval.in_daily)
    n_bars = config["n_bars"]
    ploter = ModelPlotter()
    important_signal_array = []

    for index, row in top_40.iterrows():
        symbol = row["symbol"]
        exchange = "CSELK"
        company_bought = row["companyExists"]

        # Fetch historical data
        data = get_stock_data(symbol, exchange, interval, n_bars)

        if data is not None and not data.empty:
            # Prepare features and target
            X = data[['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line', 
                      'MACD_hist', 'Rel_Volume', 'Price_Change_Pct', 
                      'Bullish_Engulfing', 'Bearish_Engulfing']].dropna()
            print(f"{index} - Length of the dataset for {symbol} = {len(X)}")
            y = data.loc[X.index, 'Signal']

            if X.empty or y.empty:
                print(f"Data for {symbol} does not contain enough valid entries.")
                continue  # Skip this symbol if data is insufficient

            # Train the neural network
            model, scaler = train_neural_network(X, y)

            # Backtesting
            backtester = BackTest(data, model)
            results = backtester.backtest_strategy(data, model)

            # Generate predictions for visualization
            predictions = model.predict(X)
            ploter.plot_predictions(data.loc[X.index], predictions, symbol)
            
            # Generate predictions for visualization
            latest_record = data.tail(1)
            if not latest_record.empty:
                latest_input = latest_record[['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line',
                                              'MACD_hist', 'Rel_Volume', 'Price_Change_Pct', 
                                              'Bullish_Engulfing', 'Bearish_Engulfing']].dropna()
                if not latest_input.empty:
                    latest_input_scaled = scaler.transform(latest_input)
                    predicted_signal = model.predict(latest_input_scaled)[0]
                    if (predicted_signal == 1) and company_bought:
                        print(f"BUY signal for {symbol} on {latest_record.index[0]}.")
                    elif (predicted_signal == 1):
                        important_signal_array.append(f"**BUY signal for {symbol} on {latest_record.index[0]}.")
                        print(f"**BUY signal for {symbol} on {latest_record.index[0]}.")
                    elif (predicted_signal == -1) and company_bought:
                        important_signal_array.append(f"**SELL signal for {symbol} on {latest_record.index[0]}.")
                        print(f"**SELL signal for {symbol} on {latest_record.index[0]}.")
                    elif predicted_signal == -1:
                        print(f"SELL signal for {symbol} on {latest_record.index[0]}.")
                    else:
                        print(f"NO BUY OR SELL signal for {symbol} on {latest_record.index[0]}.")
                else:
                    print(f"Insufficient data for prediction for {symbol}.")
            else:
                print(f"No recent data available for {symbol}.")
        else:
            print(f"Skipping {symbol} due to insufficient data.")

    print(important_signal_array)
