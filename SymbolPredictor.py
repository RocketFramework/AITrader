import json
import time
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from FetchCompanies import rank_companies
from ModelPlotter import ModelPlotter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from BackTest import BackTest
from sklearn.preprocessing import MinMaxScaler


# Function to calculate RSI - Relative Strength Index
def calculate_rsi(data, window=14):
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

def generate_signals(data):
    # Define RSI thresholds
    buy_threshold = 30
    sell_threshold = 70
    buffer_zone = 2  # Number of periods to wait after crossing RSI thresholds

    # Generate initial signals based on RSI
    data['Signal'] = data['RSI'].apply(
        lambda x: 1 if x < buy_threshold else (-1 if x > sell_threshold else 0)
    )
    
    # Implement a buffer zone to prevent immediate signal reversals
    data['Signal'] = data['Signal'].shift(1)  # Shift signals to avoid instant reversal

    # Optional: Add a confirmation rule using moving averages (e.g., 50-period MA)
    data['Short_MA'] = data['close'].rolling(window=50).mean()
    data['Long_MA'] = data['close'].rolling(window=200).mean()

    # Update signal based on trend confirmation (only allow buys in uptrend and sells in downtrend)
    data['Signal'] = data.apply(
        lambda row: 1 if row['Signal'] == 1 and row['close'] > row['Short_MA'] else
        (-1 if row['Signal'] == -1 and row['close'] < row['Short_MA'] else 0),
        axis=1
    )

    return data

# Modified get_stock_data function
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
            data = generate_signals(data) #data['RSI'].apply(lambda x: 1 if x < 35 else (-1 if x > 65 else 0))
            
            return data
        else:
            print(f"No data returned for {symbol} on {exchange}.")
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")


def train_best_model(X, y, class_weight=None):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Initialize RandomForestClassifier with class_weight
    model = RandomForestClassifier(class_weight=class_weight, random_state=42)
    
    # Use RandomizedSearchCV for faster tuning
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings sampled
        cv=3,
        random_state=42
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Best Model Accuracy: {accuracy}")
    #print(classification_report(y_test, y_pred))

    return best_model

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

            # Check if X and y are not empty
            if X.empty or y.empty:
                print(f"Data for {symbol} does not contain enough valid entries.")
                continue  # Skip this symbol if data is insufficient
            
            # Train the best model (use class weights)
            best_model = train_best_model(X, y, class_weight="balanced")
            backtester = BackTest(data, best_model)
            results = backtester.backtest_strategy(data, best_model)

            # Generate predictions for visualization
            predictions = best_model.predict(X)
            ploter.plot_predictions(data.loc[X.index], predictions, symbol)
            
            # Check the latest value for buying opportunity
            latest_record = data.tail(1)
            if not latest_record.empty:
                latest_input = latest_record[['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line', 
                                              'MACD_hist', 'Rel_Volume', 'Price_Change_Pct', 
                                              'Bullish_Engulfing', 'Bearish_Engulfing']].dropna()
                if not latest_input.empty:
                    predicted_signal = best_model.predict(latest_input)[0]                    
                    if (predicted_signal == 1) and company_bought:
                        print(f"BUY signal for {symbol} on {latest_record.index[0]}.")
                    elif (predicted_signal == 1):
                        print(f"**BUY signal for {symbol} on {latest_record.index[0]}.")
                    elif (predicted_signal == -1) and company_bought:
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
