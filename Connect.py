from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from BackTest import BackTest
from ModelPlotter import ModelPlotter

# Function to calculate RSI - Relative Strength Index
def calculate_rsi(data, window=14):
    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_stock_data(symbol, exchange, interval=Interval.in_daily, n_bars=5000):
    tv = TvDatafeed(username="champikanirosh@gmail.com", password="Basilufucker@20231217")  # Replace with environment variables

    try:
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        if data is not None:
            data.dropna(inplace=True)

            # Add technical indicators
            n_bars = len(data)           
            win_size = int(min(2900, len(data))/2)
            
            data['SMA'] = data['close'].rolling(window=win_size).mean()
            data['RSI'] = calculate_rsi(data)

            # Define a simplistic buy/sell signal
            data['Signal'] = data['RSI'].apply(lambda x: 1 if x < 30 else (-1 if x > 70 else 0))
            return data
        else:
            print("No data returned. Check the symbol or exchange.")
    except Exception as e:
        print(f"Error fetching stock data: {e}")

def prepare_lstm_data(data, target_col, window_size=2900):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[[target_col]].values)

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size, 0])
        y.append(data['Signal'].iloc[i + window_size])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results[model_name] = {
            "accuracy": accuracy,
            "classification_report": report,
            "y_pred": y_pred
        }
    return results

if __name__ == "__main__":
    # Step 1: Fetch historical data
    data = get_stock_data(symbol="JKH.N0000", exchange="CSELK")

    if data is not None:
        # Step 2: Prepare features and target for traditional models
        X = data[['SMA', 'RSI']].dropna()
        y = data.loc[X.index, 'Signal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 3: Define traditional models
        models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Support Vector Machine": SVC()
        }

        # Step 4: Train and evaluate traditional models
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, models)
        
        # Step 5: Print summary of results
        print("\nModel Comparison Results:")
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"Accuracy: {result['accuracy']}")
            print(result['classification_report'])

        # Step 6: Identify the best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        print(f"Best Model is: {best_model_name}")
        best_model = models[best_model_name]

        # Step 7: Backtest on the test dataset
        best_model_predictions = results[best_model_name]['y_pred']
        backtester = BackTest(data.loc[X_test.index], best_model_predictions, buy_only=True)
        cumulative_returns = backtester.run_backtest_buy_with_cap()

        # Step 8: Check if the most recent closing value is a buying opportunity
        print("\nChecking if the latest closing value is a buying opportunity...")

        # Use the most recent record
        latest_record = data.tail(1)  # Fetch the most recent data

        if not latest_record.empty:
            # Prepare the data for prediction
            latest_input = latest_record[['SMA', 'RSI']].dropna()

            if not latest_input.empty:
                # Predict using the best model
                predicted_signal = best_model.predict(latest_input)[0]  # Get the prediction for the latest record

                if predicted_signal == 1:
                    print(f"The latest closing value on {latest_record.index[0]} indicates a BUYING opportunity.")
                else:
                    print(f"The latest closing value on {latest_record.index[0]} does NOT indicate a buying opportunity.")
            else:
                print("The most recent record does not have sufficient data (SMA/RSI) for prediction.")
        else:
            print("No recent data available to evaluate.")

        # Step 9: Plot results
        plotter = ModelPlotter()
        plotter.plot_accuracies(results)
        plotter.plot_predictions(data, best_model_predictions)

