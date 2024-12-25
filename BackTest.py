import pandas as pd
from sklearn.impute import SimpleImputer

class BackTest:
    def __init__(self, data=None, predictions=None, buy_only=False):
        self.data = data
        self.predictions = predictions
        self.buy_only = buy_only
        
    def run_backtest(self):
        print("\nRunning Backtest...")
        self.data['Prediction'] = self.predictions
        self.data['Strategy_Return'] = self.data['Prediction'] * self.data['close'].pct_change()
        cumulative_strategy_return = (1 + self.data['Strategy_Return'].dropna()).cumprod()
        
        # Print backtest results
        print("\nBacktest Complete. Final Strategy Return:", cumulative_strategy_return.iloc[-1])
        return cumulative_strategy_return


    def run_backtest_buy(self):
        """
        Run the backtest. If buy_only is True, consider only buy signals.
        """
        self.data['Predicted_Signal'] = self.predictions

        # Filter for buy signals if buy_only is True
        if self.buy_only:
            self.data = self.data[self.data['Predicted_Signal'] == 1]

        # Calculate daily returns
        self.data['Daily_Return'] = self.data['close'].pct_change()
        self.data['Strategy_Return'] = self.data['Predicted_Signal'] * self.data['Daily_Return']

        # Calculate cumulative returns
        self.data['Cumulative_Return'] = (1 + self.data['Strategy_Return']).cumprod()
        print("\nBacktest Complete. Final Strategy Return:", self.data['Cumulative_Return'].iloc[-1])
        return self.data['Cumulative_Return'].iloc[-1]  # Return final cumulative return
    
    def run_backtest_buy_with_cap(self, cap_percentage=0.02):
        """
        Run the backtest, considering a price cap for buy signals.
        
        :param cap_percentage: Maximum acceptable price increase above the closing price for buy attempts.
        """
        self.data['Predicted_Signal'] = self.predictions

        # Initialize additional columns
        self.data['Daily_Return'] = 0
        self.data['Strategy_Return'] = 0

        # Iterate through the data to apply the cap logic
        for i in range(len(self.data) - 1):  # Exclude the last row
            today = self.data.iloc[i]
            next_day = self.data.iloc[i + 1]

            if today['Predicted_Signal'] == 1:  # Buy signal
                max_buy_price = today['close'] * (1 + cap_percentage)  # Calculate price cap
                if next_day['open'] <= max_buy_price:  # Only consider if opening price is within the cap
                    # Calculate the daily return based on next day's open price
                    daily_return = (next_day['close'] - next_day['open']) / next_day['open']
                    self.data.at[i + 1, 'Daily_Return'] = daily_return
                    self.data.at[i + 1, 'Strategy_Return'] = daily_return  # As we're buying only
                else:
                    print(f"Skipped buy on {next_day.name} - Opening price exceeded cap.")

        # Calculate cumulative returns
        self.data['Cumulative_Return'] = (1 + self.data['Strategy_Return']).cumprod()
        print("\nBacktest Complete. Final Strategy Return:", self.data['Cumulative_Return'].iloc[-1])
        return self.data['Cumulative_Return'].iloc[-1]  # Return final cumulative return

    def backtest_strategy(self, data, model):
        """
        Backtest the trading strategy on historical data using the trained model.
        
        :param data: Historical stock data with features and signals.
        :param model: Trained machine learning model.
        :return: Backtesting results with cumulative returns and strategy evaluation.
        """
        initial_balance = 10000  # Starting balance in USD
        position = 0             # Current position (number of shares held)
        cash = initial_balance   # Available cash
        portfolio_value = []     # Track portfolio value over time
        trades = []              # Store trades (buy/sell actions)

        # Imputer to handle NaN values
        imputer = SimpleImputer(strategy='mean')

        # Required features for model prediction (matching training columns)
        required_columns = ['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line', 
                            'MACD_hist', 'Rel_Volume', 'Price_Change_Pct', 
                            'Bullish_Engulfing', 'Bearish_Engulfing']

        # Drop rows with missing target values
        data = data.dropna(subset=['Signal'])

        # Ensure that only columns with valid features are passed (drop columns with all NaN values)
        features = data[required_columns]
        features = features.dropna(axis=1, how='all')  # Drop columns with all NaN values

        # Apply imputation (imputer should be fit to the training data beforehand)
        features_imputed = imputer.fit_transform(features)

        # Process each row for prediction
        for i in range(len(data) - 1):  # Skip the last row for prediction
            row = data.iloc[i]

            # Extract features in the same order as during training and ensure it's a DataFrame
            feature_data = pd.DataFrame([row[required_columns].values], columns=required_columns)
            
            # Handle NaNs in features before prediction
            feature_data_imputed = imputer.transform(feature_data)

            # Convert the imputed features back to a DataFrame with feature names
            feature_data_imputed_df = pd.DataFrame(feature_data_imputed, columns=required_columns)

            # Make prediction
            prediction = model.predict(feature_data_imputed_df)[0]

            if prediction == 1 and cash >= row['close']:  # BUY signal (check if you can afford)
                position += 1
                cash -= row['close']  # Deduct cash for buying one share
                trades.append((row.name, "BUY", row['close']))
            elif prediction == -1 and position > 0:  # SELL signal
                position -= 1
                cash += row['close']  # Add cash from selling one share
                trades.append((row.name, "SELL", row['close']))

            # Calculate current portfolio value
            current_value = cash + (position * row['close'])
            portfolio_value.append(current_value)

        # Final portfolio value (close any remaining positions)
        final_value = cash + (position * data.iloc[-1]['close'])  # Ensure final position is liquidated
        returns = ((final_value - initial_balance) / initial_balance) * 100

        print(f"Initial Balance: ${initial_balance}")
        print(f"Final Balance: ${final_value:.2f}")
        print(f"Total Returns: {returns:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        
        return {
            "portfolio_value": portfolio_value,
            "trades": trades,
            "final_balance": final_value,
            "returns": returns
        }