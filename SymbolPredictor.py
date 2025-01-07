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

def calculate_rsi_with_signals(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) and generate buy/sell signals.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        period (int): Look-back period for RSI calculation (default=14).

    Returns:
        pd.DataFrame: DataFrame with RSI and buy/sell signals.
    """
    if 'close' not in data.columns:
        raise ValueError("Column 'close' not found in DataFrame.")
    
    # Calculate price differences
    delta = data['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate rolling averages of gains and losses
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    # Handle divide-by-zero issues
    rs = avg_gain / avg_loss.replace(0, np.nan)
    
    # Calculate RSI
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Fill any remaining NaNs with 50 (neutral RSI value)
    data['RSI'] = data['RSI'].fillna(50)
    
    # Generate Buy/Sell Signals
    data['Signal'] = None
    for i in range(1, len(data)):
        # Buy Signal: RSI crosses above 30
        if data['RSI'].iloc[i] > 30 and data['RSI'].iloc[i - 1] <= 30:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: RSI crosses below 70
        elif data['RSI'].iloc[i] < 70 and data['RSI'].iloc[i - 1] >= 70:
            data.loc[data.index[i], 'Signal'] = 'Sell'
            
    data['Signal'] = data['Signal'].fillna('Neutral')
    
    return data[['RSI', 'Signal']]

def calculate_stochastic_with_signals(data, period=14, smoothing_k=3, smoothing_d=3):
    """
    Calculate the Stochastic Oscillator (%K, %D) and generate buy/sell signals.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        period (int): Look-back period for %K (default=14).
        smoothing_k (int): Smoothing period for %K (default=3).
        smoothing_d (int): Smoothing period for %D (default=3).

    Returns:
        pd.DataFrame: DataFrame with %K, %D, and buy/sell signals.
    """
    # Calculate %K
    data['Lowest_Low'] = data['low'].rolling(window=period).min()
    data['Highest_High'] = data['high'].rolling(window=period).max()
    data['%K'] = (
        (data['close'] - data['Lowest_Low']) /
        (data['Highest_High'] - data['Lowest_Low'])
    ) * 100

    # Smooth %K
    data['%K_Smoothed'] = data['%K'].rolling(window=smoothing_k).mean()

    # Calculate %D (Smoothed %K)
    data['%D'] = data['%K_Smoothed'].rolling(window=smoothing_d).mean()

    # Generate Buy/Sell Signals
    data['Signal'] = None
    for i in range(1, len(data)):
        # Buy Signal: %K crosses above %D
        if data['%K_Smoothed'].iloc[i] > data['%D'].iloc[i] and data['%K_Smoothed'].iloc[i - 1] <= data['%D'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: %K crosses below %D
        elif data['%K_Smoothed'].iloc[i] < data['%D'].iloc[i] and data['%K_Smoothed'].iloc[i - 1] >= data['%D'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Sell'

    # Clean up intermediate columns if necessary
    data.drop(['Lowest_Low', 'Highest_High'], axis=1, inplace=True)

    return data[['%K', '%K_Smoothed', '%D', 'Signal']]

def calculate_cci_with_signals(data, period=20):
    """
    Calculate the Commodity Channel Index (CCI) and generate buy/sell signals.

    :param data: A pandas DataFrame containing 'high', 'low', and 'close' prices.
    :param period: The period for calculating CCI (default is 20).
    :return: A pandas DataFrame with CCI values and buy/sell signals.
    """
    # Calculate the Typical Price (TP)
    data['TP'] = (data['high'] + data['low'] + data['close']) / 3

    # Calculate the Simple Moving Average (SMA) of the Typical Price
    data['SMA_TP'] = data['TP'].rolling(window=period).mean()

    # Calculate the Mean Deviation (MD)
    data['MD'] = data['TP'].rolling(window=period).apply(lambda x: abs(x - x.mean()).mean(), raw=False)

    # Calculate the CCI
    data['CCI'] = (data['TP'] - data['SMA_TP']) / (0.015 * data['MD'])

    # Generate Buy/Sell Signals
    data['Signal'] = None
    for i in range(1, len(data)):
        # Check for Buy Signal (CCI crossing above -100)
        if data['CCI'].iloc[i] > -100 and data['CCI'].iloc[i - 1] <= -100:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Check for Sell Signal (CCI crossing below +100)
        elif data['CCI'].iloc[i] < 100 and data['CCI'].iloc[i - 1] >= 100:
            data.loc[data.index[i], 'Signal'] = 'Sell'

    # Drop intermediate columns if necessary
    data.drop(['TP', 'SMA_TP', 'MD'], axis=1, inplace=True)

    return data[['CCI', 'Signal']]

def calculate_adx_with_signals(data, period=14):
    """
    Calculate ADX, +DI, -DI, and generate buy/sell signals.

    :param data: A pandas DataFrame containing 'high', 'low', and 'close' prices.
    :param period: The period for calculating ADX (default is 14).
    :return: A pandas DataFrame with ADX, +DI, -DI, and buy/sell signals.
    """
    # Create a shifted 'close' column for previous close values
    data['prev_close'] = data['close'].shift(1)

    # Calculate True Range (TR)
    data['TR'] = data.apply(
        lambda row: max(
            row['high'] - row['low'],
            abs(row['high'] - row['prev_close']),
            abs(row['low'] - row['prev_close'])
        ),
        axis=1
    )
    
    # Calculate +DM and -DM
    data['+DM'] = data['high'].diff().apply(lambda x: x if x > 0 else 0)
    data['-DM'] = data['low'].diff().apply(lambda x: -x if x < 0 else 0)
    
    # Smooth TR, +DM, and -DM using Wilder's smoothing method
    data['TR_smoothed'] = data['TR'].rolling(window=period).sum()
    data['+DM_smoothed'] = data['+DM'].rolling(window=period).sum()
    data['-DM_smoothed'] = data['-DM'].rolling(window=period).sum()

    # Calculate +DI and -DI
    data['+DI'] = (data['+DM_smoothed'] / data['TR_smoothed']) * 100
    data['-DI'] = (data['-DM_smoothed'] / data['TR_smoothed']) * 100

    # Calculate DX (Directional Index)
    data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])) * 100

    # Calculate ADX (Average of DX over the period)
    data['ADX'] = data['DX'].rolling(window=period).mean()

    # Generate Buy/Sell Signals
    data['Signal'] = None
    data['Signal'] = data.apply(
        lambda row: 'Buy' if row['+DI'] > row['-DI'] and row['ADX'] > 25 else 
                    ('Sell' if row['-DI'] > row['+DI'] and row['ADX'] > 25 else None),
        axis=1
    )

    # Drop intermediate columns if needed
    data.drop(['TR', '+DM', '-DM', 'TR_smoothed', '+DM_smoothed', '-DM_smoothed', 'DX'], axis=1, inplace=True)

    return data[['+DI', '-DI', 'ADX', 'Signal']]

def calculate_awesome_oscillator(data):
    """
    Calculate the Awesome Oscillator (AO) and generate buy/sell signals, with edge case handling.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
    
    Returns:
        pd.DataFrame: DataFrame with AO values and buy/sell signals.
    """
    if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
        raise ValueError("Columns 'high', 'low', and 'close' must be present in the DataFrame.")
    
    # Drop rows with missing high, low, or close values
    data = data.dropna(subset=['high', 'low', 'close'])
    
    if len(data) < 34:
        raise ValueError("Data must contain at least 34 rows for calculation.")
    
    # Calculate the median price
    data['Median_Price'] = (data['high'] + data['low']) / 2
    
    # Calculate the 34-period and 5-period simple moving averages (SMA) of the median price
    data['SMA_34'] = data['Median_Price'].rolling(window=34).mean()
    data['SMA_5'] = data['Median_Price'].rolling(window=5).mean()
    
    # Remove rows where SMAs couldn't be calculated (first 34 rows)
    data = data.dropna(subset=['SMA_34', 'SMA_5'])
    
    # Calculate the Awesome Oscillator (AO)
    data['AO'] = data['SMA_5'] - data['SMA_34']
    
    # Generate Buy/Sell signals based on AO crossing above/below zero
    data['Signal'] = None
    for i in range(1, len(data)):
        # Buy Signal: AO crosses above 0
        if data['AO'].iloc[i] > 0 and data['AO'].iloc[i - 1] <= 0:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: AO crosses below 0
        elif data['AO'].iloc[i] < 0 and data['AO'].iloc[i - 1] >= 0:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    return data[['AO', 'Signal']]

def calculate_momentum(data, period=10):
    """
    Calculate the Momentum (10) and generate buy/sell signals.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        period (int): Look-back period for momentum calculation (default is 10).
    
    Returns:
        pd.DataFrame: DataFrame with Momentum values and buy/sell signals.
    """
    if 'close' not in data.columns:
        raise ValueError("Column 'close' must be present in the DataFrame.")
    
    # Drop rows with missing close values
    data = data.dropna(subset=['close'])
    
    if len(data) < period:
        raise ValueError(f"Data must contain at least {period} rows for calculation.")
    
    # Calculate Momentum
    data['Momentum'] = data['close'] - data['close'].shift(period)
    
    # Remove rows where Momentum couldn't be calculated (first "period" rows)
    data = data.dropna(subset=['Momentum'])
    
    # Generate Buy/Sell signals based on Momentum crossing zero
    data['Signal'] = None
    for i in range(1, len(data)):
        # Buy Signal: Momentum crosses above 0
        if data['Momentum'].iloc[i] > 0 and data['Momentum'].iloc[i - 1] <= 0:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: Momentum crosses below 0
        elif data['Momentum'].iloc[i] < 0 and data['Momentum'].iloc[i - 1] >= 0:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    return data[['Momentum', 'Signal']]

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    """
    Calculate MACD (12, 26) and generate Buy/Sell signals.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        short_period (int): The short EMA period (default is 12).
        long_period (int): The long EMA period (default is 26).
        signal_period (int): The signal line period (default is 9).
    
    Returns:
        pd.DataFrame: DataFrame with MACD, Signal_Line, and Buy/Sell signals.
    """
    if 'close' not in data.columns:
        raise ValueError("Column 'close' must be present in the DataFrame.")
    
    # Drop rows with missing close values
    data = data.dropna(subset=['close'])
    
    if len(data) < long_period:
        raise ValueError(f"Data must contain at least {long_period} rows for MACD calculation.")
    
    # Calculate the MACD line (difference between 12-period and 26-period EMA)
    data['MACD'] = data['close'].ewm(span=short_period, adjust=False).mean() - data['close'].ewm(span=long_period, adjust=False).mean()
    
    # Calculate the signal line (9-period EMA of the MACD line)
    data['Signal_Line'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    # Drop rows where MACD and Signal_Line couldn't be calculated (first "long_period" rows)
    data = data.dropna(subset=['MACD', 'Signal_Line'])
    
    # Generate Buy/Sell signals based on MACD crossing Signal_Line
    data['Signal'] = None
    for i in range(1, len(data)):
        # Buy Signal: MACD crosses above the Signal_Line
        if data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and data['MACD'].iloc[i - 1] <= data['Signal_Line'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: MACD crosses below the Signal_Line
        elif data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] and data['MACD'].iloc[i - 1] >= data['Signal_Line'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    return data[['MACD', 'Signal_Line', 'Signal']]

def calculate_stochastic_rsi(data, rsi_period=14, stoch_k_period=14, stoch_d_period=3, fast_k_period=3, fast_d_period=3):
    """
    Calculate Stochastic RSI Fast (%K, %D) based on RSI values.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        rsi_period (int): The period for calculating RSI (default=14).
        stoch_k_period (int): Look-back period for Stochastic %K (default=14).
        stoch_d_period (int): Look-back period for Stochastic %D (default=3).
        fast_k_period (int): Smoothing period for Fast %K (default=3).
        fast_d_period (int): Smoothing period for Fast %D (default=3).
    
    Returns:
        pd.DataFrame: DataFrame with RSI, Stochastic RSI (%K, %D), and Buy/Sell signals.
    """
    if 'close' not in data.columns:
        raise ValueError("Column 'close' must be present in the DataFrame.")
    
    # Calculate RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi.fillna(50)  # Replace NaNs with neutral value
    
    # Calculate Stochastic RSI (%K)
    data['Lowest_RSI'] = data['RSI'].rolling(window=stoch_k_period).min()
    data['Highest_RSI'] = data['RSI'].rolling(window=stoch_k_period).max()
    data['%K'] = ((data['RSI'] - data['Lowest_RSI']) / (data['Highest_RSI'] - data['Lowest_RSI'])) * 100
    
    # Smooth %K (Fast %K)
    data['%K_Smoothed'] = data['%K'].rolling(window=fast_k_period).mean()
    
    # Calculate %D (Fast %D)
    data['%D'] = data['%K_Smoothed'].rolling(window=fast_d_period).mean()
    
    # Drop rows where Stochastic RSI couldn't be calculated (first "stoch_k_period" rows)
    data = data.dropna(subset=['%K', '%K_Smoothed', '%D'])
    
    # Initialize the 'Signal' column with None values using .loc to avoid SettingWithCopyWarning
    data.loc[:, 'Signal'] = None
    
    # Generate Buy/Sell signals based on %K crossing %D
    for i in range(1, len(data)):
        # Buy Signal: %K crosses above %D
        if data['%K_Smoothed'].iloc[i] > data['%D'].iloc[i] and data['%K_Smoothed'].iloc[i - 1] <= data['%D'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: %K crosses below %D
        elif data['%K_Smoothed'].iloc[i] < data['%D'].iloc[i] and data['%K_Smoothed'].iloc[i - 1] >= data['%D'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    return data[['RSI', '%K', '%K_Smoothed', '%D', 'Signal']]

def calculate_williams_percent_range(data, period=14):
    """
    Calculate Williams Percent Range (WPR) and generate buy/sell signals.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        period (int): The period for calculating Williams Percent Range (default=14).
    
    Returns:
        pd.DataFrame: DataFrame with WPR values and Buy/Sell signals.
    """
    if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
        raise ValueError("Columns 'high', 'low', and 'close' must be present in the DataFrame.")
    
    # Calculate the highest high and lowest low over the period
    data['Highest_High'] = data['high'].rolling(window=period).max()
    data['Lowest_Low'] = data['low'].rolling(window=period).min()
    
    # Calculate Williams Percent Range (WPR)
    data['WPR'] = ((data['Highest_High'] - data['close']) / (data['Highest_High'] - data['Lowest_Low'])) * -100
    
    # Drop rows where WPR couldn't be calculated (first "period" rows)
    data = data.dropna(subset=['WPR'])
    
    # Initialize the 'Signal' column with None values using .loc to avoid SettingWithCopyWarning
    data.loc[:, 'Signal'] = None
    
    # Generate Buy/Sell signals based on WPR
    for i in range(1, len(data)):
        # Buy Signal: WPR crosses above -80 (indicating oversold condition, possible buy)
        if data['WPR'].iloc[i] > -80 and data['WPR'].iloc[i - 1] <= -80:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: WPR crosses below -20 (indicating overbought condition, possible sell)
        elif data['WPR'].iloc[i] < -20 and data['WPR'].iloc[i - 1] >= -20:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    return data[['WPR', 'Signal']]

def calculate_bull_bear_power(data, period=13):
    """
    Calculate Bull Power and Bear Power, and generate buy/sell signals.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        period (int): Period for calculating the EMA (default=13).
    
    Returns:
        pd.DataFrame: DataFrame with Bull Power, Bear Power, and Buy/Sell signals.
    """
    if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
        raise ValueError("Columns 'high', 'low', and 'close' must be present in the DataFrame.")
    
    # Calculate the Exponential Moving Average (EMA) of the close price
    data['EMA'] = data['close'].ewm(span=period, adjust=False).mean()
    
    # Calculate Bull Power and Bear Power
    data['Bull_Power'] = data['high'] - data['EMA']
    data['Bear_Power'] = data['low'] - data['EMA']
    
    # Drop rows where Bull Power and Bear Power couldn't be calculated (first "period" rows)
    data = data.dropna(subset=['Bull_Power', 'Bear_Power'])
    
    # Generate Buy/Sell signals based on Bull and Bear Power
    data['Signal'] = None
    for i in range(1, len(data)):
        # Buy Signal: Bull Power crosses above 0 (suggesting buying power)
        if data['Bull_Power'].iloc[i] > 0 and data['Bull_Power'].iloc[i - 1] <= 0:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: Bear Power crosses below 0 (suggesting selling pressure)
        elif data['Bear_Power'].iloc[i] < 0 and data['Bear_Power'].iloc[i - 1] >= 0:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    return data[['Bull_Power', 'Bear_Power', 'Signal']]

def calculate_ultimate_oscillator(data, short_period=7, medium_period=14, long_period=28):
    """
    Calculate the Ultimate Oscillator (7, 14, 28) and generate buy/sell signals.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        short_period (int): Short period for UO (default=7).
        medium_period (int): Medium period for UO (default=14).
        long_period (int): Long period for UO (default=28).
    
    Returns:
        pd.DataFrame: DataFrame with Ultimate Oscillator and Buy/Sell signals.
    """
    if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
        raise ValueError("Columns 'high', 'low', and 'close' must be present in the DataFrame.")
    
    # Edge case handling for the first row (no previous close)
    data['Previous_Close'] = data['close'].shift(1)
    
    # Fill the NaN values in 'Previous_Close' with the current 'close' value
    data['Previous_Close'].fillna(data['close'], inplace=True)

    # Calculate Buying Pressure (BP) and True Range (TR)
    data['BP'] = data['close'] - np.minimum(data['low'], data['Previous_Close'])
    data['TR'] = np.maximum(data['high'], data['Previous_Close']) - np.minimum(data['low'], data['Previous_Close'])
    
    # Calculate the smoothed BP and TR for different periods
    data['BP_7'] = data['BP'].rolling(window=short_period).sum()
    data['TR_7'] = data['TR'].rolling(window=short_period).sum()
    
    data['BP_14'] = data['BP'].rolling(window=medium_period).sum()
    data['TR_14'] = data['TR'].rolling(window=medium_period).sum()
    
    data['BP_28'] = data['BP'].rolling(window=long_period).sum()
    data['TR_28'] = data['TR'].rolling(window=long_period).sum()
    
    # Calculate the Ultimate Oscillator
    data['UO'] = 100 * (4 * (data['BP_7'] / data['TR_7']) + 2 * (data['BP_14'] / data['TR_14']) + (data['BP_28'] / data['TR_28'])) / (4 + 2 + 1)
    
    # Drop rows with NaN values (first "long_period" rows)
    data = data.dropna(subset=['UO'])
    
    # Generate Buy/Sell signals
    data.loc[:, 'Signal'] = None
    for i in range(1, len(data)):
        # Buy Signal: UO crosses above 50 (suggesting bullish trend)
        if data['UO'].iloc[i] > 50 and data['UO'].iloc[i - 1] <= 50:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: UO crosses below 50 (suggesting bearish trend)
        elif data['UO'].iloc[i] < 50 and data['UO'].iloc[i - 1] >= 50:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    return data[['UO', 'Signal']]

def calculate_ema(data, period=10):
    """
    Calculate the Exponential Moving Average (EMA) with a given period (default 10).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        period (int): Period for the EMA calculation (default=10).
    
    Returns:
        pd.DataFrame: DataFrame with EMA and Buy/Sell signals.
    """
    # Check if 'close' column exists in the data
    if 'close' not in data.columns:
        raise ValueError("Column 'close' must be present in the DataFrame.")

    # Calculate the Exponential Moving Average (EMA)
    data['EMA'] = data['close'].ewm(span=period, adjust=False).mean()

    # Handle edge cases for missing data (NaN) by backfilling
    data['EMA'].fillna(method='bfill', inplace=True)

    # Generate Buy/Sell signals based on EMA crossover logic
    data['Signal'] = None  # Initialize Signal column with None
    for i in range(1, len(data)):
        # Buy Signal: EMA crosses above the previous value (suggesting a bullish trend)
        if data['EMA'].iloc[i] > data['EMA'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: EMA crosses below the previous value (suggesting a bearish trend)
        elif data['EMA'].iloc[i] < data['EMA'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    # Fill any remaining None values in Signal column (no signal)
    data['Signal'].fillna('No Signal', inplace=True)

    return data[['EMA', 'Signal']]

def calculate_sma(data, period=10):
    """
    Calculate the Simple Moving Average (SMA) with a given period (default 10).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        period (int): Period for the SMA calculation (default=10).
    
    Returns:
        pd.DataFrame: DataFrame with SMA and Buy/Sell signals.
    """
    # Check if 'close' column exists in the data
    if 'close' not in data.columns:
        raise ValueError("Column 'close' must be present in the DataFrame.")

    # Calculate the Simple Moving Average (SMA)
    data['SMA'] = data['close'].rolling(window=period, min_periods=1).mean()

    # Handle edge cases for missing data (NaN) by backfilling
    data['SMA'].fillna(method='bfill', inplace=True)

    # Generate Buy/Sell signals based on SMA crossover logic
    data['Signal'] = None  # Initialize Signal column with None
    for i in range(1, len(data)):
        # Buy Signal: SMA crosses above the previous value (suggesting a bullish trend)
        if data['SMA'].iloc[i] > data['SMA'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: SMA crosses below the previous value (suggesting a bearish trend)
        elif data['SMA'].iloc[i] < data['SMA'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Sell'
    
    # Fill any remaining None values in Signal column (no signal)
    data['Signal'].fillna('No Signal', inplace=True)

    return data[['SMA', 'Signal']]

def calculate_sma_20(data):
    return calculate_sma(data, period=20)
    
def calculate_ema_20(data):
    return calculate_ema(data, period=20)

def calculate_ema_30(data):
    return calculate_ema(data, period=30)

def calculate_sma_30(data):
    return calculate_sma(data, period=30)

def calculate_ema_50(data):
    return calculate_ema(data, period=50)

def calculate_sma_50(data):
    return calculate_sma(data, period=50)

def calculate_ema_100(data):
    return calculate_ema(data, period=100)

def calculate_sma_100(data):
    return calculate_sma(data, period=100)

def calculate_ema_200(data):
    return calculate_ema(data, period=200)

def calculate_sma_200(data):
    return calculate_sma(data, period=200)

def calculate_ichimoku_base_line(data, short_period=9, medium_period=26, long_period=52):
    """
    Calculate the Ichimoku Base Line (Kijun-sen) and generate Buy/Sell signals.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'high' and 'low' prices.
        short_period (int): Short period for conversion line (default=9).
        medium_period (int): Medium period for base line (default=26).
        long_period (int): Long period for span B (default=52).
    
    Returns:
        pd.DataFrame: DataFrame with Base Line (Kijun-sen) and Buy/Sell signals.
    """
    # Check if required columns exist
    if 'high' not in data.columns or 'low' not in data.columns:
        raise ValueError("Columns 'high' and 'low' must be present in the DataFrame.")
    
    # Calculate Ichimoku Base Line (Kijun-sen) for the medium period
    data['Base_Line'] = (data['high'].rolling(window=medium_period).max() + 
                         data['low'].rolling(window=medium_period).min()) / 2

    # Handle edge cases: Fill missing Base Line values (NaN) by backfilling
    data['Base_Line'].fillna(method='bfill', inplace=True)

    # Generate Buy/Sell signals based on Base Line crossover logic
    data['Signal'] = None  # Initialize Signal column with None
    for i in range(1, len(data)):
        # Buy Signal: When high price crosses above the Base Line
        if data['high'].iloc[i] > data['Base_Line'].iloc[i] and data['high'].iloc[i - 1] <= data['Base_Line'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: When low price crosses below the Base Line
        elif data['low'].iloc[i] < data['Base_Line'].iloc[i] and data['low'].iloc[i - 1] >= data['Base_Line'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Sell'

    # Fill any remaining None values in Signal column (no signal)
    data['Signal'].fillna('No Signal', inplace=True)

    return data[['Base_Line', 'Signal']]

def calculate_vwma(data, period=20):
    """
    Calculate Volume Weighted Moving Average (VWMA) and generate Buy/Sell signals.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' and 'volume'.
        period (int): Look-back period for VWMA (default=20).
    
    Returns:
        pd.DataFrame: DataFrame with VWMA and Buy/Sell signals.
    """
    # Check if required columns exist
    if 'close' not in data.columns or 'volume' not in data.columns:
        raise ValueError("Columns 'close' and 'volume' must be present in the DataFrame.")

    # Calculate VWMA
    data['VWMA'] = (
        (data['close'] * data['volume']).rolling(window=period).sum() /
        data['volume'].rolling(window=period).sum()
    )

    # Handle edge cases: Fill missing VWMA values (NaN) by backfilling
    data['VWMA'].fillna(method='bfill', inplace=True)

    # Generate Buy/Sell signals based on VWMA crossover logic
    data['Signal'] = None  # Initialize Signal column with None
    for i in range(1, len(data)):
        # Buy Signal: When close price crosses above the VWMA
        if data['close'].iloc[i] > data['VWMA'].iloc[i] and data['close'].iloc[i - 1] <= data['VWMA'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: When close price crosses below the VWMA
        elif data['close'].iloc[i] < data['VWMA'].iloc[i] and data['close'].iloc[i - 1] >= data['VWMA'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Sell'

    # Fill any remaining None values in Signal column (no signal)
    data['Signal'].fillna('No Signal', inplace=True)

    return data[['VWMA', 'Signal']]

def calculate_hma(data, period=9):
    """
    Calculate the Hull Moving Average (HMA) and generate Buy/Sell signals.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        period (int): Look-back period for HMA (default=9).
    
    Returns:
        pd.DataFrame: DataFrame with HMA and Buy/Sell signals.
    """
    # Check if 'close' column exists
    if 'close' not in data.columns:
        raise ValueError("Column 'close' must be present in the DataFrame.")

    # Weighted Moving Average (WMA) function
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(window=length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # Calculate the HMA
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    data['WMA_half'] = wma(data['close'], half_period)
    
    # WMA for full period
    data['WMA_full'] = wma(data['close'], period)
    
    # Weighted difference
    data['WMA_diff'] = (2 * data['WMA_half']) - data['WMA_full']
    
    # HMA (WMA of the weighted difference)
    data['HMA'] = wma(data['WMA_diff'], sqrt_period)
    
    # Handle edge cases: Fill missing HMA values (NaN) by backfilling
    data['HMA'].fillna(method='bfill', inplace=True)
    
    # Generate Buy/Sell signals
    data['Signal'] = None  # Initialize Signal column with None
    for i in range(1, len(data)):
        # Buy Signal: When close price crosses above HMA
        if data['close'].iloc[i] > data['HMA'].iloc[i] and data['close'].iloc[i - 1] <= data['HMA'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        # Sell Signal: When close price crosses below HMA
        elif data['close'].iloc[i] < data['HMA'].iloc[i] and data['close'].iloc[i - 1] >= data['HMA'].iloc[i - 1]:
            data.loc[data.index[i], 'Signal'] = 'Sell'

    # Fill any remaining None values in Signal column (no signal)
    data['Signal'].fillna('No Signal', inplace=True)

    # Clean up intermediate columns
    data.drop(columns=['WMA_half', 'WMA_full', 'WMA_diff'], inplace=True)

    return data[['HMA', 'Signal']]

def calculate_final_signal(data):
    """
    Calculate the final Buy/Sell/Neutral signal based on momentum indicators and moving averages.

    Parameters:
        data (dict): Dictionary with indicator names as keys and their signals as values.

    Returns:
        str: Final signal ("Buy", "Sell", or "Neutral").
    """
    
    # Separate Momentum Indicators and Moving Averages
    momentum_indicators = [
        calculate_rsi_with_signals(data)['Signal'],
        calculate_stochastic_with_signals(data)['Signal'],
        calculate_cci_with_signals(data)['Signal'],
        calculate_adx_with_signals(data)['Signal'],
        calculate_awesome_oscillator(data)['Signal'],
        calculate_momentum(data)['Signal'],
        calculate_macd(data)['Signal'],
        calculate_stochastic_rsi(data)['Signal'],
        calculate_williams_percent_range(data)['Signal'],
        calculate_bull_bear_power(data)['Signal'],
        calculate_ultimate_oscillator(data)['Signal']
    ]
    
    moving_averages = [
        calculate_ema(data)['Signal'],
        calculate_sma(data)['Signal'],
        calculate_sma_20(data)['Signal'],
        calculate_ema_20(data)['Signal'],
        calculate_ema_30(data)['Signal'],
        calculate_sma_30(data)['Signal'],
        calculate_ema_50(data)['Signal'],
        calculate_sma_50(data)['Signal'],
        calculate_ema_100(data)['Signal'],
        calculate_sma_100(data)['Signal'],
        calculate_ema_200(data)['Signal'],
        calculate_sma_200(data)['Signal'],
        calculate_ichimoku_base_line(data)['Signal'],
        calculate_vwma(data)['Signal'],
        calculate_hma(data)['Signal']
    ]
    
    # Initialize counters
    buy_count = 0
    sell_count = 0
    neutral_count = 0
    
    # Count signals for Momentum Indicators
    for signals in momentum_indicators:
        # Ensure 'Signal' contains no None values (replace with 'Neutral')
        signals = signals.fillna('Neutral')
        
        # Count each type of signal
        buy_count += (signals == 'Buy').sum()
        sell_count += (signals == 'Sell').sum()
        neutral_count += (signals == 'Neutral').sum()
    
    # Count signals for Moving Averages
    for signals in moving_averages:
        # Ensure 'Signal' contains no None values (replace with 'Neutral')
        signals = signals.fillna('Neutral')
        
        # Count each type of signal
        buy_count += (signals == 'Buy').sum()
        sell_count += (signals == 'Sell').sum()
        neutral_count += (signals == 'Neutral').sum()
    
    # Add candlestick pattern signals
    if data["Bullish_Engulfing"].iloc[-1]:
        buy_count += 1
    if data["Bearish_Engulfing"].iloc[-1]:
        sell_count += 1
    if data["Hammer"].iloc[-1]:
        buy_count += 1
        
    # Determine final signal
    if (buy_count > sell_count) and (buy_count > neutral_count):
        final_signal = 1
    elif (sell_count > buy_count) and (sell_count > neutral_count):
        final_signal = -1
    else:
        final_signal = 0
    
    print(f"Buy Signals: {buy_count}, Sell Signals: {sell_count}, Neutral Signals: {neutral_count}, Final Signal: {final_signal}")
    
    rsi_data = calculate_rsi_with_signals(data)
    rsi_data['Signal'] = rsi_data['Signal'].replace({
        'Buy': 1,
        'Sell': -1,
        'Neutral': 0
    })

    rsi_data['Signal'] = rsi_data['Signal'].fillna(0)

    return rsi_data['Signal']
    # return {
    #     "Buy Signals": buy_count,
    #     "Sell Signals": sell_count,
    #     "Neutral Signals": neutral_count,
    #     "Final Signal": final_signal
    # }

def count_signals(indicator_data):
    """
    Count the number of Buy, Sell, and Neutral signals in the indicator data.

    Parameters:
        indicator_data (list of dict): List of dictionaries, where each contains a date and a signal.

    Returns:
        dict: A dictionary with counts for each signal type.
    """
    buy_count = 0
    sell_count = 0
    neutral_count = 0

    # Iterate over the signals
    for entry in indicator_data:
        signal = entry.get("Signal", "Neutral")  # Default to "Neutral" if "Signal" key is missing
        if signal == "Buy":
            buy_count += 1
        elif signal == "Sell":
            sell_count += 1
        else:
            neutral_count += 1

    return {
        "Buy": buy_count,
        "Sell": sell_count,
        "Neutral": neutral_count
    }
    
# Function to calculate MACD Histogram
def calculate_macd(data):
    # Calculate the MACD line
    macd = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    
    # Calculate the Signal line
    Signal = macd.ewm(span=9).mean()
    
    # Calculate the Histogram (difference between MACD and Signal)
    hist = macd - Signal
    
    # Return as a dictionary
    return {
        'macd': macd,
        'Signal': Signal,
        'hist': hist
    }

# Function to calculate Relative Volume
def calculate_relative_volume(data, window=1):
    avg_volume = data['volume'].rolling(window=window).mean()
    rel_volume = data['volume'] / avg_volume
    return rel_volume

# Function to calculate Price Change Percentage
def calculate_price_change_pct(data, interval=1):
    price_change_pct = data['close'].pct_change(periods=interval) * 100
    return price_change_pct


def identify_bullish_engulfing(data):
    """Identify Bullish Engulfing Pattern."""
    condition = (
        (data['open'].shift(1) > data['close'].shift(1)) &  # Previous candle is bearish
        (data['open'] < data['close']) &                   # Current candle is bullish
        (data['open'] < data['close'].shift(1)) &          # Current open is below previous close
        (data['close'] > data['open'].shift(1))            # Current close is above previous open
    )
    return np.where(condition, 1, 0)

def identify_hammer(data):
    try:
        """Identify Hammer candlestick pattern."""
        body = abs(data['close'] - data['open'])
        lower_wick = data[['close', 'open']].min(axis=1) - data['low']
        upper_wick = data['high'] - data[['close', 'open']].max(axis=1)
        
        condition = (
            (lower_wick > 1.5 * body) &            # Lower wick is at least 2x the body
            (upper_wick <= body * 0.5) &        # Upper wick is less than half the body
            (data['close'] > data['open'])      # Optional: Bullish hammer
        )
        return np.where(condition, 1, 0)
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")

def define_bullish_signal(data):
    """
    Define Bullish and Bearish Signals based on various indicators and candlestick patterns.
    """
    data = data.copy()

    # Calculate RSI
    data['RSI'] = calculate_rsi_with_signals(data)['RSI']

    # Calculate Moving Averages
    data['50_SMA'] = calculate_sma(data, 50)['SMA']
    data['200_SMA'] = calculate_sma(data, 200)['SMA']

    # Calculate EMAs
    data['12_EMA'] = calculate_ema(data, 12)['EMA']
    data['26_EMA'] = calculate_ema(data, 26)['EMA']

    # Identify Candlestick Patterns
    data['Bullish_Engulfing'] = identify_bullish_engulfing(data)
    data['Bearish_Engulfing'] = identify_bearish_engulfing(data)  # New addition
    data['Hammer'] = identify_hammer(data)

    # Initialize Bullish Signal column
    data['Bullish_Signal'] = 0

    # Rule 1: RSI is between 30 and 70
    data.loc[(data['RSI'] > 30) & (data['RSI'] < 70), 'Bullish_Signal'] = 1

    # Rule 2: Golden Cross (50 SMA crosses above 200 SMA)
    data.loc[
        (data['50_SMA'] > data['200_SMA']) & (data['50_SMA'].shift(1) <= data['200_SMA'].shift(1)),
        'Bullish_Signal'
    ] = 1

    # Rule 3: 12 EMA crosses above 26 EMA
    data.loc[
        (data['12_EMA'] > data['26_EMA']) & (data['12_EMA'].shift(1) <= data['26_EMA'].shift(1)),
        'Bullish_Signal'
    ] = 1

    # Rule 4: Bullish Engulfing Pattern
    data.loc[data['Bullish_Engulfing'] > 0, 'Bullish_Signal'] = 1

    # Rule 5: Hammer candlestick pattern
    data.loc[data['Hammer'] > 0, 'Bullish_Signal'] = 1

    # Rule 6: Close near or above the day's high
    data.loc[data['close'] >= data['high'] * 0.95, 'Bullish_Signal'] = 1

    # Rule 7: Close breaks above the previous day's high
    data.loc[data['close'] > data['high'].shift(1), 'Bullish_Signal'] = 1

    return data

def identify_bearish_engulfing(data):
    """Identify Bearish Engulfing Pattern."""
    condition = (
        (data['close'].shift(1) > data['open'].shift(1)) &  # Previous candle is bullish
        (data['open'] > data['close']) &                   # Current candle is bearish
        (data['open'] > data['close'].shift(1)) &          # Current open is above previous close
        (data['close'] < data['open'].shift(1))            # Current close is below previous open
    )
    return np.where(condition, 1, 0)

# Modified get_stock_data function
def get_stock_data(symbol, exchange, interval=Interval.in_daily, n_bars=5000):
    tv = TvDatafeed(username="None", password="None")  # Replace with your credentials
    try:
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        if data is not None:
            data.dropna(inplace=True)
            win_size = 5
            
            # Add technical indicators
            data['Range'] = data["high"] - data["low"]
            rolling_close = data['close'].rolling(window=win_size)
            data['SMA'] = rolling_close.mean()
            rolling_std = rolling_close.std()
            data['BB_upper'] = data['SMA'] + 2 * rolling_std
            data['BB_lower'] = data['SMA'] - 2 * rolling_std
            macd_values = calculate_macd(data)
            data['MACD'] = macd_values['macd']
            data['Signal_line'] = macd_values['Signal']
            data['MACD_hist'] = macd_values['hist']
            data['Rel_Volume'] = calculate_relative_volume(data)
            data['Price_Change_Pct'] = calculate_price_change_pct(data)
            print(f"Price change {data['Price_Change_Pct'] }")
            data = define_bullish_signal(data)
            final = 0
            final = calculate_final_signal(data)
            print(f"final signal {final}")
            data['Signal'] = final
            
            #data['Bullish_Engulfing'], data['Bearish_Engulfing'] = detect_candlestick_patterns(data)
            
            # Apply the new logic in your pipeline
            
            # print(data[['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line',
            # 'MACD_hist', 'Rel_Volume', 'Price_Change_Pct',
            # 'Bullish_Engulfing', 'Bearish_Engulfing', 'Hammer']].isna().sum())

            return data
        else:
            print(f"No data returned for {symbol} on {exchange}.")
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")


def train_best_model(X, y, class_weight=None):
    # Split data (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Reduced hyperparameter grid for faster tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    # Initialize RandomForestClassifier with class_weight
    model = RandomForestClassifier(class_weight=class_weight, random_state=42)
    
    # Use RandomizedSearchCV for faster tuning with parallel processing
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=5,  # Fewer parameter settings sampled for speed
        cv=3,
        random_state=42,
        n_jobs=-1  # Utilize all CPU cores
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
    important_signal_array = []
    for index, row in top_40.iterrows():
        symbol = row["symbol"]
        exchange = "CSELK"
        company_bought = row["companyExists"]
        
        # Fetch historical data
        data = get_stock_data(symbol, exchange, interval, n_bars)
        print(len(data))
        if data is not None and not data.empty:
            # Prepare features and target
            X = data[['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line', 
                      'MACD_hist', 'Rel_Volume', 'Price_Change_Pct', 
                      'Bullish_Engulfing', 'Bearish_Engulfing', 'Hammer']].dropna()
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
                                              'Bullish_Engulfing', 'Bearish_Engulfing', 'Hammer']].dropna()
                if not latest_input.empty:
                    predicted_signal = best_model.predict(latest_input)[0]                    
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
