import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval
import json
from SymbolPredictor import get_stock_data

# Sample data (replace with your actual stock data)
np.random.seed(42)
interval = getattr(Interval, "in_daily", Interval.in_daily)
data = get_stock_data(symbol="JKH.N0000", exchange="CSELK", interval=interval, n_bars=5000)

df = pd.DataFrame(data)

# Feature scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line', 
                      'MACD_hist', 'Rel_Volume', 'Price_Change_Pct', 
                      'Bullish_Engulfing', 'Bearish_Engulfing']])
df[['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line', 
                      'MACD_hist', 'Rel_Volume', 'Price_Change_Pct', 
                      'Bullish_Engulfing', 'Bearish_Engulfing']] = scaled_features

# Visualize correlation between features and Signal
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# Prepare data for model
X = df[['SMA', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'Signal_line', 
                      'MACD_hist', 'Rel_Volume', 'Price_Change_Pct', 
                      'Bullish_Engulfing', 'Bearish_Engulfing']]
y = df['Signal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions and classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))