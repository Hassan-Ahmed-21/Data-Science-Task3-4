# FEATURE ENGINEERING
# Create additional features that might be useful for predicting equipment failure.
# Consider time-based features, rolling statistics, and any other relevant transformations.

import pandas as pd
import numpy as np

data = {
    'timestamp': pd.date_range(start='1/1/2023', periods=10, freq='D'),
    'sensor_reading': [10, 12, 13, 10, 15, 14, 13, 16, 15, 14],
    'failure': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# Time-based features
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Rolling statistics
df['rolling_mean'] = df['sensor_reading'].rolling(window=3).mean()
df['rolling_std'] = df['sensor_reading'].rolling(window=3).std()

# Lag features
df['lag_1'] = df['sensor_reading'].shift(1)
df['lag_2'] = df['sensor_reading'].shift(2)

# Cumulative sum
df['cumulative_sum'] = df['sensor_reading'].cumsum()

# Difference
df['diff'] = df['sensor_reading'].diff()

# Display the dataframe with new features
print(df)
