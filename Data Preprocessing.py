# DATA PREPROCESSING
# Handle missing values and outliers appropriately.
# Normalize or scale features as needed.
# Split the data into training and testing sets.

# 1. Handle missing values and outliers appropriately.
import pandas as pd
import numpy as np

data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, 6, 7, 8, np.nan],
    'C': [10, 11, np.nan, 14, 15],
    'D': [1, 2, 100, 4, 5]  # Outlier in column D
}

df = pd.DataFrame(data)

df_dropped = df.dropna()

df_filled = df.fillna(df.mean())

Q1 = df['D'].quantile(0.25)
Q3 = df['D'].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df['D'] < (Q1 - 1.5 * IQR)) | (df['D'] > (Q3 + 1.5 * IQR))
df_no_outliers = df[~outlier_condition]

# 2. Normalize or scale features as needed.
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 6, 7, 8, 9],
    'C': [10, 11, 12, 13, 14]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

min_max_scaler = MinMaxScaler()
normalized_df = min_max_scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_df, columns=df.columns)

# 3. Split the data into training and testing sets.
from sklearn.model_selection import train_test_split

data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 6, 7, 8, 9],
    'C': [10, 11, 12, 13, 14],
    'target': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training features:\n", X_train)
print("Testing features:\n", X_test)
print("Training target:\n", y_train)
print("Testing target:\n", y_test)