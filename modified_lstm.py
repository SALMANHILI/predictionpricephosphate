#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
import re
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

df = pd.read_csv("phosphate-36.csv", skiprows=1)

# Check the dataset
print(df.head())

# Modify the column names to match the actual column names in the dataset
df.columns = ['Mois',
            'Phosphate Price (Dollars américains par tonne métrique)',
            'Diesel Price (Dollars US par gallon)',
            'Phosphate ROC',
            'Diesel ROC',
            'Phosphate / Diesel Price Ratio']

# Modify the column names to match the actual column names in the dataset
df.columns = ['Mois',
            'Phosphate Price (Dollars américains par tonne métrique)',
            'Diesel Price (Dollars US par gallon)',
            'Phosphate ROC',
            'Diesel ROC',
            'Phosphate / Diesel Price Ratio']

# Define the month mapping

# Map the month names
df['Mois'] = df['Mois']
# Convert 'Phosphate Price' column to numeric values
df['Phosphate Price (Dollars américains par tonne métrique)'] = df['Phosphate Price (Dollars américains par tonne métrique)'].astype(str).str.replace(',', '.').astype(float)

# Convert 'Diesel Price' column to numeric values
df['Diesel Price (Dollars US par gallon)'] = df['Diesel Price (Dollars US par gallon)'].str.replace(',', '.').astype(float)

# Convert 'Phosphate ROC' column to numeric values
df['Phosphate ROC'] = df['Phosphate ROC'].replace('-', '0')  # Replace missing values ('-') with '0'
df['Phosphate ROC'] = df['Phosphate ROC'].str.replace(',', '.')  # Replace commas with dots
df['Phosphate ROC'] = df['Phosphate ROC'].str.rstrip('%').astype(float)  # Remove '%' and convert to float

# Convert 'Diesel ROC' column to numeric values
df['Diesel ROC'] = df['Diesel ROC'].replace('-', '0')  # Replace missing values ('-') with '0'
df['Diesel ROC'] = df['Diesel ROC'].str.replace(',', '.')  # Replace commas with dots
df['Diesel ROC'] = df['Diesel ROC'].str.rstrip('%').astype(float)  # Remove '%' and convert to float

# Remove both dots and commas from 'Phosphate / Diesel Price Ratio' column
df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].str.replace(',', '.')  # Replace commas with dots

# Split the "Mois" column into "Month" and "Year" columns
df[["Month", "Year"]] = df["Mois"].str.split(" ", n=1, expand=True)

# Drop the original "Mois" column
df.drop(columns=["Mois"], inplace=True)

# Display the resulting DataFrame
print(df)

month_mapping = {
    'janv': 'January',
    'févr': 'February',
    'mars': 'March',
    'avr': 'April',
    'mai': 'May',
    'juin': 'June',
    'juil.': 'July',
    'juil': 'July',
    'août': 'August',
    'sept.': 'September',
    'sept': 'September',
    'oct.': 'October',
    'oct': 'October',
    'nov.': 'November',
    'nov': 'November',
    'déc.': 'December',
    'déc': 'December'
}

# Map the month names
df['Month'] = df['Month'].apply(lambda x: month_mapping[re.search(r'[a-zA-Zéû]+', str(x)).group()] if pd.notnull(x) else x)
print(df)

# Convert month names to numerical values
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Map the month names
df['Month'] = df['Month'].apply(lambda x: month_mapping[re.search(r'[a-zA-Zéû]+', str(x)).group()] if pd.notnull(x) else x)
print(df['Month'])
#%%
# Feature scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

X, y = [], []

# Create sequences for LSTM training
for i in range(60, data_scaled.shape[0]):
    X.append(data_scaled[i - 60: i, 2:])  # Use 'Diesel Price' as the feature
    y.append(data_scaled[i, 0])  # Predict 'Phosphate Price'

X, y = np.array(X), np.array(y)

# Define the input shape for LSTM
input_shape = (X.shape[1], X.shape[2])  # (num_timesteps, num_features)

# Build the LSTM model
regressor = Sequential()
regressor.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units=120, activation='relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units=1))

# Compile the model
regressor.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)

# Train the model
regressor.fit(X, y, epochs=50, batch_size=64)
#%%
# Evaluate the model on the test set
test_loss = regressor.evaluate(X, y)
print(f"Test Loss: {test_loss}")
#%%
# Split data into training and test sets
train_size = int(100)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]
# Feature scaling
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

X_test, y_test = [], []
for i in range(60, test_scaled.shape[0]):
    X_test.append(test_scaled[i-60: i])
    y_test.append(test_scaled[i, 0])
X_test, y_test = np.array(X_test.iloc), np.array(y_test)

#%%
predictions = regressor.predict(X_test)
data_testing = df.copy()
# Scale the testing data
data_testing_scaled = scaler.transform(data_testing)

# If 'Month' values are numeric strings
matching_indices = df[(df['Month'] == '6') & (df['Year'] == '2023')].index


print(matching_indices)