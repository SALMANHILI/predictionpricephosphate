#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
import re
# Load the dataset
df = pd.read_csv("phosphate36.csv", skiprows=1)

# Check the dataset
print(df.head())

# Modify the column names to match the actual column names in the dataset
df.columns = ['Mois',
              'Phosphate Price (Dollars américains par tonne métrique)',
              'Diesel Price (Dollars US par gallon)',
              'Phosphate ROC',
              'Diesel ROC',
              'Phosphate / Diesel Price Ratio']

# Define the month mapping
# Define the month mapping
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
df['Mois'] = df['Mois'].apply(lambda x: month_mapping[re.search(r'[a-zA-Zéû]+', str(x)).group()] if pd.notnull(x) else x)
print(df)
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
df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].str.replace('[,.]', '', regex=True)

# Convert 'Phosphate / Diesel Price Ratio' column to numeric values
df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].astype(float)

# X (independent variables) and y (target variable)
X = df[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
y = df['Phosphate Price (Dollars américains par tonne métrique)']
# Perform one-hot encoding on the "Mois" column
X = pd.get_dummies(X, columns=["Mois"])

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale the data
x_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.26, random_state=0)
# Train the models
mlr = LinearRegression()
mlr.fit(x_train, y_train)
mlr_score = mlr.score(x_test, y_test)
pred_mlr = mlr.predict(x_test)
expl_mlr = explained_variance_score(pred_mlr, y_test)

tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(x_train, y_train)
tr_regressor_score = tr_regressor.score(x_test, y_test)
pred_tr = tr_regressor.predict(x_test)
expl_tr = explained_variance_score(pred_tr, y_test)

rf_regressor = RandomForestRegressor(n_estimators=28, random_state=0)
rf_regressor.fit(x_train, y_train)
rf_regressor_score = rf_regressor.score(x_test, y_test)
rf_pred = rf_regressor.predict(x_test)
expl_rf = explained_variance_score(rf_pred, y_test)
# Print the predicted prices for phosphates in June for each model
# Print the predicted prices for phosphates in June for each model
print("Multiple Linear Regression Model Score:", round(mlr_score * 100, 2))
print("Decision Tree Regression Model Score:", round(tr_regressor_score * 100, 2))
print("Random Forest Regression Model Score:", round(rf_regressor_score * 100, 2))

# %%
# Assuming that the 'June' data is available in the DataFrame 'df', you can filter it like this:
june_data = df[df['Mois'] == 'June']

# Extract the independent variables for the June data
X_june = june_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
# Convert them to strings first before performing replacement
X_june['Diesel Price (Dollars US par gallon)'] = X_june['Diesel Price (Dollars US par gallon)'].astype(str)
X_june['Phosphate / Diesel Price Ratio'] = X_june['Phosphate / Diesel Price Ratio'].astype(str)

# Now perform the replacements
X_june['Diesel Price (Dollars US par gallon)'] = X_june['Diesel Price (Dollars US par gallon)'].str.replace(',', '.')
X_june['Phosphate / Diesel Price Ratio'] = X_june['Phosphate / Diesel Price Ratio'].str.replace(',', '.').astype(float)

# Make a copy of 'X_june' to avoid the SettingWithCopyWarning
X_june_copy = X_june.copy()
# Convert the necessary columns to strings first before performing replacement
X_june_copy['Diesel Price (Dollars US par gallon)'] = X_june_copy['Diesel Price (Dollars US par gallon)'].astype(str)
X_june_copy['Phosphate / Diesel Price Ratio'] = X_june_copy['Phosphate / Diesel Price Ratio'].astype(str)

# Now perform the replacements using .loc accessor
X_june_copy['Diesel Price (Dollars US par gallon)'] = X_june_copy['Diesel Price (Dollars US par gallon)'].str.replace(',', '.')
X_june_copy['Phosphate / Diesel Price Ratio'] = X_june_copy['Phosphate / Diesel Price Ratio'].str.replace(',', '.').astype(float)

# Perform one-hot encoding on the "Mois" column for June data
# Check if 'Mois' column exists in X_june_copy DataFrame
if 'Mois' in X_june_copy.columns:
    # Perform one-hot encoding on the 'Mois' column for June data
    X_june_copy = pd.get_dummies(X_june_copy, columns=["Mois"])
else:
    # Handle the case when 'Mois' column is missing or not found
    # You might want to print a message or take appropriate action based on your requirement.
    print("Column 'Mois' not found in the DataFrame.")

# Ensure 'X_june_copy' has the same set of features and in the same order as 'X_train'
X_june_processed = X_june_copy.reindex(columns=X.columns, fill_value=0)

# Print the columns of both 'X_train' and 'X_june_processed' for comparison
print("Training Data Columns:", X.columns)
print("X_june_processed Columns:", X_june_processed.columns)

# Predict using the trained models
pred_mlr_june = mlr.predict(X_june_processed)
pred_tr_june = tr_regressor.predict(X_june_processed)
pred_rf_june = rf_regressor.predict(X_june_processed)

# Print the predicted prices for phosphates in June for each model
print("Predicted Prices for Phosphates in June:")
print("Multiple Linear Regression Model:", pred_mlr_june[0])
print("Decision Tree Regression Model:", pred_tr_june[0])
print("Random Forest Regression Model:", pred_rf_june[0])

#%%

from sklearn.metrics import explained_variance_score

# Predict using the trained models
pred_mlr_june = mlr.predict(X_june_processed)
pred_tr_june = tr_regressor.predict(X_june_processed)
pred_rf_june = rf_regressor.predict(X_june_processed)

print("Predicted Prices for Phosphates in June:")
print("Multiple Linear Regression Model:", pred_mlr_june[0])
print("Decision Tree Regression Model:", pred_tr_june[0])
print("Random Forest Regression Model:", pred_rf_june[0])


# Assuming the actual phosphate prices for June 2023 are available in 'june_data'
actual_prices = df['Phosphate Price (Dollars américains par tonne métrique)'].iloc[-1]

# Create an array-like object with the same number of elements as predicted prices
actual_prices_list = [actual_prices] * len(pred_mlr_june)

# Calculate the explained variance score for each model's predictions
accuracy_mlr = explained_variance_score(pred_mlr_june, actual_prices_list)
accuracy_tr = explained_variance_score(pred_tr_june, actual_prices_list)
accuracy_rf = explained_variance_score(actual_prices_list, pred_rf_june)

# Print the actual phosphate price for the last month
print("Actual Phosphate Price for the Last Month:", actual_prices)

# Print the accuracy for each model
print("Accuracy for Multiple Linear Regression Model:", round(accuracy_mlr * 100, 2))
print("Accuracy for Decision Tree Regression Model:", round(accuracy_tr * 100, 2))
print("Accuracy for Random Forest Regression Model:", round(accuracy_rf * 100, 2))



# %%
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Predict the variations and rolling averages for June 2023
historical_data = df[df['Mois'] < 'juin 2023']  # Filter the data for all rows before June 2023

# Create an input DataFrame for June 2023 data using historical data
input_data = historical_data[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]

print(input_data)

# Assuming you have already defined and preprocessed the 'x_train' and 'y_train' variables

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale the data
X_scaled = scaler.fit_transform(X)

# Train the models on the entire dataset
mlr = LinearRegression()
mlr.fit(X_scaled, y)

tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(X_scaled, y)

rf_regressor = RandomForestRegressor(n_estimators=28, random_state=0)
rf_regressor.fit(X_scaled, y)

# Predict the last line's phosphate price for each model
last_june_data = X.iloc[-1:].copy()  # Extract the last line's data for June
last_june_data_scaled = scaler.transform(last_june_data)  # Scale the last line's data

pred_mlr_june = mlr.predict(last_june_data_scaled)
pred_tr_june = tr_regressor.predict(last_june_data_scaled)
pred_rf_june = rf_regressor.predict(last_june_data_scaled)

# Print the predicted prices for phosphates in June for each model
print("Predicted Prices for Phosphates in June:")
print("Multiple Linear Regression Model:", pred_mlr_june[0])
print("Decision Tree Regression Model:", pred_tr_june[0])
print("Random Forest Regression Model:", pred_rf_june[0])

#%%
accuracy_mlr = (1 - abs((pred_mlr_june[0]-actual_prices) / actual_prices)) * 100

accuracy_tr = (1 - abs((pred_tr_june[0]-actual_prices) / actual_prices)) * 100

accuracy_rf = (1 - abs((pred_rf_june[0]-actual_prices) / actual_prices)) * 100
# Print the actual phosphate price for the last month
print("Actual Phosphate Price for the Last Month:", actual_prices)

# Print the accuracy for each model
print("Accuracy for Multiple Linear Regression Model:", accuracy_mlr)
print("Accuracy for Decision Tree Regression Model:", accuracy_tr)
print("Accuracy for Random Forest Regression Model:", accuracy_rf)
#%%
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Predict the variations and rolling averages for June 2023
historical_data = df[df['Mois'] <='juin 2023']  # Filter the data for all rows before June 2023


# Create an input DataFrame for June 2023 data using historical data
input_data = historical_data[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
input_data = {
    'Diesel Price (Dollars US per gallon)': 2.5,
    'Phosphate / Diesel Price Ratio': 141.6530,
}
print(input_data)

# Assuming you have already defined and preprocessed the 'x_train' and 'y_train' variables

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale the data
X_scaled = scaler.fit_transform(X)

# Train the models on the entire dataset
mlr = LinearRegression()
mlr.fit(X_scaled, y)

tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(X_scaled, y)

rf_regressor = RandomForestRegressor(n_estimators=28, random_state=0)

rf_regressor.fit(X_scaled, y)

# Predict the last line's phosphate price for each model
last_june_data = X.iloc[203:].copy()  # Extract the last line's data for June
last_june_data_scaled = scaler.transform(last_june_data)  # Scale the last line's data

pred_mlr_june = mlr.predict(last_june_data_scaled)
pred_tr_june = tr_regressor.predict(last_june_data_scaled)
pred_rf_june = rf_regressor.predict(last_june_data_scaled)

# Print the predicted prices for phosphates in June for each model
print("Predicted Prices for Phosphates in July:")
print("Multiple Linear Regression Model:", pred_mlr_june[0])
print("Decision Tree Regression Model:", pred_tr_june[0])
print("Random Forest Regression Model:", pred_rf_june[0])