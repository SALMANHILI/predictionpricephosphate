#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
import re
# Load the dataset
df = pd.read_csv("phosphate36.csv")
# Check the dataset
# Modify the column names to match the actual column names in the dataset
df.columns = ['Mois',
              'Phosphate Price (Dollars américains par tonne métrique)',
              'Diesel Price (Dollars US par gallon)',
              'Phosphate ROC',
              'Diesel ROC',
              'Phosphate / Diesel Price Ratio']

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
df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].astype(str).str.replace(',', '.').astype(float)
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

# Convert month names to numerical values
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Map the month names

df['Month'] = df['Month'].apply(lambda x: month_mapping[re.search(r'[a-zA-Zéû]+', str(x)).group()] if pd.notnull(x) else x)
df

# X (independent variables) and y (target variable)
X = df[['Month','Year','Diesel Price (Dollars US par gallon)', 'Diesel ROC','Phosphate / Diesel Price Ratio']]

y = df['Phosphate Price (Dollars américains par tonne métrique)']

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

july = mlr.predict(x_test)
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
print("Multiple Linear Regression Model Score:", round(mlr_score * 100, 2))
print("Decision Tree Regression Model Score:", round(tr_regressor_score * 100, 2))
print("Random Forest Regression Model Score:", round(rf_regressor_score * 100, 2))


manual_input = pd.DataFrame({ 'Month':[6],
    'Year': [2023],
    'Diesel Price (Dollars US par gallon)': [2.83],  # Replace with the actual diesel price for July 2023
    'Phosphate / Diesel Price Ratio': [148.4567],
    'Diesel ROC':[2.83]
 })


input = scaler.transform(manual_input)
# right
print(scaler.transform(manual_input))


# Predict phosphate price for July 2023 using each model
pred_mlr_july = mlr.predict(input)
pred_tr_july = tr_regressor.predict(input)
pred_rf_july = rf_regressor.predict(input)

print("Predicted Phosphate Price for July 2023 (MLR):", pred_mlr_july[0])
print("Predicted Phosphate Price for July 2023 (Decision Tree):", pred_tr_july[0])
print("Predicted Phosphate Price for July 2023 (Random Forest):", pred_rf_july[0])





#%%
# Actual phosphate price for July
y_actual = 342.5

"""
# Calculate the R-squared for each model
def calculate_r_squared(y_actual, y_predicted):
    y_mean = y_actual
    ss_total = sum((y_actual - y_mean) ** 2)
    ss_residual = sum((y_actual - y_predicted) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

# Calculate R-squared for Multiple Linear Regression model
r_squared_mlr = calculate_r_squared([y_actual], [pred_mlr_july[0]])

# Calculate R-squared for Decision Tree Regression model
r_squared_tr = calculate_r_squared([y_actual], [pred_tr_july[0]])

# Calculate R-squared for Random Forest Regression model
r_squared_rf = calculate_r_squared([y_actual], [pred_rf_july[0]])

print("R-squared (MLR):", r_squared_mlr)
print("R-squared (Decision Tree):", r_squared_tr)
print("R-squared (Random Forest):", r_squared_rf)"""








#%% NEXT YEAAAAAAARRRRRRR
# Create a DataFrame to store the input values for the next 12 months
future_input = pd.DataFrame({
    'Month': [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7],
    'Year': [2023,2023,2023,2023,2023,2024,2024,2024,2024,2024,2024,2024],  # Replace with the appropriate year values
    'Diesel Price (Dollars US par gallon)': [2.83,3.83,1.83,2.83,3.83,1.83,2.83,3.83,1.83,2.83,3.83,2.83],  # Replace with the actual diesel price for each month
        'Diesel ROC':[-2.71,-4.8,26.49,-6.49,-23.28,4.81,-13.31,-3.00,-5.61,-9.53,3.67,4.11],
    'Phosphate / Diesel Price Ratio': [148.4567,128.4567,108.4567,100.4567,90.4567,148.4567,158.4567,130.4567,108.4567,90.4567,80.4567,148.4567]   # Replace with the actual ratios for each month
})

# Scale the future input data
future_input_scaled = scaler.transform(future_input)

# Predict phosphate prices for the next 12 months using each model
pred_mlr_future = mlr.predict(future_input_scaled)
pred_tr_future = tr_regressor.predict(future_input_scaled)
pred_rf_future = rf_regressor.predict(future_input_scaled)

# Create a DataFrame to store the predicted prices for the next 12 months
predictions_df = pd.DataFrame({
    'Month': future_input['Month'],
    'Year': future_input['Year'],
    'Predicted Phosphate Price (MLR)': pred_mlr_future,
    'Predicted Phosphate Price (Decision Tree)': pred_tr_future,
    'Predicted Phosphate Price (Random Forest)': pred_rf_future
})

# Map the numerical month values back to month names
#predictions_df['Month'] = predictions_df['Month'].map(month_mapping)

# Display the predicted prices for the next 12 months
predictions_df















# %% PAAST YEAAAR 

# Create a DataFrame to store the input values for the next 12 months
future_input = pd.DataFrame({
    'Month': [8, 9, 10, 11, 12, 1, 2, 3, 4, 5,6, 7],
    'Year': [2022,2022,2022,2022,2022,2023,2023,2023,2023,2023,2023,2023],  # Replace with the appropriate year values
    'Diesel Price (Dollars US par gallon)': [3.6, 3.44, 4.35, 4.06, 3.12, 3.27, 2.83, 2.75, 2.59, 2.35, 3.43, 2.53],  # Replace with the actual diesel price for each month
    'Diesel ROC':[-2.71,-4.8,26.49,-6.49,-23.28,4.81,-13.31,-3.00,-5.61,-9.53,3.67,4.11],
    'Phosphate / Diesel Price Ratio': [65.91,86.58,88.98,93.15,73.07,73.83,96.24,91.82,113.87,125.59,133.05,147.05]   # Replace with the actual ratios for each month
})

# Scale the future input data
future_input_scaled = scaler.transform(future_input)

# Predict phosphate prices for the next 12 months using each model
pred_mlr_future = mlr.predict(future_input_scaled)
pred_tr_future = tr_regressor.predict(future_input_scaled)
pred_rf_future = rf_regressor.predict(future_input_scaled)

# Create a DataFrame to store the predicted prices for the next 12 months
predictions_df = pd.DataFrame({
    'Month': future_input['Month'],
    'Year': future_input['Year'],
    'Predicted Phosphate Price (MLR)': pred_mlr_future,
    'Predicted Phosphate Price (Decision Tree)': pred_tr_future,
    'Predicted Phosphate Price (Random Forest)': pred_rf_future
})

# Map the numerical month values back to month names
#predictions_df['Month'] = predictions_df['Month'].map(month_mapping)

# Display the predicted prices for the next 12 months
predictions_df
# %%
