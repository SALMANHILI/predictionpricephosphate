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


#AUGUST PREDICTION
"""july_predicted_mlr =  pred_mlr_june[0]
july_predicted_tr = pred_tr_june[0]
july_predicted_rf = pred_rf_june[0]


# Create a new DataFrame to store the predicted prices for July and set the index to match the original DataFrame 'df'
df_july_predicted = pd.DataFrame(index=df.index)
df_july_predicted['Predicted Prices July (MLR)'] = july_predicted_mlr
df_july_predicted['Predicted Prices July (Decision Tree)'] = july_predicted_tr
df_july_predicted['Predicted Prices July (Random Forest)'] = july_predicted_rf


# Concatenate the predicted prices for July with the original DataFrame 'df'
df_with_july_predicted = pd.concat([df, df_july_predicted], axis=1)

# Retrain the models on the entire dataset (including July)
X_august = df_with_july_predicted[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
X_august = pd.get_dummies(X_august, columns=["Mois"])  # Perform one-hot encoding

# Initialize the StandardScaler
scaler_august = StandardScaler()

# Scale the data
X_scaled_august = scaler_august.fit_transform(X_august)

# Retrain the models on the entire dataset (including July)
mlr.fit(X_scaled_august, y)
tr_regressor.fit(X_scaled_august, y)
rf_regressor.fit(X_scaled_august, y)

# Predict the last line's phosphate price for each model
last_august_data = X.iloc[203:].copy()  # Extract the last line's data for June
last_august_data_scaled = scaler.transform(last_august_data)  # Scale the last line's data

pred_mlr_august = mlr.predict(last_august_data_scaled)
pred_tr_august = tr_regressor.predict(last_august_data_scaled)
pred_rf_august = rf_regressor.predict(last_august_data_scaled)

# Print the predicted prices for phosphates in August for each model
print("Predicted Prices for Phosphates in August:")
print("Multiple Linear Regression Model:", pred_mlr_august[0])
print("Decision Tree Regression Model:", pred_tr_august[0])
print("Random Forest Regression Model:", pred_rf_august[0])"""


#%%# 
#%%
#%%
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

#%%
# Step 1: Prepare Data for June 2022 and Make Predictions
# Assuming you have the data for June 2022 in a DataFrame named "df_june_2022"

# Scale the data for June 2022 using the previously defined scaler (scaler_june)
df_june_2022= X.iloc[-13:].copy() 
data_june_2022_scaled = scaler.transform(df_june_2022)

# Make predictions for June 2022 using the trained models (mlr, tr_regressor, rf_regressor)
pred_mlr_june_2022 = mlr.predict(data_june_2022_scaled)
pred_tr_june_2022 = tr_regressor.predict(data_june_2022_scaled)
pred_rf_june_2022 = rf_regressor.predict(data_june_2022_scaled)

# Step 2: Create "df_with_june_predicted" DataFrame
# Create a new DataFrame to store the predicted values for each month
df_with_june_predicted = df_june_2022.copy()

# Add the predicted prices for June 2022 to the DataFrame
df_with_june_predicted['Predicted Price MLR'] = pred_mlr_june_2022
df_with_june_predicted['Predicted Price TR'] = pred_tr_june_2022
df_with_june_predicted['Predicted Price RF'] = pred_rf_june_2022

# Step 3: Continue Predictions for Subsequent Months
# Now you can proceed with the loop to make predictions for the remaining months
# (July 2022 to June 2023) using the trained models and the "df_with_june_predicted" DataFrame.

# Define the list of months in the order you want to predict (from July 2022 to June 2023)
months_to_predict = ['July 2022', 'August 2022', 'September 2022', 'October 2022', 'November 2022', 'December 2022',
                     'January 2023', 'February 2023', 'March 2023', 'April 2023', 'May 2023', 'June 2023']

# Initialize empty lists to store predicted prices for each model
predicted_prices_mlr = []
predicted_prices_tr = []
predicted_prices_rf = []

# Initialize the LabelEncoder object
label_encoder = LabelEncoder()
# Loop through each month and make predictions
for month in months_to_predict:
    print("Current Month:", month)
    print("df_with_june_predicted:")
    print(df_with_june_predicted)

    # Prepare input data for the current month
    column_name = 'Mois_' + month.capitalize()
    if column_name in df_with_june_predicted.columns:
        current_month_data = df_with_june_predicted[df_with_june_predicted[column_name] == 1]

        # Check if the DataFrame for the current month is empty
        if not current_month_data.empty:
            # Encode the "Month" column using LabelEncoder
            current_month_data['Month'] = label_encoder.transform(current_month_data['Month'])

            # Assuming 'numeric_columns' is a list of numerical column names
            numeric_columns = ['Phosphate Price (Dollars américains par tonne métrique)', 'Diesel Price (Dollars US par gallon)', 'Phosphate ROC', 'Diesel ROC', 'Phosphate / Diesel Price Ratio']

            # Scale numeric data for the current month
            current_month_data_scaled = current_month_data.copy()
            current_month_data_scaled[numeric_columns] = scaler.transform(current_month_data[numeric_columns])

            # Predict phosphate prices for the current month using the trained models
            pred_mlr_month = mlr.predict(current_month_data_scaled)
            pred_tr_month = tr_regressor.predict(current_month_data_scaled)
            pred_rf_month = rf_regressor.predict(current_month_data_scaled)

            # Append the predicted prices to the respective lists
            predicted_prices_mlr.append(pred_mlr_month[0])
            predicted_prices_tr.append(pred_tr_month[0])
            predicted_prices_rf.append(pred_rf_month[0])

            # Print the predicted prices for phosphates for each model
            print("Predicted Prices for Phosphates in", month, ":")
            print("Multiple Linear Regression Model:", pred_mlr_month[0])
            print("Decision Tree Regression Model:", pred_tr_month[0])
            print("Random Forest Regression Model:", pred_rf_month[0])

            # Calculate the accuracy of the predictions for each model
            actual_price = current_month_data['Phosphate Price (Dollars américains par tonne métrique)'].values[0]
            accuracy_mlr = 100 * (1 - abs((pred_mlr_month[0] - actual_price) / actual_price))
            accuracy_tr = 100 * (1 - abs((pred_tr_month[0] - actual_price) / actual_price))
            accuracy_rf = 100 * (1 - abs((pred_rf_month[0] - actual_price) / actual_price))

            print("Actual Phosphate Price for the Last Month:", actual_price)
            print("Accuracy for Multiple Linear Regression Model:", accuracy_mlr)
            print("Accuracy for Decision Tree Regression Model:", accuracy_tr)
            print("Accuracy for Random Forest Regression Model:", accuracy_rf)
            print("------------------------------------------------------")
        else:
            print(f"Data for {month} is not available. Skipping predictions for this month.")
    else:
        print(f"Column '{column_name}' is missing in the DataFrame. Skipping predictions for the month of {month}.")
