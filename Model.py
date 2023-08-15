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

# Assuming the actual phosphate prices for June 2023 are available in 'june_data'
actual_prices = df['Phosphate Price (Dollars américains par tonne métrique)'].iloc[-1]

# Calculate R-squared values
r2_mlr = mlr.score(X_scaled, y)
r2_tr = tr_regressor.score(X_scaled, y)
r2_rf = rf_regressor.score(X_scaled, y)

# Print R-squared values for each model
print("R-squared for Multiple Linear Regression Model:", r2_mlr)
print("R-squared for Decision Tree Regression Model:", r2_tr)
print("R-squared for Random Forest Regression Model:", r2_rf)



#%%
# Assuming you have already defined and preprocessed the 'X' and 'y' variables

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
last_july_data = X.iloc[-1:].copy()  # Extract the last line's data for July
last_july_data_scaled = scaler.transform(last_july_data)  # Scale the last line's data

pred_mlr_july = mlr.predict(last_july_data_scaled)
pred_tr_july = tr_regressor.predict(last_july_data_scaled)
pred_rf_july = rf_regressor.predict(last_july_data_scaled)

# Print the predicted prices for phosphates in July for each model
print("Predicted Prices for Phosphates in July:")
print("Multiple Linear Regression Model:", pred_mlr_july[0])
print("Decision Tree Regression Model:", pred_tr_july[0])
print("Random Forest Regression Model:", pred_rf_july[0])

# Assuming the actual phosphate prices for July 2023 are available in 'july_actual_price'
july_actual_price = 342.5

# Calculate R-squared values for each model
r2_mlr_july = mlr.score(X_scaled, y)
r2_tr_july = tr_regressor.score(X_scaled, y)
r2_rf_july = rf_regressor.score(X_scaled, y)

# Print R-squared values for July for each model
print("R-squared for Multiple Linear Regression Model in July:", r2_mlr_july)
print("R-squared for Decision Tree Regression Model in July:", r2_tr_july)
print("R-squared for Random Forest Regression Model in July:", r2_rf_july)






#%% PAST YEAR 
# Step 1: Prepare Data for June 2022 and Make Predictions
# Assuming you have the data for June 2022 in a DataFrame named "df_june_2022"

# Scale the data for June 2022 using the previously defined scaler (scaler_june)
df_june_2022= X.iloc[:-12].copy() 



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

            
            
            print("------------------------------------------------------")
        else:
            print(f"Data for {month} is not available. Skipping predictions for this month.")
    else:
        print(f"Column '{column_name}' is missing in the DataFrame. Skipping predictions for the month of {month}.")
        
        
        
#%%
#PLOT
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the "results.csv" file
data = pd.read_csv('RESULTS.csv')

# Step 2: Create a line plot
months = ['Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22', 'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23']
predicted_mlr = data['Predicted_MLR']
predicted_tr = data['Predicted_TR']
predicted_rf = data['Predicted_RF']
actual_price = data['Actual_Price']

plt.figure(figsize=(15, 11))

# Plot predicted data
plt.plot(months, predicted_mlr, label='Predicted MLR', marker='o')
plt.plot(months, predicted_tr, label='Predicted TR', marker='o')
plt.plot(months, predicted_rf, label='Predicted RF', marker='o')

# Plot actual data
plt.plot(months, actual_price, label='Actual Price', marker='o')

# Step 3: Label the axes and add a legend
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('Comparison of Predicted and Actual Prices for Phosphate')
plt.legend()
plt.xticks(rotation=45)

# Show the plot
plt.grid()
plt.tight_layout()
plt.show()

# %% NEXT YEAR 
# Step 1: Prepare Data for June 2022 and Make Predictions
# Assuming you have the data for June 2022 in a DataFrame named "df_june_2022"
df_june_2022 = X.copy() 
scaler = StandardScaler()
scaler.fit(X)

# Now the scaler is fitted, and you can use the transform method
data_june_2022_scaled = scaler.transform(df_june_2022)
# Make predictions for June 2022 using the trained models (mlr, tr_regressor, rf_regressor)
pred_mlr_june_2022 = mlr.predict(data_june_2022_scaled)
pred_tr_june_2022 = tr_regressor.predict(data_june_2022_scaled)
pred_rf_june_2022 = rf_regressor.predict(data_june_2022_scaled)

# Step 2: Create "df_with_june_predicted" DataFrame
# Create a new DataFrame to store the predicted values for each month
df_nextyear_predicted = df_june_2022.copy()

# Define the list of months in the order you want to predict (from July 2022 to June 2023)
months_to_predict = ['July 2023', 'August 2023', 'September 2023', 'October 2023', 'November 2023', 'December 2023',
                     'January 2024', 'February 2024', 'March 2024', 'April 2024', 'May 2024', 'June 2024']

# Find the rows where any of the columns have 'True' as the value
true_row_index = df_nextyear_predicted.any(axis=1)

# Extract the month names for the rows with True values
df_nextyear_predicted['Months'] = [' '.join(month for month in months_to_predict if row[f'Mois_{month.split()[0]}']) for _, row in df_nextyear_predicted.iterrows()]

# Filter the DataFrame to keep only the rows with True values
df_nextyear_predicted = df_nextyear_predicted[true_row_index]

# Add the predicted prices for June 2022 to the DataFrame
df_nextyear_predicted['Predicted Price MLR'] = pred_mlr_june_2022[true_row_index]
df_nextyear_predicted['Predicted Price TR'] = pred_tr_june_2022[true_row_index]
df_nextyear_predicted['Predicted Price RF'] = pred_rf_june_2022[true_row_index]
# Step 3: Continue Predictions for Subsequent Months
# Now you can proceed with the loop to make predictions for the remaining months
# Initialize empty lists to store predicted prices for each model
predicted_prices_mlr = []
predicted_prices_tr = []
predicted_prices_rf = []

# Initialize the LabelEncoder object
from sklearn.preprocessing import StandardScaler, LabelEncoder

label_encoder = LabelEncoder()
# Loop through each month and make predictions
for month in months_to_predict:
    print("Current Month:", month)
    print("df_nextyear_predicted:")
    print(df_nextyear_predicted)

   # Prepare input data for the current month
    column_name = 'Mois_' + month.capitalize()
    if column_name in df_nextyear_predicted.columns:

        # Replace True/False values with the corresponding month names
        df_nextyear_predicted[column_name] = df_nextyear_predicted[column_name].replace({True: month, False: ''})

        current_month_data = df_nextyear_predicted[df_nextyear_predicted[column_name] == month]

        # Check if the DataFrame for the current month is empty
        if not current_month_data.empty:
                   df_nextyear_predicted[column_name] = df_nextyear_predicted[column_name].replace({True: month, False: ''})

        current_month_data = df_nextyear_predicted[df_nextyear_predicted[column_name] == month]

        # Check if the DataFrame for the current month is empty
        if not current_month_data.empty:
            # Encode the "Month" column using LabelEncoder
            current_month_data['Month'] = label_encoder.transform([month_name_to_value[month]] * len(current_month_data))

            # Add a new "Month" column to the DataFrame and set it to the month's name
            df_nextyear_predicted['Month'] = month

            # Assuming 'numeric_columns' is a list of numerical column names
            numeric_columns = ['Phosphate Price (Dollars américains par tonne métrique)', 'Diesel Price (Dollars US par gallon)', 'Phosphate ROC', 'Diesel ROC', 'Phosphate / Diesel Price Ratio']

            # Scale numeric data for the current month
            current_month_data_scaled = current_month_data.copy()
            current_month_data_scaled[numeric_columns] = scaler.transform(current_month_data[numeric_columns])
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
#%%
# Assuming you have the data for June 2022 in a DataFrame named "df_june_2022"
df_june_2022 = X.copy() 
scaler = StandardScaler()
scaler.fit(X)

# Now the scaler is fitted, and you can use the transform method
data_june_2022_scaled = scaler.transform(df_june_2022)
# Make predictions for June 2022 using the trained models (mlr, tr_regressor, rf_regressor)
pred_mlr_june_2022 = mlr.predict(data_june_2022_scaled)
pred_tr_june_2022 = tr_regressor.predict(data_june_2022_scaled)
pred_rf_june_2022 = rf_regressor.predict(data_june_2022_scaled)

# Step 2: Create "df_with_june_predicted" DataFrame
# Create a new DataFrame to store the predicted values for each month
df_nextyear_predicted = df_june_2022.copy()

# Print only the specified columns from df_nextyear_predicted
selected_columns = ['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio',
                    'Months','Predicted Price MLR', 'Predicted Price TR', 'Predicted Price RF']

selected_data = df_nextyear_predicted.loc[:, selected_columns]

# Save the selected columns to a CSV file
selected_data.to_csv('output_nextyear.csv', index=False)


#%%
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

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
df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].str.replace(',', '.')  # Replace commas with dots



# Filter the data for all rows before June 2023
historical_data = df[df['Mois'] <= 'juin 2023']

# X (independent variables) and y (target variable)
X = df[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
y = df['Phosphate Price (Dollars américains par tonne métrique)']



# Perform one-hot encoding on the "Mois" column
X = pd.get_dummies(X, columns=["Mois"])

# Create an input DataFrame for June 2023 data using historical data
input_data = historical_data[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
input_data = {
    'Diesel Price (Dollars US per gallon)': 2.5,
    'Phosphate / Diesel Price Ratio': 500.5,
}
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on historical data
scaler.fit(historical_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']])
mlr = LinearRegression()
mlr.fit(historical_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']],
        historical_data['Phosphate Price (Dollars américains par tonne métrique)'])

tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(historical_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']],
                 historical_data['Phosphate Price (Dollars américains par tonne métrique)'])

rf_regressor = RandomForestRegressor(n_estimators=28, random_state=0)
rf_regressor.fit(historical_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']],
                 historical_data['Phosphate Price (Dollars américains par tonne métrique)'])
# Define the number of iterations (months to predict)
num_iterations = 12

# Loop through each iteration to predict prices for the next 12 months
for i in range(num_iterations):
    print("Iteration:", i+1)
    
    # Create an input DataFrame for June 2023 data using historical data
    input_data = {
        'Diesel Price (Dollars US par gallon)': 2.59,
        'Phosphate / Diesel Price Ratio': 450.426,
    }

    # Prepare input data for prediction
    input_df = pd.DataFrame(input_data, index=[0])

    # Make sure the column names match the ones used during training
    input_df = input_df[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]

    # Transform the input data using the trained scaler
    input_scaled = scaler.transform(input_df)

    # Predict the phosphate price for the next month using the trained models
    pred_mlr_next_month = mlr.predict(input_scaled)
    pred_tr_next_month = tr_regressor.predict(input_scaled)
    pred_rf_next_month = rf_regressor.predict(input_scaled)
    
    
    # Print the predicted prices for the next month for each model
    print("Predicted Prices for month num:" , i+1)
    print("Multiple Linear Regression Model:", pred_mlr_next_month[0])
    print("Decision Tree Regression Model:", pred_tr_next_month[0])
    print("Random Forest Regression Model:", pred_rf_next_month[0])
    
    
    initial_value = df['Phosphate Price (Dollars américains par tonne métrique)'].iloc[-1:].values.item()
    initial_ratio = df['Phosphate / Diesel Price Ratio'].iloc[-1:].values.item()
    initial_ratio_f = float(initial_ratio)
    # Make sure i+7 doesn't exceed 12
    if i + 7 <= 12:
        month_value = i + 7
    else:
        month_value = (i + 7) - 12

    # Store the predicted prices in the dataset
    new_row = {
    'Mois': pd.Timestamp.now().replace(month=month_value).strftime('%B %Y'),
    'Phosphate Price (Dollars américains par tonne métrique)': pred_rf_next_month[0],
    'Diesel Price (Dollars US par gallon)': 2.43,
    'Phosphate ROC': ((pred_rf_next_month[0] - initial_value) / initial_value) * 100,
    'Diesel ROC': 0.00, 
    'Phosphate / Diesel Price Ratio': pred_rf_next_month[0] / initial_ratio_f
}

    # Convert the dictionary to a DataFrame
    new_row_df = pd.DataFrame(new_row, index=[0])

    # Concatenate the new row DataFrame with the existing DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)

    
    
    print("Predicted prices for", pd.Timestamp.now().replace(month=month_value).strftime('%B %Y'), "added to the dataset.")
    print("-------------------------------")

# Save the updated DataFrame with predicted prices
df.to_csv('phosphate36_predicted.csv', index=False)


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

df = pd.read_csv('C:\\Users\\asus\\Desktop\\intern\\predictionpricephosphates\\phosphate36.csv')

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
#%%

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
#%%
# Convert month names to numerical values
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Map the month names
df['Month'] = df['Month'].apply(lambda x: month_mapping[re.search(r'[a-zA-Zéû]+', str(x)).group()] if pd.notnull(x) else x)
print(df['Month'])
#%%
scalar = MinMaxScaler()
data_training = df.iloc[:-100].copy()
data_training

data_training_scaled = scalar.fit_transform(data_training)
print(data_training_scaled.shape)
data_training_scaled
data_testing = df.copy()
data_testing
X_train = []
y_train = []

for i in range(60, data_training.shape[0]):
    X_train.append(data_training_scaled[i-60: i])
    y_train.append(data_training_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape, y_train.shape

# Define the input shape for LSTM
input_shape = (X_train.shape[1], X_train.shape[2])  # (num_timesteps, num_features)

regressor = Sequential()

regressor.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units = 120, activation = 'relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units = 1))
regressor.summary()

#%%
# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)
regressor.fit(X_train, y_train, epochs=50, batch_size = 64)
#%%

# Evaluate the model on the test set
test_loss = regressor.evaluate(X_train, y_train)
print(f"Test Loss: {test_loss}")

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
X_test, y_test = np.array(X_test), np.array(y_test)

#%%
X_test, y_test = [], []
print("X_test shape:")
for i in range(60,test_scaled.shape[0]):
    X_test.append(test_scaled[i-60:i])
    print("X_test shape:")
    print(i)
    y_test.append(test_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
# Print shapes for debugging
print("X_test shape:", X_test.shape)
print("Expected input shape for model:", regressor.layers[0].input_shape)

#%%
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Assuming data_testing is your DataFrame
data_testing = df.copy()

# Convert non-numeric month names to numeric values (example mapping)
month_mapping = {
    'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
    'juillet': 7, 'août': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
}

data_testing['Month'] = data_testing['Month'].map(month_mapping)

# Extract numeric features for scaling
numeric_features = ['Phosphate Price (Dollars américains par tonne métrique)','Diesel Price (Dollars US par gallon)','Phosphate ROC','Diesel ROC','Phosphate / Diesel Price Ratio']  # Include other features here

# Initialize the scaler and fit it on the numeric features
scalar = MinMaxScaler()
scalar.fit(data_testing[numeric_features])

# Scale the testing data
data_testing_scaled = scalar.transform(data_testing[numeric_features])


# If 'Month' values are numeric strings
matching_indices = df[(df['Month'] == 'juin') & (df['Year'] == '2023')].index

print(matching_indices)
# Choose a specific index for prediction (replace 0 with the desired index)
specific_date_index = matching_indices[0]

# Prepare the input sequence for prediction
input_sequence = data_testing_scaled[specific_date_index - 60 : specific_date_index]

input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
# Make the prediction
predicted_scaled_price = regressor.predict(input_sequence)

# Inverse transform the predicted scaled price to the original scale
predicted_price = scalar.inverse_transform([[predicted_scaled_price, 0, 0, 0, 0, 0]])[0, 0]

print(f"Predicted Phosphate Price for Specific Date (Index {specific_date_index}):", predicted_price)


#%%
#NEEEEEEW JULYYYYYY
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
#%%

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
#%%
# Convert month names to numerical values
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Map the month names
df['Month'] = df['Month'].apply(lambda x: month_mapping[re.search(r'[a-zA-Zéû]+', str(x)).group()] if pd.notnull(x) else x)
print(df['Month'])
#%%
# X (independent variables) and y (target variable)
X = df[['Month','Year','Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
y = df['Phosphate Price (Dollars américains par tonne métrique)']
# Perform one-hot encoding on the "Mois" column
X = pd.get_dummies(X, columns=["Month"])
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
print("Multiple Linear Regression Model Score:", round(mlr_score * 100, 2))
print("Decision Tree Regression Model Score:", round(tr_regressor_score * 100, 2))
print("Random Forest Regression Model Score:", round(rf_regressor_score * 100, 2))

#%%
# Create input features for July 2023
input_features_july = pd.DataFrame({
    'Year': [2023],
    'Diesel Price (Dollars US par gallon)': [2.43],  # Replace with the actual diesel price for July 2023
    'Phosphate / Diesel Price Ratio': [141.6530],
    'Month_1': [0],
    'Month_2': [0],
    'Month_3': [0],
    'Month_4': [0],
    'Month_5': [0],
    'Month_6': [0],
    'Month_7': [1],
    'Month_8': [0],
    'Month_9': [0],
    'Month_10': [0],
    'Month_11': [0],
    'Month_12': [0]
})


# Initialize the StandardScaler
scaler = StandardScaler()
# Scale the data
input_features_july_scaled = scaler.fit_transform(input_features_july)
# Predict phosphate price for July 2023 using each model
pred_mlr_july = mlr.predict(input_features_july_scaled)
pred_tr_july = tr_regressor.predict(input_features_july_scaled)
pred_rf_july = rf_regressor.predict(input_features_july_scaled)

print("Predicted Phosphate Price for July 2023 (MLR):", pred_mlr_july[0])
print("Predicted Phosphate Price for July 2023 (Decision Tree):", pred_tr_july[0])
print("Predicted Phosphate Price for July 2023 (Random Forest):", pred_rf_july[0])




# %% ARIMAAAAAAAAAAAAAAA


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
df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].replace(',', '.').astype(float)

# Split the "Mois" column into "Month" and "Year" columns
df[["Month", "Year"]] = df["Mois"].str.split(" ", n=1, expand=True)

# Drop the original "Mois" column
df.drop(columns=["Mois"], inplace=True)

# Display the resulting DataFrame
print(df)
#%%

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
#%%
# Convert month names to numerical values
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Map the month names
df['Month'] = df['Month'].apply(lambda x: month_mapping[re.search(r'[a-zA-Zéû]+', str(x)).group()] if pd.notnull(x) else x)
print(df['Month'])
#%%
# X (independent variables) and y (target variable)
X = df[['Month','Year','Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
y = df['Phosphate Price (Dollars américains par tonne métrique)']
# Perform one-hot encoding on the "Mois" column
X = pd.get_dummies(X, columns=["Month"])
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

# Train ARIMA model
from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(y_train, order=(3, 2, 3))  # Adjust the order (p,d,q) as needed
arima_model_fit = arima_model.fit()

# Make predictions with ARIMA
arima_forecast = arima_model_fit.forecast(steps=len(y_test))

# Print the predicted prices for phosphates in June for each model
print("Multiple Linear Regression Model Score:", round(mlr_score * 100, 2))
print("Decision Tree Regression Model Score:", round(tr_regressor_score * 100, 2))
print("Random Forest Regression Model Score:", round(rf_regressor_score * 100, 2))
print("ARIMA Model RMSE:", np.sqrt(np.mean((arima_forecast - y_test)**2)))

# Create input features for July 2023
# ... (your existing code for creating input_features_july and scaling it)
# Create input features for July 2023
input_features_july = pd.DataFrame({
    'Year': [2023],
    'Diesel Price (Dollars US par gallon)': [2.43],  # Replace with the actual diesel price for July 2023
    'Phosphate / Diesel Price Ratio': [141.6530],
    'Month_1': [0],
    'Month_2': [0],
    'Month_3': [0],
    'Month_4': [0],
    'Month_5': [0],
    'Month_6': [0],
    'Month_7': [1],
    'Month_8': [0],
    'Month_9': [0],
    'Month_10': [0],
    'Month_11': [0],
    'Month_12': [0]
})


# Initialize the StandardScaler
scaler = StandardScaler()
# Scale the data
input_features_july_scaled = scaler.fit_transform(input_features_july)

# Predict phosphate price for July 2023 using each model
pred_mlr_july = mlr.predict(input_features_july_scaled)
pred_tr_july = tr_regressor.predict(input_features_july_scaled)
pred_rf_july = rf_regressor.predict(input_features_july_scaled)
pred_arima_july = arima_model_fit.forecast(steps=1).iloc[0]  # Forecast using ARIMA

print("Predicted Phosphate Price for July 2023 (MLR):", pred_mlr_july[0])
print("Predicted Phosphate Price for July 2023 (Decision Tree):", pred_tr_july[0])
print("Predicted Phosphate Price for July 2023 (Random Forest):", pred_rf_july[0])
print("Predicted Phosphate Price for July 2023 (ARIMA):", pred_arima_july)

