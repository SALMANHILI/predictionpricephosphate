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
df_june_2022= X.iloc[-12:].copy() 
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

# Define the number of iterations (months to predict)
num_iterations = 12

# Loop through each iteration to predict prices for the next 12 months
for i in range(num_iterations):
    print("Iteration:", i+1)
    

    
    # Load pre-trained models
    # Load pre-trained models
    mlr = LinearRegression()
    mlr.fit(historical_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']], historical_data['Phosphate Price (Dollars américains par tonne métrique)'])

    tr_regressor = DecisionTreeRegressor(random_state=0)
    tr_regressor.fit(historical_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']], historical_data['Phosphate Price (Dollars américains par tonne métrique)'])

    rf_regressor = RandomForestRegressor(n_estimators=28, random_state=0)
    rf_regressor.fit(historical_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']], historical_data['Phosphate Price (Dollars américains par tonne métrique)'])

        
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