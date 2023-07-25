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

df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].apply(lambda x: re.sub('[^0-9.]', '', x)).astype(float)

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

#%%
# Predict the price for a specific month
new_data = pd.DataFrame({
    'Phosphate Price (Dollars américains par tonne métrique)': [60.00],  # Replace with the desired value
    'Diesel Price (Dollars US par gallon)': [2.09],  # Replace with the desired value
    'Phosphate / Diesel Price Ratio': [210.426]  # Replace with the desired value
})

predicted_price = tr_regressor.predict(new_data)

print("Predicted Price:", predicted_price[0])

print("Multiple Linear Regression Model Score:", round(mlr_score * 100, 2))
print("Decision Tree Regression Model Score:", round(tr_regressor_score * 100, 2))
print("Random Forest Regression Model Score:", round(rf_regressor_score * 100, 2))

# %%