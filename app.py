from flask import Flask ,render_template,jsonify,url_for,flash,redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField,BooleanField
from wtforms.validators import DataRequired,length,Email,Regexp,EqualTo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import json
import requests
from email_validator import validate_email
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import datetime
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

app=Flask(__name__)
app.config['SECRET_KEY']='293b8cad24d6f3600ee5a2d67dc1c3efc48f19d0f84b5c2c229e997409a06d20'



@app.route("/index")
def index():
    df = pd.read_csv("prix.csv")
    df = pd.DataFrame(df, columns=['Mois', 'Prix', 'Variation'])
    actual_price = float(df['Prix'].iloc[-1])  # Actual price from the last line of 'prix.csv'
    return render_template('public/index.html', actual_price=actual_price)
   



class inscrireform(FlaskForm):
    fname = StringField('First Name',validators=[DataRequired(),length(min=2,max=25)])
    lname = StringField('Last Name',validators=[DataRequired(),length(min=2,max=25)])
    username = StringField('Username',validators=[DataRequired(),length(min=2,max=25)])
    email = StringField("Email",validators=[DataRequired(),Email()])
    password = PasswordField('Password',validators=[DataRequired(),Regexp("^(?=.*[A-Z])(?=.*[a-z])(?=.*[@$!%*?&_])[A-Za-z\d@$!%*?&_]{8,32}$")])
    confirm_password = PasswordField('Confirm_password',validators=[DataRequired(),EqualTo('password')])
    submit = SubmitField("Sign Up")

class loginform(FlaskForm):
    
    email = StringField("Email",validators=[DataRequired(),Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    remember = BooleanField('Remember Me') 
    submit = SubmitField("Login")


@app.route("/login",methods=["GET","POST"])
def login():
    form= loginform()
    if form.validate_on_submit():
        if form.email.data =='ouyoussmeryem@gmail.com' and form.password.data == "PASS??word123":
            flash("You have been logged in !!","success")
            return redirect("/index")
        else :
            flash("Login Unsuccessful , please check credentials","danger")
    return render_template("public/login.html",form=form)



@app.route("/inscrire",methods=["GET","POST"])
def inscrire():
    form= inscrireform()
    if form.validate_on_submit():
        flash(f"Account created successfully for {form.username.data}","success")
        return redirect("/index")
    return render_template("public/inscription.html",form=form)


   


# Collect and clean the data
"""df = pd.read_csv("prix.csv")
numeric_columns = df.columns.drop('Mois')
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df = pd.DataFrame(df, columns=['Mois', 'Prix', 'Variation'])"""


"""from sklearn.preprocessing import OneHotEncoder

# Convert month names to one-hot encoded vectors
encoder = OneHotEncoder(sparse=False)
month_encoded = encoder.fit_transform(df[['Mois']])
month_encoded_df = pd.DataFrame(month_encoded, columns=encoder.categories_[0])
df = pd.concat([df.drop(columns=['Mois']), month_encoded_df], axis=1)"""


"""
df['Variation'] = df['Variation'].str.replace('-', '')  # Remove the hyphen
df['Variation'] = df['Variation'].str.replace(',', '.')  # Replace commas with decimal points
df['Variation'] = df['Variation'].str.replace(' %', '')  # Remove percentage signs
df['Variation'] = df['Variation'].str.replace('"', '')  # Remove percentage signs
df['Variation'] = pd.to_numeric(df['Variation'])  # Convert the values to numeric format

df['Prix'] = df['Prix'].str.replace(',', '.')  # Replace commas with decimal points
   
df['Mois'] = df['Mois'].str.replace('.', '')  # Remove percentage signs


df = df.dropna()
df.head(363)
df.dtypes
df.describe()

df.dtypes
for x in df:
    if df[x].dtypes == "int64":
        df[x] = df[x].astype(float)
        print (df[x].dtypes)"""



# Read the CSV file
df = pd.read_csv("prix.csv")

# Clean the data and convert columns to the appropriate data types

# Replace commas with decimal points in the 'Prix' column
df['Prix'] = df['Prix'].str.replace(',', '.').astype(float)

# Remove unnecessary characters and convert 'Variation' column to numeric format
df['Variation'] = df['Variation'].str.replace('-', '').str.replace(',', '.').str.replace('%', '').astype(float)

window_size = 3  # Adjust the window size as needed

df['Rolling_Average'] = df['Variation'].rolling(window=window_size).mean()

# Drop missing values
df = df.dropna()

@app.route("/prediction",methods=['POST'])
def prediction():
    
    return render_template("public/phosphatesRF.html")

    
@app.route("/phosphates")
def phosphatesRF():
    
    # Create and train the model
    # Train the model
    #df = df.select_dtypes()
    X = df[['Variation', 'Rolling_Average']]
    y = df['Prix']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Train a random forest regressor model
    regressor = RandomForestRegressor(n_estimators=1000, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    
    # Optional: Predict the price for June 2023 using the trained model
    #variation_2023 = df['Variation'].iloc[-1]  # Variation value of the last line
    #input_data = pd.DataFrame({'Variation': [variation_2023]})
    variation_2023 = df['Variation'].iloc[-1]  # Variation value of the last line
    rolling_average_2023 = df['Rolling_Average'].iloc[-1]  # Rolling average of the last line
    input_data = pd.DataFrame({'Variation': [variation_2023], 'Rolling_Average': [rolling_average_2023]})
    predicted_price = regressor.predict(input_data)[0]   
    actual_price = df['Prix'].iloc[-1]
    difference = actual_price - predicted_price
    accuracy = (1 - abs(difference / actual_price)) * 100

    print('Predicted Price for June 2023:', predicted_price)
    print('Actual Price for June 2023:', actual_price)
    print('Difference:', difference)
    print('Accuracy:', accuracy)
    
    """df=df.fillna(df.mean())
    X = df[['Variation']]  # Select the 'Variation' column as input feature
    y = df['Prix']  # Select the 'Prix' column as the target variable
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    regressor.fit(X_train, y_train)"""
    
    
    
    """model = RandomForestRegressor() n,k
    model.fit(X, y)  # Train the model

    # Prepare input data for prediction
    variation_2023 = df['Variation'].iloc[-2]  # Variation value of the bfr last line
    input_data = pd.DataFrame({'Variation': [variation_2023]})

    # Predict the price for June 2023
    predicted_price = model.predict(input_data)[0]
    actual_price = float(df['Prix'].iloc[-1])  # Actual price from the last line of 'prix.csv'
    difference = actual_price - predicted_price
    accuracy = (1 - abs((actual_price - predicted_price) / actual_price)) * 100"""

    """y_pred = regressor.predict(X_test)
    df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
    print(y_test,y_pred)
    
    
    from sklearn import metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')"""
    
    
    
    current_date = datetime.now()
    prediction_date = current_date.strftime("%B")
    
    return render_template("public/phosphatesRF.html", predicted_price=predicted_price, actual_price=actual_price, difference=difference, accuracy=accuracy, prediction_date=prediction_date)

@app.route("/nextmonth")
def nextmonth():
    
    # Create and train the model
    # Train the model
    X = df[['Variation']]  # Select the 'Variation' column as input feature
    y = df['Prix']  # Select the 'Prix' column as the target variable
    model = RandomForestRegressor()
    model.fit(X, y)  # Train the model

    # Prepare input data for prediction
    variation_nextmonth = df['Variation'].iloc[-1]  # Variation value of the last line
    #variation_2023 = 0.0  
    input_data = pd.DataFrame({'Variation': [variation_nextmonth]})

    # Predict the price for June 2023
    next_month = model.predict(input_data)[0]
    


    current_date = datetime.now()
    prediction_date = (current_date + timedelta(days=30)).strftime("%B")  # Get the next month

    
    return render_template("public/nextmonth.html", next_month=next_month, variation_nextmonth=variation_nextmonth, prediction_date=prediction_date)

"""@app.route("/Neural")
def neural():
   
    # Split the dataset into features (X) and target (y)
    X = df[['Mois', 'Variation']].values
    y = df['Prix'].values

    # Normalize the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the ANN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=0)
    print('Mean Squared Error:', loss)

    # Make predictions
    new_data = np.array([[0.8, 0.05], [0.9, 0.02]])  # Example input for prediction
    scaled_data = scaler.transform(new_data)
    predictions = model.predict(scaled_data)

    # Print the predicted prices
    for i, pred in enumerate(predictions):
        print('Prediction for data point', i+1, ':', pred)"""

"""@app.route("/nextmonth")
def nextmonth():

    

    # Create and train the model
    X = df[['Variation']]  # Select the 'Variation' column as input feature
    y = df['Prix']  # Select the 'Prix' column as the target variable
    model = RandomForestRegressor()
    model.fit(X, y)  # Train the model


    # Set the date column as the index (assuming it's present in the dataset)
    # Set the date column as the index (assuming it's present in the dataset)
    #df['Mois'] = pd.to_datetime(df['Mois'], format="%B%Y")
    #df.set_index('Mois', inplace=True)
    # Set the date column as the index (assuming it's present in the dataset)
    df['Mois'] = pd.to_datetime(df['Mois'], errors='coerce')
    df.dropna(subset=['Mois'], inplace=True)  # Remove rows with invalid dates
    df.set_index('Mois', inplace=True)

    # Create the SARIMA model
    model = SARIMAX(df['Variation'], order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))

    # Fit the model to the data
    model_fit = model.fit()

    # Forecast the variation for the next month
    next_month_forecast = model_fit.forecast(steps=1).values

    # Get the predicted variation for the next month
    variation_next_month = next_month_forecast[0]  # Variation value for the next month

    input_data = pd.DataFrame({'Variation': [variation_next_month]})

    # Predict the price for the next month
    #nextmonth_price = model.predict(input_data)[0]
    nextmonth_price = model_fit.get_prediction(input_data.index[-1], dynamic=False)
    nextmonth_price = nextmonth_price.predicted_mean[0]


    current_date = datetime.now()
    prediction_date = (current_date + timedelta(days=30)).strftime("%B")  # Get the next month

    return render_template("public/nextmonth.html", nextmonth_price=nextmonth_price, prediction_date=prediction_date, variation_next_month=variation_next_month)"""


if __name__ == '__main__':
    app.run(debug=True)

    



import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Load the dataset
df = pd.read_csv("phosphate36.csv", skiprows=1)

# Modify the column names to match the actual column names in the dataset
df.columns = ['Mois',
              'Phosphate Price (Dollars américains par tonne métrique)',
              'Diesel Price (Dollars US par gallon)',
              'Phosphate ROC',
              'Diesel ROC',
              'Phosphate / Diesel Price Ratio']


# Convert columns to numeric values
df['Phosphate Price (Dollars américains par tonne métrique)'] = df['Phosphate Price (Dollars américains par tonne métrique)'].str.replace(',', '.').astype(float)
df['Diesel Price (Dollars US par gallon)'] = df['Diesel Price (Dollars US par gallon)'].str.replace(',', '.').astype(float)
df['Phosphate ROC'] = df['Phosphate ROC'].replace('-', '0').str.replace(',', '.').str.rstrip('%').astype(float)
df['Diesel ROC'] = df['Diesel ROC'].replace('-', '0').str.replace(',', '.').str.rstrip('%').astype(float)

# Assuming df is your DataFrame
df['Phosphate / Diesel Price Ratio'] = pd.to_numeric(df['Phosphate / Diesel Price Ratio'], errors='coerce')

# Now, apply the format_with_commas function
def format_with_commas(x):
    return '{:,.4f}'.format(x)

df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].map(format_with_commas)

df['Phosphate / Diesel Price Ratio'] = df['Phosphate / Diesel Price Ratio'].str.replace(',', '.').astype(float)

#df = df.dropna()

# Create 'X' and 'y' (independent and target variables)
X = df[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
y = df['Phosphate Price (Dollars américains par tonne métrique)']

# Define the month mapping and map the 'Mois' column
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

X['Mois'] = X['Mois'].apply(lambda x: month_mapping[re.search(r'[a-zA-Zéû]+', str(x)).group()] if pd.notnull(x) else x)

# Perform one-hot encoding for the 'Mois' column
categorical_features = ['Mois']
numeric_features = ['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Transform the data using the preprocessor
X_transformed = preprocessor.fit_transform(X)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.26, random_state=0)

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
    'Diesel Price (Dollars US par gallon)': [2.09],  # Replace with the desired value
    'Phosphate / Diesel Price Ratio': [210.426]  # Replace with the desired value
})

predicted_price_tr = tr_regressor.predict(new_data)
print("Predicted Price for June 2023 (Decision Tree Regressor):", predicted_price_tr[0])

# Print model scores
print("Multiple Linear Regression Model Score:", round(mlr_score * 100, 2))
print("Decision Tree Regression Model Score:", round(tr_regressor_score * 100, 2))
print("Random Forest Regression Model Score:", round(rf_regressor_score * 100, 2))

# Assuming that the 'June' data is available in the DataFrame 'df', you can filter it like this:
june_data = df[df['Mois'] == 'June']

# Extract the independent variables for the June data
X_june = june_data[['Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]

# Scale the data using the same StandardScaler used during training
X_june_scaled = scaler.transform(X_june)

# Predict using the trained models
pred_mlr_june = mlr.predict(X_june_scaled)
pred_tr_june = tr_regressor.predict(X_june_scaled)
pred_rf_june = rf_regressor.predict(X_june_scaled)

# Print the predicted prices for phosphates in June for each model
print("Predicted Prices for Phosphates in June:")
print("Multiple Linear Regression Model:", pred_mlr_june[0])
print("Decision Tree Regression Model:", pred_tr_june[0])
print("Random Forest Regression Model:", pred_rf_june[0])

