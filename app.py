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
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
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

# Assuming the actual phosphate prices for June 2023 are available in 'june_data'
actual_prices = df['Phosphate Price (Dollars américains par tonne métrique)'].iloc[-1]

scaler = StandardScaler()
    # Predict the variations and rolling averages for June 2023
historical_data = df[df['Mois'] < 'juin 2023']  # Filter the data for all rows before June 2023

    # Create an input DataFrame for June 2023 data using historical data
input_data = historical_data[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]


    # Assuming you have already defined and preprocessed the 'x_train' and 'y_train' variables
X = pd.get_dummies(X, columns=["Mois"])

    # Initialize the StandardScaler

    # Scale the data
X_scaled = scaler.fit_transform(X)

    # Train the models on the entire dataset
mlr = LinearRegression()
mlr.fit(X_scaled, y)

tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(X_scaled, y)

rf_regressor = RandomForestRegressor(n_estimators=28, random_state=0)
rf_regressor.fit(X_scaled, y)


@app.route("/predprevmonth",methods=['POST'])
def predprevmonth():
    #JUNE 

    current_date = datetime.now()
    prediction_date = (current_date - timedelta(days=60)).strftime("%B")  # Get the next month

    
    # Predict the last line's phosphate price for each model
    last_june_data = X.iloc[-1:].copy()  # Extract the last line's data for June
    last_june_data_scaled = scaler.transform(last_june_data)  # Scale the last line's data

    pred_mlr_june = mlr.predict(last_june_data_scaled)
    pred_tr_june = tr_regressor.predict(last_june_data_scaled)
    pred_rf_june = rf_regressor.predict(last_june_data_scaled)

    accuracy_mlr = (1 - abs((pred_mlr_june[0]-actual_prices) / actual_prices)) * 100

    accuracy_tr = (1 - abs((pred_tr_june[0]-actual_prices) / actual_prices)) * 100

    accuracy_rf = (1 - abs((pred_rf_june[0]-actual_prices) / actual_prices)) * 100
    return render_template("public/prevmonth.html", prediction_date=prediction_date,actual_prices=actual_prices, accuracy_mlr=accuracy_mlr, accuracy_tr=accuracy_tr, accuracy_rf=accuracy_rf)


@app.route("/predpastyear",methods=['POST'])
def predpastyear():
    
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

                # Calculate the accuracy of the predictions for each model
                actual_price = current_month_data['Phosphate Price (Dollars américains par tonne métrique)'].values[0]
                accuracy_mlr = 100 * (1 - abs((pred_mlr_month[0] - actual_price) / actual_price))
                accuracy_tr = 100 * (1 - abs((pred_tr_month[0] - actual_price) / actual_price))
                accuracy_rf = 100 * (1 - abs((pred_rf_month[0] - actual_price) / actual_price))

                print("------------------------------------------------------")
            else:
                print("------------------------------------------------------")
        else:
                print("------------------------------------------------------")
            
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv("RESULTS.csv")

        # Create a new DataFrame containing only the 'Mois' and 'Actual_Price' columns
        past_year = df.copy()
        return render_template("public/predjune.html", actual_price=actual_price, accuracy_mlr=accuracy_mlr, accuracy_tr=accuracy_tr, accuracy_rf=accuracy_rf, past_year=past_year)


@app.route("/prednextmonth",methods=['POST'])
def prednextmonth():
    #JULY 
    last_month= X.iloc[-1:].copy() 

    # Predict the variations and rolling averages for June 2023
    historical_data = df[df['Mois'] <='juin 2023']  # Filter the data for all rows before June 2023


    # Create an input DataFrame for June 2023 data using historical data
    input_data = historical_data[['Mois', 'Diesel Price (Dollars US par gallon)', 'Phosphate / Diesel Price Ratio']]
    input_data = {
        'Diesel Price (Dollars US per gallon)': 2.5,
        'Phosphate / Diesel Price Ratio': 141.6530,
    }

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

    return render_template("public/predjune.html")

    
@app.route("/prednextyear")
def phosphatesRF():
 




    current_date = datetime.now()
    prediction_date = (current_date + timedelta(days=30)).strftime("%B")  # Get the next month

    
    return render_template("public/nextmonth.html")


if __name__ == '__main__':
    app.run(debug=True)

  