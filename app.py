from flask import Flask, render_template, jsonify, url_for, flash, redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, length, Email, Regexp, EqualTo
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import json
import requests
from email_validator import validate_email
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from sklearn.metrics import explained_variance_score
import seaborn as sns
import re
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Response
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app=Flask(__name__)
app.config['SECRET_KEY']='293b8cad24d6f3600ee5a2d67dc1c3efc48f19d0f84b5c2c229e997409a06d20'
app.config['STATIC_FOLDER'] = 'static'


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


@app.route("/index")
def index():
    df = pd.read_csv("phosphate36.csv")
    df = pd.DataFrame(df, columns=['Mois','Phosphate Price (Dollars américains par tonne métrique)','Diesel Price (Dollars US par gallon)','Phosphate ROC','Diesel ROC','Phosphate/Diesel Price Ratio'])
    actual_price_str = df['Phosphate Price (Dollars américains par tonne métrique)'].iloc[-1]
    actual_price = float(actual_price_str.replace(',', '.'))
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









@app.route("/predpastyear",methods=['GET', 'POST'])
def predpastyear():
    
                
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


        actual_data = pd.read_csv('phosphate36.csv')['Phosphate Price (Dollars américains par tonne métrique)'].iloc[-12:].values
        # Round the predicted prices for MLR and Random Forest
        rounded_pred_mlr_future = pred_mlr_future.round(2)
        rounded_pred_rf_future = pred_rf_future.round(2)

        # Create the futurepredictions_df DataFrame
        futurepredictions_df = pd.DataFrame({
            'Month': future_input['Month'],
            'Year': future_input['Year'],
            'Predicted Phosphate Price (MLR)': rounded_pred_mlr_future,
            'Predicted Phosphate Price (Decision Tree)': pred_tr_future,
            'Predicted Phosphate Price (Random Forest)': rounded_pred_rf_future,
            'actual_data': actual_data
        })



        # Map the numerical month values back to month names
        #predictions_df['Month'] = predictions_df['Month'].map(month_mapping)
        # Define a dictionary to map numerical month values to month names
        reverse_month_mapping = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                                7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

        # Map the numerical month values back to month names
        futurepredictions_df['Month'] = futurepredictions_df['Month'].map(reverse_month_mapping)


        # Display the predicted prices for the next 12 months
        futurepredictions_df



        # Create the function to generate the plot as an image response

        futurepredictions_df.to_csv('predicted_prices_vf.csv', index=False)

        
        data = pd.read_csv('predicted_prices_vf.csv')

        # Combine 'Month' and 'Year' columns for the x-axis labels
        months_years = data['Month'] + ' ' + data['Year'].astype(str)

        # Extract data for plotting
        predicted_mlr = data['Predicted Phosphate Price (MLR)']
        predicted_tree = data['Predicted Phosphate Price (Decision Tree)']
        predicted_rf = data['Predicted Phosphate Price (Random Forest)']
        actual_data = df['Phosphate Price (Dollars américains par tonne métrique)'].iloc[-12:].values
        actual_data
        # Calculate the accuracy for each algorithm
        data['MLR Accuracy'] = (100 * (1 - abs(data['Predicted Phosphate Price (MLR)'] - actual_data) / actual_data)).round(2)
        data['Decision Tree Accuracy'] = (100 * (1 - abs(data['Predicted Phosphate Price (Decision Tree)'] - actual_data) / actual_data)).round(2)
        data['Random Forest Accuracy'] = (100 * (1 - abs(data['Predicted Phosphate Price (Random Forest)'] - actual_data) / actual_data)).round(2)

        # Add the accuracy columns to the futurepredictions_df DataFrame
        futurepredictions_df['MLR Accuracy'] = data['MLR Accuracy']
        futurepredictions_df['Decision Tree Accuracy'] = data['Decision Tree Accuracy']
        futurepredictions_df['Random Forest Accuracy'] = data['Random Forest Accuracy']

        # Create a line plot
        plt.figure(figsize=(10, 6))

        plt.plot(months_years, predicted_mlr, marker='o', label='MLR')
        plt.plot(months_years, predicted_tree, marker='o', label='Decision Tree')
        plt.plot(months_years, predicted_rf, marker='o', label='Random Forest')
        plt.plot(months_years, actual_data, marker='o', label='actual data')

        plt.xlabel('Month and Year')
        plt.ylabel('Predicted Phosphate Price')
        plt.title('Predicted Phosphate Prices by Model')
        plt.legend()
        plt.xticks(rotation=45)
        # Customize y-axis tick locations and labels
        plt.yticks(range(100, 401, 200))  # Adjust the range as needed

        plt.tight_layout()
        # Save the plot as a PNG file in the /static directory
        plot_filename = 'static/plot.png'
        plt.savefig(plot_filename)
        plot_to_image(months_years, pred_mlr_future, pred_tr_future, pred_rf_future, actual_data, plot_filename)

        return render_template("public/predpastyear.html", futurepredictions_df=futurepredictions_df, plot_filename=plot_filename)


def plot_to_image(months_years, pred_mlr_future, pred_tr_future, pred_rf_future, actual_data, filename):
    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # ... Your plot code here ...
            # Plotting the data
    ax.plot(months_years, pred_mlr_future, marker='o', label='MLR')
    ax.plot(months_years, pred_tr_future, marker='o', label='Decision Tree')
    ax.plot(months_years, pred_rf_future, marker='o', label='Random Forest')
    ax.plot(months_years, actual_data, marker='o', label='Actual Data')  # Adding actual data to the plot
            
    ax.set_xlabel('Month and Year')
    ax.set_ylabel('Predicted Phosphate Price')
    ax.set_title('Predicted Phosphate Prices by Model')
    ax.legend()
    ax.set_xticklabels(months_years, rotation=45)

    # Save the plot to a file
    fig.savefig(filename)




            




@app.route("/predpastmonth", methods=['GET', 'POST'])
def predpastmonth():
    #JULY 
    manual_input = pd.DataFrame({ 'Month':[7],
        'Year': [2023],
        'Diesel Price (Dollars US par gallon)': [2.43],  # Replace with the actual diesel price for July 2023
        'Diesel ROC':[3.67],
        'Phosphate / Diesel Price Ratio': [141.65]
    })


    input = scaler.transform(manual_input)
    # right


    # Predict phosphate price for July 2023 using each model
    pred_mlr_july = mlr.predict(input)
    pred_tr_july = tr_regressor.predict(input)
    pred_rf_july = rf_regressor.predict(input)

    pred_mlr_july[0]
    pred_tr_july[0]
    pred_rf_july[0]
    
    df = pd.read_csv("phosphate36.csv")
    df = pd.DataFrame(df, columns=['Mois','Phosphate Price (Dollars américains par tonne métrique)','Diesel Price (Dollars US par gallon)','Phosphate ROC','Diesel ROC','Phosphate/Diesel Price Ratio'])
    actual_price_str = df['Phosphate Price (Dollars américains par tonne métrique)'].iloc[-1]
    actual_price = float(actual_price_str.replace(',', '.'))
    abs_percent_error_mlr = abs((pred_mlr_july - actual_price) / actual_price) * 100
    abs_percent_error_tr = abs((pred_tr_july - actual_price) / actual_price) * 100
    abs_percent_error_rf = abs((pred_rf_july - actual_price) / actual_price) * 100

    # Calculate the accuracy for each model (100 - absolute percentage error)
    accuracy_mlr = 100 - abs_percent_error_mlr
    accuracy_tr = 100 - abs_percent_error_tr
    accuracy_rf = 100 - abs_percent_error_rf

    return render_template("public/predpastmonth.html", pred_mlr_july=pred_mlr_july, pred_tr_july=pred_tr_july, pred_rf_july=pred_rf_july, actual_price=actual_price, accuracy_mlr=accuracy_mlr, accuracy_tr=accuracy_tr, accuracy_rf=accuracy_rf)






@app.route("/prednextmonth", methods=['GET', 'POST'])
def prednextmonth():
    #AUGUST 
    manual_input = pd.DataFrame({ 'Month':[8],
        'Year': [2023],
        'Diesel Price (Dollars US par gallon)': [2.53],  # Replace with the actual diesel price for July 2023
        'Diesel ROC':[4.11],
        'Phosphate / Diesel Price Ratio': [135.3767]
    })


    input = scaler.transform(manual_input)
    # right
    # Predict phosphate price for July 2023 using each model
    pred_mlr_august = mlr.predict(input)

    pred_tr_august = tr_regressor.predict(input)
    pred_rf_august = rf_regressor.predict(input)

    pred_mlr_august[0]
    pred_tr_august[0]
    pred_rf_august[0]

    return render_template("public/prednextmonth.html",pred_mlr_august= pred_mlr_august,pred_tr_august= pred_tr_august,pred_rf_august= pred_rf_august)

    
@app.route("/prednextyear", methods=['GET', 'POST'])
def prednextyear():
 

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


    # Round the predicted prices for MLR and Random Forest
    rounded_pred_mlr_future = pred_mlr_future.round(2)
    rounded_pred_rf_future = pred_rf_future.round(2)

    # Create the futurepredictions_df DataFrame
    futurepredictionss_df = pd.DataFrame({
        'Month': future_input['Month'],
        'Year': future_input['Year'],
        'Predicted Phosphate Price (MLR)': rounded_pred_mlr_future,
        'Predicted Phosphate Price (Decision Tree)': pred_tr_future,
        'Predicted Phosphate Price (Random Forest)': rounded_pred_rf_future,
    })



    # Map the numerical month values back to month names
    #predictions_df['Month'] = predictions_df['Month'].map(month_mapping)
    # Define a dictionary to map numerical month values to month names
    reverse_month_mapping = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    # Map the numerical month values back to month names
    futurepredictionss_df['Month'] = futurepredictionss_df['Month'].map(reverse_month_mapping)


    # Display the predicted prices for the next 12 months



    # Create the function to generate the plot as an image response

    futurepredictionss_df.to_csv('predicted_prices_vfff.csv', index=False)

    
    data = pd.read_csv('predicted_prices_vfff.csv')

    # Combine 'Month' and 'Year' columns for the x-axis labels
    months_years = data['Month'] + ' ' + data['Year'].astype(str)

    # Extract data for plotting
    predicted_mlr = data['Predicted Phosphate Price (MLR)']
    predicted_tree = data['Predicted Phosphate Price (Decision Tree)']
    predicted_rf = data['Predicted Phosphate Price (Random Forest)']
    # Calculate the accuracy for each algorithm


    # Create a line plot
    plt.figure(figsize=(10, 6))

    plt.plot(months_years, predicted_mlr, marker='o', label='MLR')
    plt.plot(months_years, predicted_tree, marker='o', label='Decision Tree')
    plt.plot(months_years, predicted_rf, marker='o', label='Random Forest')

    plt.xlabel('Month and Year')
    plt.ylabel('Predicted Phosphate Price')
    plt.title('Predicted Phosphate Prices by Model')
    plt.legend()
    plt.xticks(rotation=45)
    # Customize y-axis tick locations and labels
    plt.yticks(range(100, 401, 200))  # Adjust the range as needed

    plt.tight_layout()
    # Save the plot as a PNG file in the /static directory
    plot_filename = 'static/plotfuture.png'
    plt.savefig(plot_filename)
    plot_to_image(months_years, pred_mlr_future, pred_tr_future, pred_rf_future,  plot_filename)
    plt.close()  # Close the current plot

    return render_template("public/prednextyear.html", futurepredictionss_df=futurepredictionss_df, plot_filename=plot_filename)


def plot_to_image(months_years, pred_mlr_future, pred_tr_future, pred_rf_future, filename):
    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # ... Your plot code here ...
            # Plotting the data
    ax.plot(months_years, pred_mlr_future, marker='o', label='MLR')
    ax.plot(months_years, pred_tr_future, marker='o', label='Decision Tree')
    ax.plot(months_years, pred_rf_future, marker='o', label='Random Forest')
            
    ax.set_xlabel('Month and Year')
    ax.set_ylabel('Predicted Phosphate Price')
    ax.set_title('Predicted Phosphate Prices by Model')
    ax.legend()
    ax.set_xticklabels(months_years, rotation=45)

    # Save the plot to a file
    fig.savefig(filename)










if __name__ == '__main__':
    app.run(debug=True)

  