import io
from io import StringIO
import json
import pickle
from io import BytesIO


from flask import Flask, request, app, jsonify, url_for, render_template, make_response
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
loaded_data = pickle.load(open('regmodel.pkl', 'rb'))
regmodel = loaded_data["model"]
scalar = pickle.load(open('scaling.pkl', 'rb'))
country_currency_mapping = loaded_data["country_mapping"]
education_mapping = loaded_data["education_mapping"]


# Load country and currency mapping from your dataset
country_currency_mapping = {
    'Other': '$',  
    'United States': '$',
    'India': '₹',
    'United Kingdom': '£',
    'Germany': '€',
    'Canada': '$',
    'Brazil': 'R$',
    'France': '€',
    'Spain': '€',
    'Australia': '$',
    'Netherlands': '€',
    'Poland': 'zł',
    'Italy': '€',
    'Russian Federation': '₽',
    'Sweden': 'kr',
}

# Load education options from your dataset
education_options = ['Bachelor’s degree', 'Master’s degree', 'Less than a Bachelors',
       'Post grad'] 

@app.route('/')
def home():
    return render_template('home.html', country_options=country_currency_mapping.keys(), education_options=education_options)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)[0]

    # Convert salary based on country currency
    country = data['Country']
    if country in country_currency_mapping:
        currency_symbol = country_currency_mapping[country]
        output = f'{currency_symbol} {output}'

    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]

    if len(data) != len(regmodel.coef_):
        return render_template("home.html", prediction_text="Invalid input. Please provide values for all features.")

    final_input = scalar.transform([data])
    output = regmodel.predict(final_input)[0]

    # Convert salary based on country currency
    country = request.form['Country']
    if country in country_currency_mapping:
        currency_symbol = country_currency_mapping[country]
        output = f'{currency_symbol} {output}'

    formatted_output = round(output, 2)
    return render_template("home.html", prediction_text="Your estimated salary is {}".format(formatted_output), country_options=country_currency_mapping.keys(), education_options=education_options)

if __name__ == "__main__":
    app.run(debug=True)
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]

    print("Input Data:", data)  # Add this line to print input data

    if len(data) != len(regmodel.coef_):
        print("Invalid input. Please provide values for all features.")
        return render_template("home.html", prediction_text="Invalid input. Please provide values for all features.")

    final_input = scalar.transform([data])

    print("Transformed Input Data:", final_input)  # Add this line to print transformed input data
    print("Model Coefficients:", regmodel.coef_)  # Add this line to print model coefficients

    output = regmodel.predict(final_input)[0]

    # Convert salary based on country currency
    country = request.form['Country']
    if country in country_currency_mapping:
        currency_symbol = country_currency_mapping[country]
        output = f'{currency_symbol} {output}'

    formatted_output = round(output, 2)
    print("Predicted Output:", formatted_output)  # Add this line to print predicted output
    return render_template("home.html", prediction_text="Your estimated salary is {}".format(formatted_output), country_options=country_currency_mapping.keys(), education_options=education_options)
