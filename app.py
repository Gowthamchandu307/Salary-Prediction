import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
loaded_data = pickle.load(open('regmodel.pkl', 'rb'))
regmodel = loaded_data["model"]
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/explore.html')
def explore():
    return render_template('explore.html')

@app.route('/login.html')
def login():
          return render_template('login.html')

@app.route('/Signup.html')
def signup():
    return render_template('Signup.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        country_mapping = {'Other': 0,  
                           'United States': 2,
                           'India': 1,
                            'United Kingdom': 3,
                            'Germany': 4,
                            'Canada': 5,
                            'Brazil': 6,
                            'France': 7,
                            'Spain': 8,
                            'Australia': 9,
                            'Netherlands': 10,
                            'Poland': 11,
                            'Italy': 12,
                            'Russian Federation': 13,
                            'Sweden': 14 }
        
        education_mapping = {
            'Bachelor’s degree': 1,
            'Master’s degree': 2,
            'Less than a Bachelors': 0,
            'Post grad': 3,
        }
        # Convert string values to numeric using mapping dictionaries
        if data['Country'] in country_mapping:
            data['Country'] = country_mapping[data['Country']]
        else:
            return jsonify({'error': 'Invalid country'})

        if data['education'] in education_mapping:
            data['education'] = education_mapping[data['education']]
        else:
            return jsonify({'error': 'Invalid education level'})

        # Ensure the 'experience' value is converted to float
        data['experience'] = float(data['experience'])

        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = regmodel.predict(new_data)[0]

        # Return the prediction result as JSON
        return jsonify({'output': output})
    

    except KeyError as e:
        # Handle the case when a required key is missing
        return jsonify({'error': f"Invalid input. KeyError: {e}"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
         data = {
            'Country': request.form['Country'],
            'education': request.form['education'],
            'experience': float(request.form['experience']),
    }
    except KeyError as e:
        return render_template("home.html", prediction_text=f"Invalid input. KeyError: {e}")

    # Check if required fields are not empty
    if any(value == '' for value in data.values()):
        return render_template("home.html", prediction_text="Invalid input. Please provide values for all features.")

        
           
    final_input = scalar.transform([list(data.values())])
    output = regmodel.predict(final_input)[0]


    formatted_output = round(output, 2)
    return render_template("home.html", prediction_text="{}".format(formatted_output))

if __name__ == "__main__":
    app.run(debug=True)

