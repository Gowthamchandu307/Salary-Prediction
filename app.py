import json
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns



app = Flask(__name__)

# Load the model
loaded_data = pickle.load(open('regmodel.pkl', 'rb'))
regmodel = loaded_data["model"]
scalar = pickle.load(open('scaling.pkl', 'rb'))

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

# Set the style for seaborn
sns.set(style="whitegrid")
def save_plot_as_image(fig, filename):
    fig.savefig(filename, format='png')  # Specify the format
    plt.close(fig)

def save_pie_chart(data, filename):
    plt.figure(figsize=(10,8))
    plt.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    plt.axis("equal")
    plt.savefig(filename)
    plt.close()

def save_bar_chart(data, filename):
    plt.figure()
    plt.bar(data.index, data)
    plt.savefig(filename)
    plt.close()

def save_line_chart(data, filename):
    plt.figure()
    plt.plot(data.index, data)
    plt.savefig(filename)
    plt.close()

def boxplot(data, filename, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='EdLevel', y='Salary', data=data, ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    save_plot_as_image(fig, filename)

def barplot(data, filename, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='DevType', y='Salary', data=data, ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    save_plot_as_image(fig, filename)

def countplot(data, filename, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Country', data=data, ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    save_plot_as_image(fig, filename)

def scatterplot(data, filename, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='YearsCodePro', y='Salary', data=data, ax=ax)
    ax.set_title(title)
    save_plot_as_image(fig, filename)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/explore.html')
def explore():
    plots = []

    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "DevType", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
    df = df[df["ConvertedComp"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedComp"] <= 250000]
    df = df[df["ConvertedComp"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedComp": "Salary"}, axis=1)

    df['DevType'] = df['DevType'].str.split(";")
    df['DevType'] = df['DevType'].str[0]
    dev_map = shorten_categories(df['DevType'].value_counts(), 100)
    df['DevType']  = df['DevType'].map(dev_map)
    df['DevType'].value_counts()

    # Save plots as images
    
    save_pie_chart(df['Country'].value_counts(), 'static/pie_chart.png')
    save_bar_chart(df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True), 'static/bar_chart.png')
    save_line_chart(df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True), 'static/line_chart.png')
    barplot(df, 'static/bar_plot.png', 'Average Salary by Developer Type')
    boxplot(df, 'static/box_plot.png', 'Salary Distribution by Education Level')
    countplot(df, 'static/count_plot.png', 'Distribution of Respondents by Country')
    scatterplot(df, 'static/scatter_plot.png', 'Salary vs Years of Professional Coding Experience')


    # Append image URLs to the plots list
    plots.append('static/pie_chart.png')
    plots.append('static/bar_chart.png')
    plots.append('static/line_chart.png')
    #plots.append('static/count_plot.png')
    return render_template('explore.html', plots=plots)


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
        devtype_mapping = {
            'Developer, back-end': 0,
            'Developer, full-stack': 1,
            'Database administrator': 2,
            'Developer, front-end': 3,
            'Data or business analyst': 4,
            'Academic researcher': 5,
            'Designer': 6,
            'Developer, desktop or enterprise applications': 7,
            'Data scientist or machine learning specialist': 8,
            'Developer, mobile': 9,
            'Developer, embedded applications or devices': 10,
            'Other': 11,
            'DevOps specialist': 12,
            'Developer, QA or test': 13,
            'Engineer, data': 14,
            'Engineering manager': 15,
            'Developer, game or graphics': 16
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
        if data['devtype'] in devtype_mapping:
            data['devtype'] = devtype_mapping[data['devtype']]
        else:
            return jsonify({'error': 'Invalid Job Role'})

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
            'devtype' : request.form['devtype'],
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
    return render_template("home.html", prediction_text="$ {} [annually]".format(formatted_output))

if __name__ == "__main__":
    app.run(debug=True)
