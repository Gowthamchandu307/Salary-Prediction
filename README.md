# Build A Salary Prediction Web App With Flask

## Overview:

This project implements a web application for predicting salaries based on user input. The predictive model is trained using machine learning techniques and deployed using Flask.

## Features:

- Predicts salary based on user inputs such as Country, Education Level, Developer Type, and Years of Professional Coding Experience.
- Utilizes a Decision Tree Regressor model for accurate salary predictions.
- Implements a user-friendly web interface for input and result display.

## File Structure:

- **app.py:** The main Flask application file responsible for handling HTTP requests and rendering HTML templates.
- **regmodel.pkl:** Pickled file containing the trained Decision Tree Regressor model.
- **scaling.pkl:** Pickled file containing the StandardScaler used for standardizing input data.
- **templates folder:** Contains HTML templates for the web pages.
- **static folder:** Stores static files like CSS and images.

## Installation:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/SalaryPredictionFlask.git
    cd SalaryPrediction
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask application:

    ```bash
    python app.py
    ```

4. Open your browser and go to [http://localhost:5000](http://localhost:5000) to use the Salary Prediction Web App.

## Usage:

1. Enter the required details: Country, Education Level, Developer Type, and Years of Professional Coding Experience.
2. Click on the "Predict Salary" button.
3. The predicted salary will be displayed on the page.

## Deployment:

To deploy the Flask app, you can use platforms like Heroku, AWS, or any other hosting service. Ensure that the necessary environment variables are set and configurations are adjusted for production deployment.

## Note:

- This project is for educational purposes and may require further enhancements for production use.
- Ensure that you have Python installed, and use a virtual environment for better dependency management.

Feel free to explore and customize the project according to your needs. If you encounter any issues or have suggestions, please let us know!
