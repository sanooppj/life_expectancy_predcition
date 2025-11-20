from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Initialize the Flask application
app = Flask(__name__)

# --- MODEL LOADING ---
# Load the trained model and the scaler from the 'models' directory
try:
    model = joblib.load('models/random_forest_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    # This is the R² score from your notebook for the Random Forest model
    model_accuracy = 0.960258 * 100 
except FileNotFoundError:
    print("Error: Model or scaler files not found. Make sure you run the Jupyter notebook to save them.")
    model = None
    scaler = None

# These are the exact 16 feature names the model was trained on
# It's crucial that the form and the DataFrame use these exact names and order
feature_names = [
    'Status', 'Adult Mortality', 'Alcohol', 'Hepatitis B', 'Measles ', ' BMI ', 
    'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 
    'GDP', 'Population', ' thinness  1-19 years', 'Income composition of resources', 
    'Schooling'
]

# --- WEB ROUTES ---

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to display the data entry form
@app.route('/predict_form')
def predict_form():
    return render_template('data-entry.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "Model not loaded. Please check the server logs.", 500

    try:
        # Get the form data and convert it to a list of floats
        # The order of getting data must match feature_names
        input_data = [float(request.form[name]) for name in feature_names]

        # Create a pandas DataFrame from the input data
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_df)

        # Make a prediction using the loaded model
        prediction_result = model.predict(scaled_data)

        # The result is an array, so we get the first element
        final_prediction = prediction_result[0]

        # Render the results page with the prediction and accuracy
        return render_template(
            'results.html', 
            prediction=f"{final_prediction:.2f}", 
            accuracy=f"{model_accuracy:.1f}"
        )
    except Exception as e:
        # Handle potential errors like missing form fields or invalid data
        print(f"An error occurred: {e}")
        return f"An error occurred during prediction: {e}", 400


# Run the app
if __name__ == '__main__':
    # Use debug=True for development, which allows auto-reloading
    app.run(debug=True)