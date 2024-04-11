from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the crop prediction model
with open("trained_model.pickle", "rb") as file:
    crop_model = pickle.load(file)

# Load the crop yield prediction model
with open('model.pickle', 'rb') as f:
    yield_model = pickle.load(f)

# Define a function to preprocess input data
def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])
    # Perform any necessary preprocessing
    # For example, one-hot encode categorical variables
    # Assuming 'Area' is not provided as input in this case
    input_df_encoded = pd.get_dummies(input_df)
    # Ensure input features are in the same order as trained model input
    # For example, drop 'Area' column if present in input data
    if 'Area' in input_df_encoded.columns:
        input_df_encoded.drop('Area', axis=1, inplace=True)
    return input_df_encoded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    # Get input data from form
    input_data = {
        'Year': float(request.form['year']),
        'hg/ha_yield': float(request.form['hg/ha_yield']),
        'average_rain_fall_mm_per_year': float(request.form['average_rainfall']),
        'pesticides_tonnes': float(request.form['pesticides']),
        'avg_temp': float(request.form['avg_temp']),
        'Kharif': 1 if request.form['Season'] == 'Kharif' else 0,
        'Rabi': 1 if request.form['Season'] == 'Rabi' else 0
    }
    # Preprocess input data
    input_df_encoded = preprocess_input(input_data)
    # Make prediction
    predicted_crop = crop_model.predict(input_df_encoded)
    # Output prediction
    return render_template('index.html', prediction=predicted_crop[0])

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    # Get the form data
    year = float(request.form['year'])
    avg_rainfall = float(request.form['average_rainfall'])
    pesticides = float(request.form['pesticides'])
    avg_temp = float(request.form['avg_temp'])
    item = request.form['item']  # assuming you have an input named 'item' in your form
    Season = request.form['Season']  # assuming you have an input named 'item' in your form
    
    # Create a DataFrame with the input data
    input_data = {
        'Year': [year],
        'average_rain_fall_mm_per_year': [avg_rainfall],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [avg_temp],
        'Kharif': [0],  # Set all season types to 0 initially
        'Rabi': [0], # Set all season types to 0 initially
        'Cassava': [0],  # Set all crop types to 0 initially
        'Maize': [0],
        'Potatoes': [0],
        'Rice, paddy': [0],
        'Sorghum': [0],
        'Soybeans': [0],
        'Sweet potatoes': [0],
        'Wheat': [0]
    }

    input_data[Season] = [1]  # Set the selected crop type to 1

    input_data[item] = [1]  # Set the selected crop type to 1
    
    input_df = pd.DataFrame(input_data)
    
    # Make prediction
    prediction = yield_model.predict(input_df)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
    