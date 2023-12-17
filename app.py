from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
# Ensure that you have the 'prediction_model.h5' and 'scaler.pkl' files from your training process
from keras.models import load_model
import joblib

model = load_model('prediction_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Get input parameters from the request
        data = request.get_json()
        input_data = [data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                      data['SkinThickness'], data['Insulin'], data['BMI'],
                      data['DiabetesPedigreeFunction'], data['Age']]

        # Create a NumPy array with the input features
        input_data = np.array([input_data])

        # Standardize the input features
        input_scaled = scaler.transform(input_data)

        # Make predictions using the trained model
        prediction_prob = model.predict(input_scaled)
        prediction = int(prediction_prob > 0.5)

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
