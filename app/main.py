from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import joblib

# Load your trained model
model = joblib.load('rf_model.pkl')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.json

    # Ensure 'prices' key is present in the input data
    if 'prices' not in data:
        return jsonify({'error': 'No prices provided'}), 400

    # Convert to NumPy array (assuming your model expects a 1D array)
    prices = np.array(data['prices']).reshape(1, -1)

    # Make predictions
    predictions = model.predict(prices)

    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
