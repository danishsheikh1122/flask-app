# from flask import Flask, request, jsonify
# import numpy as np
# import joblib
# import flask_cors
# # Load your trained model
# model = joblib.load('rf_model.pkl')

# app = Flask(__name__)
# flask_cors.CORS(app)
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the data from the request
#     data = request.json

#     # Convert to DataFrame (assuming your model expects a DataFrame)
#     data['prices'] = np.array(list(data['prices'].values()))
    

#     # Make predictions
#     predictions = model.predict(data['prices'].reshape(1,-1))
#     # Return the predictions as JSON
#     return jsonify(predictions.tolist())

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS  # Use the correct import here

# Load your trained model
model = joblib.load('rf_model.pkl')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json
        
        # Ensure data exists and has the correct structure
        if 'prices' not in data:
            return jsonify({"error": "Invalid input, 'prices' key missing."}), 400
        
        # Convert to NumPy array (assuming it's a dictionary with numeric values)
        prices = np.array(list(data['prices'].values()))
        
        # Make predictions
        predictions = model.predict(prices.reshape(1, -1))
        
        # Return the predictions as JSON
        return jsonify(predictions.tolist())
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Change host to 0.0.0.0 and set debug=False
