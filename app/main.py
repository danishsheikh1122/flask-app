from flask import Flask, request, jsonify
import numpy as np
import joblib
import flask_cors
# Load your trained model
model = joblib.load('rf_model.pkl')

app = Flask(__name__)
flask_cors.CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.json

    # Convert to DataFrame (assuming your model expects a DataFrame)
    data['prices'] = np.array(list(data['prices'].values()))
    

    # Make predictions
    predictions = model.predict(data['prices'].reshape(1,-1))
    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=False)
