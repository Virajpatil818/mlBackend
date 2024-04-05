from flask import Flask, jsonify, request
from catboost import CatBoostRegressor
import joblib
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the CatBoost model
catboost_model = joblib.load('catboost_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()

        # Extract features from the data
        dew_point = data['Dew_Point_dc']
        temperature = data['Temperature_dc']
        wind_speed = data['Wind_Speed_kph']

        # Make prediction
        prediction = catboost_model.predict([[dew_point, temperature, wind_speed]])

        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


if __name__ == '__main__':
    app.run(debug=True)
