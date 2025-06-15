from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and scaler with error handling
try:
    model = joblib.load('trained_data/student_pass_model.pkl')
    scaler = joblib.load('trained_data/scaler.pkl')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def index():
    return "Server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not properly loaded'}), 500

        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_columns = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)']
        
        missing_fields = [field for field in required_columns if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

        try:
            study_hours = float(data['Study_Hours_per_Week'])
            sleep_hours = float(data['Sleep_Hours_per_Night'])
            attendance = float(data['Attendance (%)'])
            stress_level = float(data['Stress_Level (1-10)'])
        except (ValueError, TypeError):
            return jsonify({'error': 'All input values must be numeric'}), 400

        if not (0 <= study_hours <= 80):
            return jsonify({'error': 'Study hours must be between 0 and 80 per week'}), 400
        if not (0 <= sleep_hours <= 24):
            return jsonify({'error': 'Sleep hours must be between 0 and 24 per night'}), 400
        if not (0 <= attendance <= 100):
            return jsonify({'error': 'Attendance must be between 0 and 100 percent'}), 400
        if not (1 <= stress_level <= 10):
            return jsonify({'error': 'Stress level must be between 1 and 10'}), 400

        input_data = pd.DataFrame([{
            'Study_Hours_per_Week': study_hours,
            'Sleep_Hours_per_Night': sleep_hours,
            'Attendance (%)': attendance,
            'Stress_Level (1-10)': stress_level
        }])

        logger.info(f"Input data: {input_data.iloc[0].to_dict()}")

        expected_columns = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)']
        input_data = input_data[expected_columns]

        input_data_scaled = scaler.transform(input_data)
        logger.info(f"Scaled input: {input_data_scaled}")

        prediction_probabilities = model.predict_proba(input_data_scaled)
        logger.info(f"Prediction probabilities: {prediction_probabilities}")

        if len(prediction_probabilities[0]) != 2:
            return jsonify({'error': 'Model output format unexpected'}), 500

        pass_probability = prediction_probabilities[0][1] * 100

        if pass_probability >= 70:
            result = 'pass'
            confidence = 'MODERATE'
        elif pass_probability >= 50:
            result = 'pass'
            confidence = 'MODERATE'
        elif pass_probability >= 30:
            result = 'fail'
            confidence = 'MODERATE'
        else:
            result = 'fail'
            confidence = 'HIGH'

        response = {
            'result': result,
            'percentage': f"{pass_probability:.2f}%",
            'confidence': confidence,
            'interpretation': get_interpretation(study_hours, sleep_hours, attendance, stress_level, pass_probability)
        }

        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

def get_interpretation(study_hours, sleep_hours, attendance, stress_level, pass_prob):
    """Return one overall motivational message based on inputs and pass probability"""

    if pass_prob >= 70:
        return "🎉 Fantastic job! You're on the right track to success. Keep up your hard work and positive habits!"
    elif pass_prob >= 50:
        return ("👍 You're doing well, but there's room to improve. Stay focused, keep balancing your studies and self-care, "
                "and great results will follow!")
    elif pass_prob >= 30:
        return ("💪 Don't be discouraged! Every step forward counts. Use this as motivation to adjust your habits and "
                "push towards your goals. You’ve got this!")
    else:
        return ("🌟 Remember, setbacks are just setups for comebacks. Keep believing in yourself, make small improvements, "
                "and success will come. Keep going—you’re capable of amazing things!")

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        model_type = type(model).__name__
        
        info = {
            'model_type': model_type,
            'feature_names': ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)'],
            'status': 'loaded'
        }
        
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
        if hasattr(model, 'classes_'):
            info['classes'] = model.classes_.tolist()
            
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
