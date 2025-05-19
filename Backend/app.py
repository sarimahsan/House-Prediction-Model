from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Hello, ML Web App is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    area = float(data['area'])
    bedrooms = int(data['bedrooms'])
    bathrooms = int(data['bathrooms'])
    stories = int(data['stories'])
    
    mainroad = 1 if data['mainroad'].lower() == 'yes' else 0
    guestroom = 1 if data['guestroom'].lower() == 'yes' else 0
    basement = 1 if data['basement'].lower() == 'yes' else 0
    hotwaterheating = 1 if data['hotwaterheating'].lower() == 'yes' else 0
    airconditioning = 1 if data['airconditioning'].lower() == 'yes' else 0
    prefarea = 1 if data['prefarea'].lower() == 'yes' else 0

    parking = int(data['parking'])

    furnishingstatus = {
        'furnished': 2,
        'semi-furnished': 1,
        'unfurnished': 0
    }
    furnishing = furnishingstatus.get(data['furnishingstatus'].lower(), 0)  
    features = [[
        area, bedrooms, bathrooms, stories, mainroad,
        guestroom, basement, hotwaterheating, airconditioning, parking,prefarea,
        furnishing
    ]]

    prediction = model.predict(features)
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
