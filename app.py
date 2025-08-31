from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return "Heart Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # input as JSON
    features = np.array(data["features"]).reshape(1, -1)  # reshape to 2D
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]  # probability of disease
    
    result = {
        "prediction": int(prediction),
        "probability": float(prob),
        "label": "Heart Disease Present" if prediction == 1 else "No Heart Disease"
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
