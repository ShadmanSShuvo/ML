from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("house_price_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Extract features from form
            features = [
                float(request.form.get("MedInc")),
                float(request.form.get("HouseAge")),
                float(request.form.get("AveRooms")),
                float(request.form.get("AveBedrms")),
                float(request.form.get("Population")),
                float(request.form.get("AveOccup")),
                float(request.form.get("Latitude")),
                float(request.form.get("Longitude"))
            ]
            # Scale and predict
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
        except:
            prediction = "Invalid input. Please enter numbers only."
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)