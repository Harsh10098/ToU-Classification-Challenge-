from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("../diabetes_model.pkl", "rb"))
scaler = pickle.load(open("../scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        features = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodpressure"]),
            float(request.form["skinthickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]

        # Convert to numpy array
        final_features = np.array([features])

        # Apply same scaling used in training
        final_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(final_features)[0]

        if prediction == 1:
            result = "⚠️ High Risk of Diabetes"
        else:
            result = "✅ Low Risk of Diabetes"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error in input values")

if __name__ == "__main__":
    app.run(debug=True)
