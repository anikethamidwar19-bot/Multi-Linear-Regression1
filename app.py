from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained Multiple Linear Regression model
with open("MLR_MODEL.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form (R&D, Admin, Marketing, State)
        features = [float(x) for x in request.form.values()]

        # Convert to 2D array
        final_features = np.array([features])

        # Make prediction
        prediction = model.predict(final_features)[0]

        # Round to 2 decimal places
        output = round(prediction, 2)

        return render_template("index.html", prediction_text=output)

    except Exception as e:
        return render_template("index.html", prediction_text="Error in prediction")


if __name__ == "__main__":
    app.run(debug=True)