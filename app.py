from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    cost = None
    hourly_rate = None

    if request.method == "POST":

        # Get hourly rate
        hourly_rate = float(request.form["HourlyRate"])

        # Project data
        data = {
            "TeamExp": float(request.form["TeamExp"]),
            "ManagerExp": float(request.form["ManagerExp"]),
            "YearEnd": float(request.form["YearEnd"]),
            "Length": float(request.form["Length"]),
            "Transactions": float(request.form["Transactions"]),
            "Entities": float(request.form["Entities"]),
            "PointsNonAdjust": float(request.form["PointsNonAdjust"]),
            "Adjustment": float(request.form["Adjustment"]),
            "PointsAjust": float(request.form["PointsAjust"]),
            "Language": float(request.form["Language"])
        }

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Predict effort
        prediction = model.predict(input_data)[0]

        # Effort cannot be negative
        prediction = max(0, prediction)

        # Calculate estimated cost
        cost = prediction * hourly_rate

    return render_template(
        "index.html",
        prediction=prediction,
        cost=cost,
        hourly_rate=hourly_rate
    )


if __name__ == "__main__":
    app.run(debug=True)