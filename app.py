from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib


# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)


# ==========================================
# Load ML Model
# ==========================================

MODEL_PATH = "model/churn_model.pkl"
FEATURE_PATH = "model/feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

THRESHOLD = 0.35


# ==========================================
# Home Route
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Prediction API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ------------------------------
        # Convert input to model format
        # ------------------------------

        country = data["country"]
        gender = data["gender"]

        customer = {
            "credit_score": float(data["credit_score"]),
            "age": float(data["age"]),
            "tenure": float(data["tenure"]),
            "balance": float(data["balance"]),
            "products_number": int(data["products_number"]),
            "credit_card": int(data["credit_card"]),
            "active_member": int(data["active_member"]),
            "estimated_salary": float(data["estimated_salary"]),

            "country_Germany": 1 if country == "Germany" else 0,
            "country_Spain": 1 if country == "Spain" else 0,

            "gender_Male": 1 if gender == "Male" else 0
        }


        # ------------------------------
        # Create DataFrame
        # ------------------------------

        input_data = pd.DataFrame(
            [customer],
            columns=feature_names
        )


        # ------------------------------
        # Prediction
        # ------------------------------

        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = int(
            probability >= THRESHOLD
        )


        # ------------------------------
        # Risk Level
        # ------------------------------

        if probability >= 0.70:
            risk = "Very High"

        elif probability >= 0.50:
            risk = "High"

        elif probability >= 0.35:
            risk = "Medium"

        else:
            risk = "Low"


        # ------------------------------
        # Response
        # ------------------------------

        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "risk": risk
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True
    )