import logging
import os
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

model_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "xgb_final_model.joblib"
)
model = load(model_file_path)


def preprocess_data(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    object_cols = df.select_dtypes(include="object").columns
    df[object_cols] = df[object_cols].astype("category")
    return df


@app.route("/")
def intro():
    return "Welcome to our risk evaluation API"


@app.route("/predict_risk", methods=["POST"])
def predict_risk_probability():
    try:
        data = request.json
        data_cleaned = preprocess_data(data)
        proba = round((model.predict_proba(data_cleaned)[:, -1][0]) * 100, 2)
        return jsonify({"Probability of Late Payment": proba})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500


if __name__ == "__main__":
    app.run(debug=True)
