"""
app.py  –  Flask backend for Amazon Product Rating Prediction
Endpoints:
  POST /predict       – return predicted rating for given inputs
  GET  /model-info    – return R², MAE, RMSE, coefficients
  GET  /health        – simple health-check
"""

import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model & metadata once at startup ────────────────────────
with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("model_info.json", "r") as f:
    MODEL_INFO = json.load(f)

# ── Routes ────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple health-check."""
    return jsonify({"status": "ok"})


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model performance metrics, coefficients, and available categories."""
    return jsonify(MODEL_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict product rating.

    Expected JSON body:
    {
        "discounted_price"   : 399,
        "actual_price"       : 1099,
        "discount_percentage": 0.64,      # as decimal, e.g. 64% → 0.64
        "rating_count"       : 24269,
        "main_category"      : "Computers&Accessories"
    }
    Returns:
    {
        "predicted_rating": 4.12,
        "input": { ... }
    }
    """
    data = request.get_json(force=True)

    required = ["discounted_price", "actual_price",
                "discount_percentage", "rating_count", "main_category"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = pd.DataFrame([{
            "discounted_price"   : float(data["discounted_price"]),
            "actual_price"       : float(data["actual_price"]),
            "discount_percentage": float(data["discount_percentage"]),
            "rating_count"       : float(data["rating_count"]),
            "main_category"      : str(data["main_category"]),
        }])

        pred = float(MODEL.predict(row)[0])
        # Clamp to valid rating range [1, 5]
        pred = max(1.0, min(5.0, round(pred, 2)))

        return jsonify({"predicted_rating": pred, "input": data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
