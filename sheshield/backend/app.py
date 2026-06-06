"""
SHESHIELD — Flask Backend

Features:
  - Serves the frontend (templates + static)
  - Loads the trained ML model from backend/model/model.pkl
  - Real-time prediction API:
        POST /api/predict   (JSON)
  - Form handler (server-side render fallback):
        POST /predict
  - Health check:
        GET /health

Run:
  pip install -r requirements.txt
  python model_training.py --generate-assets --target-rows 5000
  python app.py
"""

from __future__ import annotations

import json
import logging
import pickle
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.pkl"
DATA_PATH = ROOT / "data" / "data_extended.csv"
CONTACT_CSV = ROOT / "data" / "contact_messages.csv"


app = Flask(__name__)
# allow frontend JS fetch to /api
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("sheshield")


def load_model() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        log.warning("model.pkl not found. Run model_training.py first.")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log.exception("Failed to load model: %s", e)
        return None


SAVED_MODEL = load_model()


def _compute_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {
        "records": None,
        "areas": None,
        "accuracy": None,
        "features": 5,
    }
    try:
        if DATA_PATH.exists():
            import pandas as pd

            df = pd.read_csv(DATA_PATH)
            stats["records"] = int(len(df))
            stats["areas"] = int(df["Area_Name"].nunique()
                                 ) if "Area_Name" in df.columns else None
    except Exception:
        pass

    if SAVED_MODEL:
        stats["accuracy"] = float(SAVED_MODEL.get("accuracy", 0)) * 100
    return stats


def _encode_inputs(
    crime_rate: float,
    lighting: str,
    police_distance: float,
    time_of_day: str,
    crowd_density: str,
) -> list[float]:
    """
    Encode exactly the same way as training (see model_training.py).
    """
    lighting = (lighting or "Good").title()
    time_of_day = (time_of_day or "Day").title()
    crowd_density = (crowd_density or "Medium").title()

    lighting_enc = 1 if lighting == "Good" else 0
    time_enc = 1 if time_of_day == "Day" else 0
    crowd_enc = {"Low": 0, "Medium": 1, "High": 2}.get(crowd_density, 1)

    return [float(crime_rate), float(lighting_enc), float(police_distance), float(time_enc), float(crowd_enc)]


def predict_risk(payload: dict[str, Any]) -> dict[str, Any]:
    if not SAVED_MODEL:
        return {
            "status": "error",
            "prediction": None,
            "risk_code": "medium",
            "emoji": "⚠️",
            "message": "Model not loaded. Please run model_training.py first.",
        }

    crime_rate = float(payload.get("crime_rate", 5))
    police_distance = float(payload.get("police_distance", 1.0))
    lighting = str(payload.get("lighting", "Good"))
    time_of_day = str(payload.get("time_of_day", "Day"))
    crowd_density = str(payload.get("crowd_density", "Medium"))

    features = _encode_inputs(crime_rate, lighting,
                              police_distance, time_of_day, crowd_density)
    pred_code = int(SAVED_MODEL["model"].predict([features])[0])

    results = {
        0: {
            "prediction": "Low Risk",
            "risk_code": "low",
            "emoji": "🟢",
            "message": (
                "This area appears relatively safe based on available data. "
                "Standard precautions are sufficient."
            ),
        },
        1: {
            "prediction": "Medium Risk",
            "risk_code": "medium",
            "emoji": "🟡",
            "message": (
                "Moderate risk detected. Stay alert, avoid isolated shortcuts, "
                "and prefer travelling with a companion."
            ),
        },
        2: {
            "prediction": "High Risk",
            "risk_code": "high",
            "emoji": "🔴",
            "message": (
                "HIGH RISK AREA detected. Avoid if possible. If travel is necessary, "
                "inform someone of your location and stay in well-lit, crowded areas."
            ),
        },
    }
    out = results.get(pred_code, results[1])
    return {"status": "success", **out}


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": bool(SAVED_MODEL),
            "model_path": str(MODEL_PATH.name),
        }
    )


@app.get("/")
def home():
    return render_template("index.html", result=None, stats=_compute_stats())


@app.post("/predict")
def predict_form():
    """
    Server-rendered fallback (works even if JS is disabled).
    """
    form = request.form
    area_name = form.get("area_name", "Unknown Area")
    payload = {
        "crime_rate": form.get("crime_rate", 5),
        "lighting": form.get("lighting", "Good"),
        "police_distance": form.get("police_distance", 1.0),
        "time_of_day": form.get("time_of_day", "Day"),
        "crowd_density": form.get("crowd_density", "Medium"),
    }

    api_out = predict_risk(payload)
    result = {
        "prediction": api_out.get("prediction"),
        "risk_level": api_out.get("prediction"),
        "risk_code": api_out.get("risk_code"),
        "emoji": api_out.get("emoji"),
        "message": api_out.get("message"),
    }

    return render_template(
        "index.html",
        result=result,
        area_name=area_name,
        crime_rate=payload["crime_rate"],
        lighting=payload["lighting"],
        time_of_day=payload["time_of_day"],
        crowd_density=payload["crowd_density"],
        police_distance=payload["police_distance"],
        stats=_compute_stats(),
    )


@app.get("/map")
def map_view():
    return render_template("map.html")


@app.post("/api/predict")
def api_predict():
    try:
        data = request.get_json(force=True)
        out = predict_risk(data if isinstance(data, dict) else {})
        return jsonify(out)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.post("/api/contact")
def api_contact():
    """
    Contact form endpoint (demo-friendly):
    - Saves messages into backend/data/contact_messages.csv
    """
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid JSON body"}), 400

        name = str(data.get("name", "")).strip()
        email = str(data.get("email", "")).strip()
        message = str(data.get("message", "")).strip()

        if len(name) < 2:
            return jsonify({"status": "error", "message": "Name is required"}), 400
        if "@" not in email or "." not in email:
            return jsonify({"status": "error", "message": "Valid email is required"}), 400
        if len(message) < 5:
            return jsonify({"status": "error", "message": "Message is too short"}), 400

        CONTACT_CSV.parent.mkdir(parents=True, exist_ok=True)
        is_new = not CONTACT_CSV.exists()
        with open(CONTACT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp_utc", "name", "email", "message"])
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    name,
                    email,
                    message,
                ]
            )

        return jsonify({"status": "success", "message": "Thanks! Your message has been saved."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.get("/api/model-info")
def api_model_info():
    if not SAVED_MODEL:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify(
        {
            "status": "success",
            "model_name": SAVED_MODEL.get("model_name"),
            "accuracy": SAVED_MODEL.get("accuracy"),
            "feature_cols": SAVED_MODEL.get("feature_cols"),
            "class_labels": SAVED_MODEL.get("class_labels"),
        }
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SHESHIELD — Backend starting")
    print("=" * 60)
    if SAVED_MODEL:
        print(
            f"✅ Model loaded: {SAVED_MODEL.get('model_name')} ({SAVED_MODEL.get('accuracy', 0)*100:.2f}% acc)")
    else:
        print("⚠️  Model not loaded. Run: python model_training.py --generate-assets")
    print("\nOpen: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
