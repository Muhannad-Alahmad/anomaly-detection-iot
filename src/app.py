from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from pydantic import ValidationError

from schema import SensorEvent
from storage import init_db, insert_prediction, fetch_latest_anomalies

app = Flask(__name__)

MODEL_PATH = Path("models") / "isoforest.joblib"
MODEL_VERSION = "isoforest-v1"

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


def anomaly_score(event: SensorEvent) -> float:
    """
    IsolationForest:
    - score_samples(X) returns higher values for more normal points.
    We invert it so higher = more anomalous for easier interpretation.
    """
    if model is None:
        return 0.0

    X = np.array([[event.temperature_c, event.humidity_pct, event.sound_db]], dtype=float)
    normality = float(model.score_samples(X)[0])
    return -normality


def is_anomaly(event: SensorEvent) -> bool:
    """
    IsolationForest:
    predict(X) returns -1 for anomaly, +1 for normal.
    """
    if model is None:
        return False

    X = np.array([[event.temperature_c, event.humidity_pct, event.sound_db]], dtype=float)
    pred = int(model.predict(X)[0])
    return pred == -1


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_version": MODEL_VERSION,
            "model_loaded": model is not None,
            "model_path": str(MODEL_PATH),
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    try:
        event = SensorEvent.model_validate(payload)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 422

    score = anomaly_score(event)
    flagged = is_anomaly(event)

    output = {
        "station_id": event.station_id,
        "sequence": event.sequence,
        "timestamp": event.timestamp,
        "anomaly_score": score,
        "is_anomaly": flagged,
        "model_version": MODEL_VERSION,
    }

    insert_prediction(
        timestamp=event.timestamp,
        station_id=event.station_id,
        sequence=event.sequence,
        temperature_c=event.temperature_c,
        humidity_pct=event.humidity_pct,
        sound_db=event.sound_db,
        anomaly_score=score,
        is_anomaly=flagged,
        model_version=MODEL_VERSION,
        raw_input=payload,
        raw_output=output,
    )

    return jsonify(output)


@app.get("/latest_anomalies")
def latest_anomalies():
    limit = request.args.get("limit", default="10")
    try:
        limit_int = max(1, min(100, int(limit)))
    except ValueError:
        limit_int = 10

    return jsonify(fetch_latest_anomalies(limit_int))


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
