# Anomaly Detection in an IoT Setting

This project demonstrates how to bring a machine learning model into a production-like environment.
A simulated IoT sensor stream sends events to a REST API which serves anomaly detection predictions and logs them to a local database.

## Overview

**Goal:** Detect anomalous sensor behavior in a manufacturing/IoT setting.

**Core components**
- **Simulated stream** (Python): generates sensor events (temperature, humidity, sound) and injects anomalies occasionally.
- **Model training** (Python, scikit-learn): trains an **Isolation Forest** on normal simulated data.
- **Model serving** (Flask REST API): `/predict` endpoint returns anomaly scores and labels.
- **Persistence** (SQLite): logs all predictions for monitoring and traceability.
- **Monitoring (basic)**: query latest anomalies via `/latest_anomalies`.

## Architecture

Data Flow:
1. `simulate_stream.py` generates sensor events (continuous stream).
2. The stream POSTs JSON events to the Flask API (`/predict`).
3. The API validates inputs, runs the anomaly model, and returns a prediction.
4. Inputs + predictions are stored in SQLite (`data/events.db`).
5. Anomalies can be inspected via `/latest_anomalies`.

## Tech Stack

- Python 3.12
- Flask (REST API)
- scikit-learn (Isolation Forest)
- Pydantic (request validation)
- SQLite (prediction logging)
- Requests (simulated client)

## Repository Structure

```
anomaly-detection-iot/
├── src/
│   ├── app.py               # Flask API (serving + logging)
│   ├── schema.py            # Pydantic request schema
│   ├── storage.py           # SQLite persistence
│   ├── simulate_stream.py   # Stream simulator (POSTs events)
│   └── train_model.py       # Isolation Forest training
├── models/
│   └── model_meta.json      # Metadata about the trained model
├── data/
│   ├── training_sample.csv  # Small sample of generated normal data
│   └── .gitkeep
└── requirements.txt

```

> Note: The binary model artifact `models/isoforest.joblib` is intentionally not committed.
> It is generated locally by running the training step.

## Setup (Windows / PowerShell)

### 1) Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Quick Start (Run Order)

### 1) Train the model (creates `models/isoforest.joblib` locally):

```powershell
python src/train_model.py
```

### 2. Start the API:

```powershell
python src/app.py
```

### 3. In a second terminal, run the simulator:

```powershell
python src/simulate_stream.py --count 30 --interval 0.2
```

### 4. Verify:

* `http://127.0.0.1:5000/health`
* `http://127.0.0.1:5000/latest_anomalies?limit=10`

## Train the Model

The training command is shown in **Quick Start (Run Order)** above.  
Running it generates `models/isoforest.joblib` locally.

### Expected outputs:

- `models/isoforest.joblib` (local model artifact)
- `models/model_meta.json`
- `data/training_sample.csv`

## Run the API

The start command is shown in **Quick Start (Run Order)** above.

Health check:

- `GET http://127.0.0.1:5000/health`

## Run the Stream Simulator

The basic run command is shown in **Quick Start (Run Order)** above.

By default, the simulator sends events to:

- `http://127.0.0.1:5000/predict`

### You can override:

```powershell
python src/simulate_stream.py --url http://127.0.0.1:5000/predict --count 200 --interval 1
```

## API Endpoints

### `GET /health`

Returns service and model status.

### `POST /predict`

Validates input and returns anomaly score + label.

Example request body:

```json
{
  "timestamp": "2025-01-01T12:00:00+00:00",
  "sequence": 1,
  "station_id": "station_001",
  "temperature_c": 70.2,
  "humidity_pct": 44.7,
  "sound_db": 66.1
}
```

### `GET /latest_anomalies?limit=10`

Returns recent anomalies logged in SQLite.

## Notes on Production Considerations

* **Serving:** Flask API demonstrates a model-serving interface; in production a WSGI server (e.g., gunicorn) would be used.
* **Validation:** Pydantic ensures robust input validation.
* **Versioning:** `model_version` is returned in each response and stored with predictions.
* **Monitoring:** prediction logs are stored and anomalies can be queried; metrics could be extended with Prometheus/Grafana.