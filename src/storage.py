import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


DB_PATH = Path("data") / "events.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                station_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                temperature_c REAL NOT NULL,
                humidity_pct REAL NOT NULL,
                sound_db REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                is_anomaly INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                raw_input_json TEXT NOT NULL,
                raw_output_json TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_station ON predictions(station_id);")


def insert_prediction(
    *,
    timestamp: str,
    station_id: str,
    sequence: int,
    temperature_c: float,
    humidity_pct: float,
    sound_db: float,
    anomaly_score: float,
    is_anomaly: bool,
    model_version: str,
    raw_input: Dict[str, Any],
    raw_output: Dict[str, Any],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, station_id, sequence,
                temperature_c, humidity_pct, sound_db,
                anomaly_score, is_anomaly, model_version,
                raw_input_json, raw_output_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                timestamp,
                station_id,
                sequence,
                temperature_c,
                humidity_pct,
                sound_db,
                float(anomaly_score),
                1 if is_anomaly else 0,
                model_version,
                json.dumps(raw_input, ensure_ascii=False),
                json.dumps(raw_output, ensure_ascii=False),
            ),
        )


def fetch_latest_anomalies(limit: int = 10) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT timestamp, station_id, sequence,
                   temperature_c, humidity_pct, sound_db,
                   anomaly_score, is_anomaly, model_version
            FROM predictions
            WHERE is_anomaly = 1
            ORDER BY id DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
