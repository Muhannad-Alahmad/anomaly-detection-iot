from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


BASE_TEMP_C = 70.0
BASE_HUMIDITY_PCT = 45.0
BASE_SOUND_DB = 65.0

TEMP_STEP = 0.4
HUM_STEP = 0.6
SOUND_STEP = 0.5


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def generate_normal_stream(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate 'normal' sensor readings using random-walk + mean reversion.
    No anomaly injection here, by design.
    """
    rng = np.random.default_rng(seed)

    temp = BASE_TEMP_C + rng.uniform(-1.0, 1.0)
    hum = BASE_HUMIDITY_PCT + rng.uniform(-2.0, 2.0)
    sound = BASE_SOUND_DB + rng.uniform(-2.0, 2.0)

    rows: list[tuple[float, float, float]] = []

    for _ in range(n):
        temp += rng.uniform(-TEMP_STEP, TEMP_STEP)
        hum += rng.uniform(-HUM_STEP, HUM_STEP)
        sound += rng.uniform(-SOUND_STEP, SOUND_STEP)

        # mean reversion
        temp += (BASE_TEMP_C - temp) * 0.02
        hum += (BASE_HUMIDITY_PCT - hum) * 0.02
        sound += (BASE_SOUND_DB - sound) * 0.02

        # plausible bounds
        temp = clamp(float(temp), 40.0, 120.0)
        hum = clamp(float(hum), 10.0, 90.0)
        sound = clamp(float(sound), 30.0, 110.0)

        rows.append((temp, hum, sound))

    df = pd.DataFrame(rows, columns=["temperature_c", "humidity_pct", "sound_db"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of normal samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.03, help="Expected anomaly rate at inference time")
    args = parser.parse_args()

    data_dir = Path("data")
    models_dir = Path("models")
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    df = generate_normal_stream(args.n, seed=args.seed)

    X = df[["temperature_c", "humidity_pct", "sound_db"]].values

    model = IsolationForest(
        n_estimators=300,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X)

    model_path = models_dir / "isoforest.joblib"
    joblib.dump(model, model_path)

    # Save a small sample for your README/slides
    sample_path = data_dir / "training_sample.csv"
    df.head(200).to_csv(sample_path, index=False)

    meta = {
        "model_type": "IsolationForest",
        "features": ["temperature_c", "humidity_pct", "sound_db"],
        "n_samples": int(args.n),
        "contamination": float(args.contamination),
        "random_state": int(args.seed),
        "artifact": str(model_path.as_posix()),
    }
    meta_path = models_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Saved sample data: {sample_path}")


if __name__ == "__main__":
    main()
