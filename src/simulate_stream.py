import argparse
import time
import random
import requests
from datetime import datetime, timezone


# Baselines (normal operating conditions)
BASE_TEMP_C = 70.0
BASE_HUMIDITY_PCT = 45.0
BASE_SOUND_DB = 65.0

# Random walk step sizes (small natural fluctuations)
TEMP_STEP = 0.4
HUM_STEP = 0.6
SOUND_STEP = 0.5

# Probability of an anomaly at each event
ANOMALY_PROB = 0.03  # 3%


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def maybe_inject_anomaly(temp_c: float, humidity_pct: float, sound_db: float) -> tuple[float, float, float, bool]:
    if random.random() > ANOMALY_PROB:
        return temp_c, humidity_pct, sound_db, False

    anomaly_type = random.choice(["temp_spike", "hum_spike", "sound_spike", "multi"])
    if anomaly_type == "temp_spike":
        temp_c += random.uniform(8, 18)
    elif anomaly_type == "hum_spike":
        humidity_pct += random.uniform(12, 25)
    elif anomaly_type == "sound_spike":
        sound_db += random.uniform(8, 20)
    else:
        temp_c += random.uniform(6, 14)
        humidity_pct += random.uniform(10, 20)
        sound_db += random.uniform(6, 16)

    return temp_c, humidity_pct, sound_db, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5000/predict")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=0, help="0 = run forever, otherwise number of events to send")
    args = parser.parse_args()

    api_url = args.url
    interval_sec = args.interval

    temp_c = BASE_TEMP_C + random.uniform(-1.0, 1.0)
    humidity_pct = BASE_HUMIDITY_PCT + random.uniform(-2.0, 2.0)
    sound_db = BASE_SOUND_DB + random.uniform(-2.0, 2.0)

    seq = 0
    print(f"Streaming to {api_url} every {interval_sec:.2f}s. Ctrl+C to stop.")

    while True:
        seq += 1

        temp_c += random.uniform(-TEMP_STEP, TEMP_STEP)
        humidity_pct += random.uniform(-HUM_STEP, HUM_STEP)
        sound_db += random.uniform(-SOUND_STEP, SOUND_STEP)

        temp_c += (BASE_TEMP_C - temp_c) * 0.02
        humidity_pct += (BASE_HUMIDITY_PCT - humidity_pct) * 0.02
        sound_db += (BASE_SOUND_DB - sound_db) * 0.02

        temp_c = clamp(temp_c, 40.0, 120.0)
        humidity_pct = clamp(humidity_pct, 10.0, 90.0)
        sound_db = clamp(sound_db, 30.0, 110.0)

        t2, h2, s2, injected = maybe_inject_anomaly(temp_c, humidity_pct, sound_db)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": seq,
            "station_id": "station_001",
            "temperature_c": round(t2, 2),
            "humidity_pct": round(h2, 2),
            "sound_db": round(s2, 2),
        }

        try:
            r = requests.post(api_url, json=payload, timeout=3)
            print(f"[{seq:05d}] sent (anomaly_injected={injected}) -> status={r.status_code} resp={r.text[:120]}")
        except requests.RequestException as e:
            print(f"[{seq:05d}] ERROR posting to API: {e}")

        if args.count and seq >= args.count:
            print(f"Done. Sent {seq} events.")
            break

        time.sleep(interval_sec)


if __name__ == "__main__":
    main()
