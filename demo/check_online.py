"""Does the online pipeline reproduce the offline results? (consistency check, no LSL)

Streams each S3 locked-test patient's raw recording through the same steps the live demo
uses (causal 1-40 Hz filter in 16-sample chunks, window cut at each cue, harmonisation,
Session with k = 5) and compares the final own/pooled accuracies with the offline locked-test
evaluation (results/summary/s3_locked_test.csv). Not a new hypothesis test: the method and the
patients are the same; only the preprocessing is streaming instead of offline.

Usage: .venv/Scripts/python -I demo/check_online.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from donors import load_donors  # noqa: E402

from hemishift.data.loaders import wang2026sf_affected_side  # noqa: E402
from hemishift.online.lsl import wang2026sf_recording  # noqa: E402
from hemishift.online.session import (  # noqa: E402
    CausalBandpass,
    Session,
    covariances,
    harmonised_epoch,
    window_samples,
)
from hemishift.paths import CONFIG_DIR, RESULTS_DIR  # noqa: E402

K = 5
CHUNK = 16


def stream_patient(subject, donor_C, donor_y):
    data, names, sfreq, events = wang2026sf_recording(subject)
    band = CausalBandpass(sfreq)
    filtered = np.hstack([band(data[:, i:i + CHUNK]) for i in range(0, data.shape[1], CHUNK)])
    offset, length = window_samples(sfreq)
    side = wang2026sf_affected_side()[subject]
    session, latencies, gate = Session(donor_C, donor_y, k=K), [], None
    for trial, (sample, label) in enumerate(events, start=1):
        if label == "sixth_finger_mi":
            session.add(None, label, trial)
            continue
        started = time.perf_counter()
        window = filtered[:, sample + offset:sample + offset + length]
        C = covariances(harmonised_epoch(window, names, sfreq, side)[None])[0]
        for event in session.add(C, label, trial):
            if event["type"] == "calibrated":
                gate = event
        latencies.append(1000 * (time.perf_counter() - started))
    accuracy = {name: 100 * float(np.mean(h)) for name, h in session.hits.items()}
    return accuracy, len(session.hits["own"]), gate, float(np.median(latencies))


def main():
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    offline = pd.read_csv(RESULTS_DIR / "summary" / "s3_locked_test.csv")
    offline = offline[offline["k"] == K].set_index("subject")
    donor_C, donor_y = load_donors()
    rows = []
    for subject in split["test"]:
        accuracy, n, gate, latency = stream_patient(subject, donor_C, donor_y)
        rows.append({"subject": subject, "n_scored": n, "online_own": accuracy["own"],
                     "online_pooled": accuracy["pooled"], "offline_own": offline.loc[subject, "own"],
                     "offline_pooled": offline.loc[subject, "pooled"], "gate": gate["gate"],
                     "latency_ms": latency})
        r = rows[-1]
        print(f"sub {subject:2d}: online own {r['online_own']:.1f} pooled {r['online_pooled']:.1f} | "
              f"offline own {r['offline_own']:.1f} pooled {r['offline_pooled']:.1f} | gate {r['gate']} | "
              f"{n} scored | {latency:.1f} ms per epoch")
    table = pd.DataFrame(rows)
    print(f"\nmeans: online own {table['online_own'].mean():.1f} %, pooled {table['online_pooled'].mean():.1f} % | "
          f"offline own {table['offline_own'].mean():.1f} %, pooled {table['offline_pooled'].mean():.1f} % | "
          f"median processing {table['latency_ms'].median():.1f} ms per epoch")
    table.to_csv(RESULTS_DIR / "summary" / "demo_online_check.csv", index=False)


if __name__ == "__main__":
    main()
