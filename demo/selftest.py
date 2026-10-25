"""Headless end-to-end test of the live path: LSL replay -> LSL receiver -> session.

Replays one patient over real LSL streams (fast), decodes with the same Receiver the dashboard
uses, and prints the final accuracies and per-epoch processing latency. They should match
demo/check_online.py for the same patient up to tiny differences in chunk timing.

Usage: .venv/Scripts/python -I demo/selftest.py [--patient 22] [--speed 20]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from donors import load_donors  # noqa: E402

from hemishift.data.loaders import wang2026sf_affected_side  # noqa: E402
from hemishift.online.lsl import Receiver, Replay, wang2026sf_recording  # noqa: E402
from hemishift.online.session import Session  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=int, default=22)
    parser.add_argument("--speed", type=float, default=20.0)
    args = parser.parse_args()
    data, names, sfreq, events = wang2026sf_recording(args.patient)
    donor_C, donor_y = load_donors()
    collected = []
    session = Session(donor_C, donor_y, k=5)
    receiver = Receiver(session, wang2026sf_affected_side()[args.patient], collected.append,
                        name="HemiShiftSelftest")
    replay = Replay(data, names, sfreq, events, speed=args.speed, name="HemiShiftSelftest")
    started = time.perf_counter()
    receiver.start()
    replay.start()
    replay.finished.wait()
    time.sleep(2.0)  # let the receiver drain the last chunks
    receiver.stop_flag.set()
    receiver.join(timeout=5)
    replay.stop_flag.set()
    replay.join(timeout=5)

    kinds = [e["type"] for e in collected]
    scores = [e for e in collected if e["type"] == "score"]
    latencies = [e["latency_ms"] for e in collected if e["type"] == "decision" and "latency_ms" in e]
    gate = next((e for e in collected if e["type"] == "calibrated"), None)
    print(f"replayed {data.shape[1] / sfreq:.0f} s of {len(names)}-channel EEG in {time.perf_counter() - started:.0f} s "
          f"(x{args.speed}); cues {kinds.count('cue')}, decisions {kinds.count('decision')}, eeg chunks {kinds.count('eeg')}")
    print(f"gate: {gate['gate'] if gate else None} (calibration CV {gate['cv_correct'] if gate else None})")
    if scores:
        print(f"final: {scores[-1]['n']} scored epochs | own {scores[-1]['accuracy']['own']:.1f} % | "
              f"pooled {scores[-1]['accuracy']['pooled']:.1f} %")
    if latencies:
        print(f"processing per epoch: median {np.median(latencies):.1f} ms, max {np.max(latencies):.1f} ms")


if __name__ == "__main__":
    main()
