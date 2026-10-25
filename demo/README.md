# HemiShift live demo

A stroke patient's EEG streamed over **Lab Streaming Layer (LSL)**, the protocol real EEG
amplifiers use, and decoded live in the browser.

1. The system **calibrates on just 5 rest + 5 motor-imagery trials** (about 2.5 minutes).
2. It then decodes every following trial, showing two decoders side by side:
   - the **patient's own decoder**, trained on the calibration trials only;
   - **HemiShift pooled**, the same calibration plus 1,282 epochs from 16 other stroke
     patients, re-centred so different people and sessions line up. On the 8 locked-test
     patients this beat the own decoder at 5 calibration pairs: 81.4 vs 77.1 %, p = 0.035,
     pre-registered and run once.
3. After calibration, a **safety gate** checks, from the calibration trials alone, whether a
   decoder is reliable enough to give feedback. If not, it says so instead of guessing.

The default patient (22) is from the locked test set and was never used while developing the
method.

## Run

```bash
.venv/Scripts/python -I demo/server.py                 # patient 22, replay x2, k = 5, port 8000
.venv/Scripts/python -I demo/server.py --patient 4 --speed 1   # real time
```

Open http://127.0.0.1:8000 and press **Start session**. Locked-test patients: 1 4 5 6 8 15 17 22.
The first run builds the donor pool (about 5 minutes) and caches it in
`results/models/demo_donors.npz`.

## What runs

| Piece | Code | Role |
|---|---|---|
| Replay | `hemishift.online.lsl.Replay` | Publishes the raw 64-channel, 250 Hz recording and its cue markers as two LSL streams, paced in real time (× speed) |
| Receiver | `hemishift.online.lsl.Receiver` | Subscribes to the streams; causal 1–40 Hz filter; cuts 0.5–4.0 s after each cue as soon as it arrives; harmonises (128 Hz, 14 motor-strip channels, common average, lesion side) |
| Session | `hemishift.online.session.Session` | Calibrates on the first 5 rest + imagery pairs, runs the gate, decodes and scores later trials (the offline P3 protocol: rest epochs count when paired with an imagery trial; sixth-finger trials are not decoded) |
| Decoders | `hemishift.models.pooling` | `OwnTSLR` and `PooledTSLR`, unit-tested to match the evaluated experiment code exactly |
| Dashboard | `demo/server.py`, `demo/static/index.html` | FastAPI + server-sent events; live EEG, cue, calibration, gate, accuracy, history |

Because the Receiver only needs an LSL EEG stream and a marker stream, it would run unchanged
on a live headset.

## Checks

- `demo/check_online.py` streams all 8 locked-test patients through the online pipeline
  without LSL. It reproduces the offline evaluation: own 77.1 % (offline 77.1), pooled 82.5 %
  (offline 81.4). Processing takes 12.6 ms per epoch (median).
- `demo/selftest.py` runs the real LSL path headless at high speed. Patient 22: own 78.6 %,
  pooled 84.3 %. Patient 4: 90.0 / 97.1 %. Both are identical to the check above. Median
  processing was 21–23 ms per epoch, worst 54 ms, under full machine load.
- The decision for each trial is available 4.0 s after its cue (the end of the imagery
  window) plus that processing time.
