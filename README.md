# HemiShift

Decoding a stroke patient's intent to move from EEG, for brain-computer-interface rehabilitation.
After a stroke, the "I'm trying to move" signal weakens and shifts toward the healthy hemisphere.
HemiShift teaches decoders that shift by turning healthy EEG into stroke-like EEG (LesionSim),
then tests them on hospitals they have never seen.

- Plan, phases and gates: [PLAN.md](PLAN.md)
- Prior work and novelty evidence: [LITERATURE.md](LITERATURE.md)
- Every experiment, including the failures: [ACHIEVEMENTS.md](ACHIEVEMENTS.md)
- Live demo, a never-seen patient streamed over LSL and decoded in the browser:
  [demo/README.md](demo/README.md). Run `.venv/Scripts/python -I demo/server.py`.

## Results so far

- **Evaluation rules:** patient-level splits, chronological calibration, and blocked CV. Each
  cohort has a locked test set of never-seen patients, evaluated once per pre-registered
  hypothesis.
- **Cohorts:**
  - S1 = Beijing acute stroke, left vs right hand (Liu et al. 2024).
  - S3 = Tianjin chronic stroke, affected-hand imagery vs rest (Wang et al. 2026).
  - S4 = Tianjin stroke, gait imagery vs idle, three sessions per patient (Liu et al. 2025).
    It is a fresh replication cohort of 27 patients, from the same institution as S3.

| Finding | Result | Status |
|---|---|---|
| Pooling other patients' data into a re-centred Riemannian decoder, 5 calibration pairs (S3) | 77.1 → **81.4 %** (+4.3); 7 of 8 patients helped; matches own decoder at 10 pairs | **Confirmed on the locked test** (pre-registered, p = 0.035) |
| The same pooling on a new cohort (S4, 27 patients) | 69.1 → 70.8 % (+1.6); 14 helped, 6 harmed | **Confirmed** (pre-registered replication, p = 0.018) |
| Pooled decoder with a mu / low-beta / high-beta band bank (S4) | 70.8 → **73.1 %** (+2.3 over pooled, +4.0 over own) | **Confirmed** (pre-registered, Holm p = 0.021) |
| A returning patient's own earlier sessions as donors (S4 post and follow-up) | 67.4 → 70.7 % (+3.2); +5.4 at follow-up | **Confirmed** (pre-registered, Holm p = 0.005) |
| No labelled calibration for a returning patient: earlier sessions' decoder only (S4) | 69.7 % with no labels vs 67.4 % with 5 labelled pairs | **Confirmed non-inferior** (margin 2 points, Holm p = 0.021) |
| Robustness to ICA cleaning (S3, authors' cleaned release) | Eye control falls to 55–58 %, motor decoding holds; on cleaned locked-test data the band-bank pooled decoder keeps +3.5, single-band pooling +1.6 | Sensitivity analysis |
| SafeTransfer gate: per-patient choice from calibration data only (S3) | Patients harmed by transfer: 9 → 2 of 16 (5 pairs), 11 → 1 (10 pairs) | Dev (p = 0.008 / 0.001) |
| Window ensemble: 16 tangent-space models incl. the post-imagery window (S1) | 54.4 → 58.0 % locked test; 57.4 → 61.5 % dev | Same direction on locked test, not significant (p = 0.083) |
| Calibration-only screening of who can use a left/right BCI (S1) | Passed 72.6 % vs abstained 51.8 % on later trials (dev); 1 of 15 passed on the locked test, at 78.9 % | Dev strong; locked test not evaluable |
| LesionSim: healthy-to-stroke simulation, 6 forms incl. learned settings (S1, S3) | No gain; real stroke donors help where simulated ones do not | Negative |
| Deep learning (EEGNet, healthy pretraining) and an EEG foundation model (CBraMod) | No gain over per-patient Riemannian decoders | Negative |

**Also tested, without gain:** 40 decoder variants on S3 dev (more channels, longer windows,
donor weighting and selection, Riemannian stretching, SVM, ensembles, mixup). Only band banks
and online co-adaptive recalibration helped there.

**Pitfalls quantified on the way:**
- In an imagery-vs-rest paradigm, non-motor channels separate the classes too. On S3, frontal
  1–8 Hz (eyes) reaches 77 % and occipital channels 74 %. The decoders stay on the motor
  template, and the claims are checked on ICA-cleaned data.
- The Liu2024 published baselines do not reproduce (about 50 % vs 56–58 %).
- Choosing the best window/band per patient without nested CV inflates accuracy by about 13
  points.
- Offline block re-centring gains +5 points only through look-ahead.
- Euclidean alignment on common-average data needs a rank-aware inverse. Otherwise one null
  direction is amplified a millionfold, which faked a "negative transfer" result here until
  it was found and fixed.

## Setup (Windows, Python 3.11, NVIDIA GPU)

```bash
python -m venv .venv
# torch and torchaudio first, from the CUDA index. A plain `pip install torch` gives a CPU-only
# build, and braindecode would otherwise pull a PyPI torchaudio that can't load with this torch.
.venv/Scripts/python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --retries 10 --timeout 120
.venv/Scripts/python -m pip install -r requirements.txt
# exact versions of the tested environment: requirements.lock.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

`HEMISHIFT_DEVICE=cpu|cuda` pins the device. `HEMISHIFT_DATA` moves the data folder (default `data/`).

## Data

```bash
.venv/Scripts/python -I scripts/download_all.py              # all non-deferred cohorts (~22 GB)
.venv/Scripts/python -I scripts/download_all.py liu2025      # deferred cohort, 21.8 GB
```

Cohorts, licences and sizes are in `src/hemishift/data/registry.py`. MOABB downloads go to
`data/raw/moabb`. This also sets MNE's global `MNE_DATA` config on this machine.
