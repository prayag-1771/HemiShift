# HemiShift: project summary

**Faster calibration of brain-computer interfaces (BCIs) for stroke rehabilitation, using
EEG from other patients and from the patient's own earlier sessions.**

Repository: https://github.com/prayag-1771/HemiShift
Period: 19 October to 13 November 2026 (study complete on 1 November; manuscripts after that).

---

## 1. The problem

- BCI therapy helps arm recovery after stroke. A meta-analysis of 32 randomized trials
  (1187 patients) found +3.85 points on the upper-limb Fugl-Meyer score over conventional
  therapy.
- Every BCI needs a decoder trained on the patient. That calibration takes many trials, and
  patients early after a stroke can rarely complete more than a few.
- With only 5 calibration pairs, a patient's own decoder reaches about 75 to 77% on the main
  cohort, and accuracy varies widely between patients.
- The usual fix is to borrow data from elsewhere: healthy people, simulated data, big
  pretrained models or other patients. No earlier study had compared these sources head to
  head on the same stroke patients.

## 2. Data

No EEG was recorded for this project. Four public cohorts were harmonized into one benchmark.

| Cohort | Source | Patients | Task | Licence |
|---|---|---|---|---|
| S1 | Liu 2024, Beijing | 50 acute (1 to 30 days) | Left vs right hand imagery | CC BY 4.0 |
| S3 | Wang 2026, Tianjin | 24 chronic (3 to 14 months), 3 time points | Affected-hand imagery vs rest | CC BY 4.0 |
| S4 | Liu 2025, Tianjin | 27 (1 to 12 months), 3 sessions | Walking imagery vs idle | CC BY |
| H1 | PhysioNet EEGMMIDB | 109 healthy | Hand imagery | ODC-By |

- **Harmonization:** common channel template, common-average reference, resampling, and the
  lesion flipped to the same side for every patient.
- **Locked test patients:** 8 of the 24 S3 patients and 15 of the S1 patients were set aside
  before any method was tuned. S4 was kept as a fresh validation cohort.
- **Screened and excluded:** HEFMI-ICH (20 haemorrhage patients and 17 controls), checked but
  not used for decoding. UET175 was not used because its archive is password-protected.

## 3. What was built (the full ML pipeline)

| Stage | What was done |
|---|---|
| Preprocessing | Band-pass filtering, epoching, re-referencing, harmonization across devices and hospitals, ICA-cleaned versions for artifact checks |
| Classical ML | Riemannian tangent-space decoders: the patient's own decoder, a pooled decoder recentred per patient, and a band bank (mu, low beta, high beta) |
| Deep learning | EEGNet with Euclidean alignment, pretrained for 60 epochs on donor data and fine-tuned for 40 on the patient |
| Foundation model | CBraMod (a large pretrained EEG model), used frozen with a linear probe |
| Data augmentation | **LesionSim**: turns healthy EEG into stroke-like EEG by editing band envelopes (27 learned settings, 3 modes) |
| Online adaptation | Online recentring (no labels), coadaptive updating (cue labels), online Bayesian model averaging |
| Safety | **SafeTransfer gate**: decides from calibration data alone whether borrowed data can be trusted for a patient |
| Deployment | Real-time streaming over Lab Streaming Layer (LSL) with a live browser dashboard (FastAPI) |
| Statistics | Wilcoxon signed-rank, Holm correction, bootstrap 95% CIs, exact McNemar, Mann-Whitney |

**Tools:** Python 3.11, MNE-Python, MOABB, pyRiemann, scikit-learn, PyTorch and braindecode,
pylsl and FastAPI.

## 4. How it was evaluated

- **Preregistration:** 15 hypotheses were committed to git, with timestamps, before the
  patients they concerned were loaded. Each was tested once.
- **Locked test sets:** never used during development.
- **Leakage audits:** shuffled labels, chronological calibration, trial-order checks,
  shared-patient checks, and unit tests showing that every online decision uses only past
  trials.
- **Artifact checks:** every main claim was rechecked on recordings cleaned with ICA.

**Outcome: 12 of 15 hypotheses confirmed, 2 not confirmed, 1 not tested.** The last one was
not tested because the fixed test order stopped at the hypothesis before it.

## 5. Main results

### The combined decoder (S3, 8 locked patients, 5 calibration pairs)

| Hypothesis | Result | p | Verdict |
|---|---|---|---|
| H-C1: beats own decoder, first session | 77.1% → **88.7%** (+11.6), 8 of 8 improved | 0.0039 | Confirmed |
| H-C2: beats own decoder, later sessions | 75.9% → **88.1%** (+12.2), 8 of 8 improved | 0.0039 | Confirmed |
| H-C3: beats coadaptive recalibration (best older method) | 86.0% → 89.1% (**+3.0**), 7 of 8 | 0.047 | Confirmed |
| H-C4: no labels after calibration, returning patients | 75.9% → 87.7% (+11.9), 8 of 8 | 0.0039 | Confirmed |
| H-C5: no labels at all, earlier sessions only | 75.9% → 84.2% (+8.4), 8 of 8 | 0.0039 | Confirmed |
| H-C6: no labels after calibration, first session | 77.1% → 85.8% (+8.7), 7 of 8 | 0.0078 | Confirmed |

- With updates made only between complete rest–imagery pairs (the audited version), the gain
  is +12.0 points in both the first and later sessions.
- With 10 calibration pairs, the gain is +7.7 points (first session) and +12.5 points (later
  sessions), and every patient improved.

### Building blocks

| Hypothesis | Cohort | Result | p | Verdict |
|---|---|---|---|---|
| Pooling other patients beats own decoder | S3 locked (8) | 77.1% → 81.4% (+4.3), 7 of 8; beats own decoder at 10 pairs | 0.047 | Confirmed |
| H-R1: pooling on a second cohort | S4 (27) | 69.1% → 70.8% (+1.6), 14 helped, 6 harmed | 0.019 | Confirmed |
| H-R2: band bank beats pooling | S4 (27) | 70.8% → 73.1% (+2.3) | 0.021 | Confirmed |
| H-R3: patient's earlier sessions help | S4 (23) | 67.4% → 70.7% (+3.2); +5.4 at follow-up | 0.0045 | Confirmed |
| H-R4: no new labels is not worse than 5 labelled pairs | S4 (23) | 69.7% vs 67.4% | 0.021 | Confirmed non-inferior |

### Second hospital (S1 Beijing, 49 acute patients, left vs right hand)

| Hypothesis | Result | p | Verdict |
|---|---|---|---|
| H-E1: combined decoder beats own decoder | 52.5% → 56.9% (+4.4), 31 helped, 14 harmed | 0.0027 | Confirmed |
| H-E2: beats coadaptive recalibration | 56.2% → 56.9% (+0.7) | 0.17 | Not confirmed |
| H-E3: no labels after calibration | 57.1% (+4.6) | (0.0009, descriptive) | Not tested |
| Window ensemble (S1 locked, 15) | 54.4% → 58.0% (+3.6) | 0.083 | Not confirmed |

### Development data

- S3 development: 81.7% → 91.0% (+9.3). S4: 69.1% → 80.3% (+11.2), 26 of 27 patients
  improved.
- Choices made on one cohort held on the other (+9 to +12 points).
- The method has plateaued at about **90% on S3 and 80% on S4**. A third round of seven more
  ideas added at most 1 point.

## 6. Audits (all passed)

| Audit | Result |
|---|---|
| Shuffled labels | All 27 decoders at chance (47 to 52%); model averaging 48.5% and 49.1%, against 91.0% and 80.3% with true labels |
| Chronological calibration (S4) | 80.6% and 79.1%, +13.6 and +11.9 over own decoder; 27 of 27 and 22 of 23 improved |
| Trial order | Honest margin over coadaptive recalibration corrected from +5.2 to **+3.0** points |
| Shared patients | Without 2 locked patients who may also be in S4: all 6 hypotheses hold (p = 0.016, 6 of 6) |
| ICA-cleaned recordings | All 6 hypotheses hold; margin over coadaptive +2.6 (p = 0.016) |
| Causality | Unit tests confirm that each online decision uses only earlier trials |

## 7. Why it works (neurophysiology)

- **Stroke patients' decision directions agree with each other:** median cosine 0.87 to other
  patients against 0.78 to healthy people (p = 0.0002; closer for 22 of 24). The ceiling
  (split-half of a patient's own data) is 0.92.
- **Stroke motor signals are stronger, not weaker.** Mu desynchronization was −17% to −25% in
  patients against −6% to −17% in healthy people, significant at 11 of 14 channels.
- **Laterality reverses.** Laterality index +0.11 (healthy) vs −0.10 (stroke) in mu, and
  +0.13 vs −0.17 in beta (both p < 0.001).
- **Signals are stable across sessions:** 87.8%, 85.8% and 85.6% at the three time points
  (no significant change). That is why earlier sessions transfer so well.

## 8. Safety gate and live streaming

- **SafeTransfer gate (S3 development, 16 patients):** patients harmed by transfer fell
  **from 9 to 2** (5 pairs, p = 0.008) and **from 11 to 1** (10 pairs, p = 0.001). Mean
  accuracy rose by 4.2 and 8.1 points over always transferring.
- **Streaming = offline:** own decoder 77.1% both ways; pooled decoder 82.5% online vs 81.4%
  offline.
- **Speed:** median 12.6 ms per trial; 21 to 23 ms over real LSL streams under full load
  (maximum 54 ms). A decision is ready about 4 s after each cue.
- **Live demo:** calibrates on 5 rest + 5 imagery trials (about 2.5 minutes), uses 1,282
  donor epochs from 16 patients, and shows the own and pooled decoders side by side in the
  browser.

## 9. What did not work (negative results, reported in full)

| Approach | Result |
|---|---|
| Healthy donors | 78.9%, against 84.0% with patient donors (−5.1 points) |
| LesionSim (simulated stroke) | All 27 settings between 78.5% and 80.0%; no gain over healthy donors |
| EEGNet (deep learning, fine-tuned) | 4.5 to 12.2 points below the own decoder |
| CBraMod (frozen foundation model) | 14 to 24 points below (58.2% to 68.2%) |
| 40 decoder variants (S3 sweep) | Only the band bank (+3.0 and +3.7) and coadaptive updating (88.6% vs 84.0%) helped |
| S1 left vs right hand | Own decoder 57.7%; only 8 of 34 patients at or above the 65% chance threshold; nothing tested beat it |

**Main lesson:** real patient data beats simulated, healthy and pretrained sources.

## 10. Evaluation pitfalls quantified

Each of these produced effects as large as the differences between methods:

1. **Tuning each patient without nested CV:** 72.7% vs 59.2% honest, so 13.5 points of fake
   gain.
2. **Euclidean alignment on common-average data:** amplified noise about a million-fold and
   faked a "negative transfer" result (22 of 34 patients, p = 0.034) until it was fixed.
3. **Look-ahead drift correction:** +5.1 points that vanish in a real session.
4. **Updating after every trial in alternating designs:** inflated the margin from 3.0 to 5.2
   points.
5. **Floating-point ties in rank tests:** turned p = 0.047 into 0.035.

Also found: in imagery vs rest, eye channels alone reach 77.3% and occipital channels 74.3%.
After ICA cleaning, eye channels fall to 55.2% while motor decoding holds at 83.6% to 84.1%.
The baselines published with S1 did not reproduce: the dataset's own code gave 50.6% and
49.3%, against the published 55.6% and 57.6%.

## 11. What is new

1. The first head-to-head comparison of healthy, simulated, pretrained and patient sources on
   the same stroke patients with short calibration.
2. A returning patient's earlier sessions used instead of calibration (+8.4 points with no
   labels at all).
3. An explanation of why other patients help (shared decision directions) and healthy people
   don't.
4. A safety gate that cuts harmful transfer from calibration data alone.
5. One combined decoder: +12 points over the standard method, +3 over the best older method.
6. A preregistered, locked-test evaluation for stroke BCI, with five pitfalls quantified.

## 12. Limitations

- Small locked test set (8 patients) at one site for the combined decoder.
- At the second hospital it beat the own decoder but not coadaptive recalibration.
- Offline only: no closed-loop feedback to patients and no clinical outcome measured.
- The largest gains need the cue label of every trial during the session.
- Healthy and stroke data came from different devices, so part of that gap may be technical.

## 13. Outputs

| Output | Details |
|---|---|
| Manuscripts | IEEE Access version (`paper/ieee/`) and Springer Nature version (`paper/springer/`); about 12,300 words, 9 figures, 3 tables, 39 references (23 from 2022 or later) |
| Code | 25 library modules (1,795 lines), 42 experiment scripts (7,249 lines), 81 unit tests |
| Demo | Live LSL streaming and browser dashboard (`demo/`) |
| Records | Every experiment, including failures, in [ACHIEVEMENTS.md](ACHIEVEMENTS.md); plan in [PLAN.md](PLAN.md); literature in [LITERATURE.md](LITERATURE.md) |
| History | 150+ commits, with each preregistered hypothesis timestamped in git |
