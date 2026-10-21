# HemiShift — Results Log

Every reported number records its config and split hash (PLAN.md §7). Newest entries first.

## Pre-registration (written 2026-10-20, before any experiment was run)

**H1, the main hypothesis.**
- **Setting:** P4 (leave-site-out), Task A, k = 10 calibration trials per class.
- **Comparison:** R3 (healthy + LesionSim → stroke fine-tune) vs R2 (healthy pretrain → stroke
  fine-tune, the Nagarajan-style recipe), on per-patient balanced accuracy.
- **Claim:** R3 beats R2 in at least 2 stroke cohorts.
- **Test:** Wilcoxon signed-rank, Holm-corrected p < 0.05.

**H2.**
- **Claim:** SafeTransfer lowers the negative-transfer rate compared with always transferring,
  without lowering mean balanced accuracy.
- **Negative transfer:** a patient scores ≥ 2 points below R0/R1 at the same k.
- **"Without lowering":** the bootstrap 95 % CI of the mean difference includes or exceeds 0.

**Metrics.**
- **Primary:** per-patient balanced accuracy.
- **Secondary:** Cohen's κ, AUC (Task B), negative-transfer rate, worst-10 % patient mean, and the
  share of patients above the exact binomial chance threshold for their test-trial count.

**Decision rules.**
- **GP** (week-3 pilot): more patients helped than harmed, and a mean gain of ≥ 1–2 points.
- **G2:** H1 holds.

Anything not listed above is exploratory and will be labelled that way in the paper.

**Amendment, 2026-10-20 (still before any experiment):** the k calibration trials are each
patient's first k trials per class in recording order. Random draws are a sensitivity check
only.

---

## Log

### 2026-10-21 · Premise check on S3 Task B with only 5 + 5 calibration trials (all 24, no tuning)

`premise_check.py --cohort wang2026sf --k 5`; tested on the remaining ~70 epochs per patient.

| Model | Mean | Median | Worst patient |
|---|---|---|---|
| Own 5 + 5 (R0) | 80.8 % | 85.0 % | 47.1 % |
| Pooled | 79.5 % | 84.3 % | 54.3 % |
| Balanced pooling | 81.7 % | 84.3 % | 55.7 % |
| MDWM | 81.1 % | 83.6 % | 60.0 % |

- **Task B is easy enough that 10 trials already give ~81 %,** so the mean has little headroom.
  None of the mean differences is significant (p 0.19–0.61).
- **Transfer lifts the floor.** Among the 6 weakest patients (by their own score), balanced
  pooling gives +5.2 pts (5/6 improved) and MDWM +4.5 (5/6).
  - **Caveat:** selecting by the own score invites regression to the mean. A fair test must
    select weak patients on independent data, as SafeTransfer's calibration-based estimate
    will.
- This is direct support for C3: decide per patient, and favour transfer where the own decoder
  is weak.

### 2026-10-21 · S1 deep-dive step 2: preprocessing (34 dev patients): no significant gain

`scripts/s1_preprocessing.py`; R0 = TS + LR on 8–30 Hz, first 10 + 10 trials, tested on the
last 20. Results in `results/summary/s1_preprocessing_dev.csv`.

| Variant | Mean | vs baseline (Wilcoxon, Holm) | ≥ 75 % |
|---|---|---|---|
| Baseline: core 9, CAR over them | 55.0 % | — | 5 |
| Extended 14 motor channels | 57.1 % | +2.1 mean, p 0.22 (1.0) | 4 |
| 14, CAR over all 29 | 56.5 % | p 0.43 | 5 |
| + interpolate faulty channels | 54.7 % | p 0.96 | 4 |
| + EOG regression | 54.9 % | p 0.95 | 3 |
| Laplacian (CSD) | 53.8 % | p 0.72 | 1 |
| All 29 channels | 56.2 % | p 0.57 | 5 |

- **No preprocessing choice changes S1 Task A significantly.** Together with the audit
  (lateralised signal at chance-level prevalence), every model at ~55 %, and the
  non-reproducible paper, the evidence points to the left/right information being largely
  **absent** from most S1 recordings, not hidden by bad processing.
- **Power problem:** 20 test trials per patient means ±11 points per patient and a ~2-point SE
  across 34. A +2-point effect cannot be resolved with one chronological split. Method
  selection on dev will use blocked 4-fold CV (10 consecutive trials per fold, all 40 trials
  scored). The final locked-test evaluation keeps the realistic first-k protocol.
- **Kept going forward:** the extended 14-channel template (best mean, no harm).

### 2026-10-21 · S1 deep-dive step 1: data audit (34 dev patients, label-free checks)

`scripts/audit_liu2024.py` and `hemishift.preprocess.quality`; results in
`results/summary/audit_liu2024_dev.csv`.
- **Corrupted recordings.** In the raw `.mat` files of patients **5, 13, 15 and 49**,
  neighbouring channels are uncorrelated (median r ≤ 0.08), which real EEG never shows.
  - The EDF versions are intact (r 0.66–0.89), except **patient 13, corrupted in both**.
  - Decisions: the S1 deep-dive reads the EDF. **Patient 13 is excluded** by a label-free rule
    applied to all 50 patients (median adjacent-channel r < 0.3); only 13 fails it.
  - G1 attempt 3 included the four corrupted `.mat` patients. That cannot explain the gap
    (EDF and `.mat` agree on average), but it is noted.
- **Channel faults:** every dev patient has 1–9 technically faulty channels (flat,
  broadband-noisy, or neighbour r < 0.4). 27 of 34 have a faulty *motor* channel. Possible
  pathological slowing appears in 4 patients, and only 4 of 11 slow channels are on the lesion
  side, so most are artifacts rather than lesion slowing.
- **The classic lateralised signal (C3−C4 power by class) is nearly absent:** p < 0.05 in 2 (mu)
  and 4 (beta) of 34, where chance alone gives 1.7; median |d| ≈ 0.2.
- **Rhythm peak above 1/f:** present in only **20 of 34** patients (median 9.6 Hz, range 7.25–12).
  The first estimate's "7 Hz" was a search-edge artifact. 14 patients have no clear mu/alpha
  rhythm at all.
- **EOG power differs by class (|d| > 0.5) in 8 patients**, a possible confound to keep out.
- **The EDF is already notch-filtered** (50 Hz ratio 1.0, vs 7.4 in the raw `.mat`).

### 2026-10-21 · Decodability per cohort and the S1 deep-dive decision

**Within-patient R0** (TS + LR, first 10 + 10 trials, chronological test):

| Cohort | Task | Mean | Above chance |
|---|---|---|---|
| S3 Tianjin (24 chronic) | **B: affected MI vs rest** | **82.9 %** (median 84.7) | **23 / 24** |
| S3 Tianjin | sixth-finger vs affected MI (the paper's task) | 54.2 % | 3 / 24 |
| S5 HEFMI-ICH, 17 healthy | A / B | 52.0 / 51.3 % | 0 / 0 of 17 |
| S5 HEFMI-ICH, 19 ICH | A / B | 54.2 / 55.3 % | 5 / 3 of 19 |
| S1 Beijing (50) | A: left vs right | 55.8 % | 6 / 50 |

- **S3 Task B is strongly decodable.**
- **S3's sixth-finger task:** the paper reports ~85 % with random 10-fold CV, but it falls to
  chance with chronological calibration. Random-vs-chronological is still to be tested on
  identical features.
- **HEFMI-ICH:** even healthy controls are at chance, all 32 channels included. The ERD there is
  slow and weak (healthy 1: −10 to −17 % only at 4–9 s), and a later 4.0–7.5 s MI window gives
  only 52.7 % vs 50.9 %. Its EEG looks weak; it stays a diagnosis cohort for now.
- **Decision (with the user):** a systematic S1 Task A deep-dive, building from the basics.
  1. data audit
  2. preprocessing
  3. strong standard models
  4. our own methods: LesionSim, variability-aware and laterality-aware approaches
- **Guard against overfitting S1:** a **locked test set** of 15 of 50 patients
  (`configs/splits/liu2024_locked_test.json`, hash 4999799d11, stratified by paralysed side ×
  NIHSS > median, seed 20261021). All development uses the 35 dev patients with nested
  validation. The test patients are scored once, at the end. Earlier exploratory runs used all
  50, but no method was tuned on them.

### 2026-10-21 · Gap diagnosis on the core 9-channel template, two cohorts (exploratory)

Rerun with the core template (F3 Fz F4 C3 Cz C4 P3 Pz P4). Results in
`results/summary/gap_comparison_*.csv`.

| Measure | S1 vs H1: stroke / healthy | p (Holm) | HEFMI-ICH (same device): ICH / healthy | p (Holm) |
|---|---|---|---|---|
| Trial-to-trial variability, beta C3 | **0.526 / 0.436** | **0.005** | 0.567 / 0.485 | 1.0 (raw p 0.15) |
| Laterality mu | 0.02 / 0.14 | 1.0 (raw p 0.10) | **−0.13 / +0.32** | 0.28 (raw p 0.025) |
| ERD mu C3 | −19.5 / −21.8 % | 1.0 | +11.1 / −3.3 % | 1.0 |

- **The two cohorts agree in direction:** more variable modulation, and laterality shifted
  toward the contralesional side. They differ in hospital, device and stroke type (acute
  mostly ischaemic vs ICH). Only the S1 variability result survives correction; HEFMI-ICH is
  small (19 vs 17).
- **Caveat 1 (HEFMI patient mapping):** it is unverified (configs/hefmi_ich_patients.csv). That
  the expected laterality direction shows up weakly suggests the mapping is right.
- **Caveat 2 (reference window):** in S1 and HEFMI-ICH the −1.5..−0.5 s reference falls during
  the on-screen cue, which biases ERD toward 0 (HEFMI-ICH healthy ERD only −3 %). The bias is
  shared by both groups within HEFMI-ICH. To do: cohort-specific rest references (HEFMI-ICH
  −8..−3 s; S1 the 2 s break).
- **LesionSim priorities are confirmed:**
  - O4b (per-trial variability)
  - O2 (contralesional shift)

### 2026-10-21 · Premise check: does pooling other patients help? S1, k = 10 per class

`scripts/premise_check.py`: Riemannian tangent space + LR, 8–30 Hz, re-centred; tested on each
patient's last 20 trials.

| Model | Mean | Patients ≥ 75 % (chance threshold, 20 trials) |
|---|---|---|
| Own first 10 + 10 trials (R0) | **55.8 %** | 6 / 50 |
| Pooled: other 49 patients + own (R1 naive) | **55.2 %** | 4 / 50 |

- **Naive pooling gives no gain** (median 0.0, p = 0.81). Over two thousand trials from
  others swamp a patient's 20. This is the transfer problem from the literature, reproduced.
- **A side result:** a Riemannian per-patient model reaches 55.8 %, matching the paper's
  CSP+LDA 55.57 %.
- **Smarter transfer, same 50 patients:**

  | Model | Mean | p vs own | Patients ≥ 75 % |
  |---|---|---|---|
  | Balanced pooling (own trials weigh as much as all others) | 56.1 % | 0.80 | 4 |
  | MDWM (weighted source/target means, λ = 0.5) | 54.6 % | 0.26 | 3 |

  TLRotate was too slow to run: about 13 s per source patient per target, so hours.
- **Conclusion:** on S1 Task A **no transfer variant beats the patient's own 20 trials**. The
  limit is the signal, not the method; most S1 patients carry little left/right information.
  S1 Task A is the wrong place to look for gains. Focus Task B (MI vs rest) on S3/S5, where
  the signal is stronger.

### 2026-10-21 · Gate G1, attempt 3 (authors' code + authors' raw data): resolved, not reproducible

The authors' MATLAB pipeline (ported) on their raw `sourcedata/*.mat` (all 40 trials usable
for all 50 patients). Results in `results/summary/reproduce_liu2024_paper_mat.csv`.

| Pipeline | Raw .mat, mean (95 % CI) | Cleaned EDF | Paper |
|---|---|---|---|
| CSP+LDA | 50.56 % (47.8–53.4) | 51.86 % | 55.57 % |
| FBCSP+SVM | 49.29 % (46.9–51.7) | 49.17 % | 57.57 % |

- **Raw vs cleaned shows no difference** (paired Wilcoxon p = 0.11 and 0.47). The
  eye-artifact hypothesis from attempt 2 is **rejected**.
- **Conclusion:** the published baselines cannot be reproduced from the authors' released
  code and data. We get chance-level decoding, about 50 %.
  - This is consistent with Via et al. 2026: about 49.5–51.8 % on this cohort.
  - Our data handling is verified: structure and counts match, two independent data
    versions agree, and the best patients reach 87.5 %.
- **Decision:** G1 counts as **resolved**. The pipeline is verified, and the
  non-reproducibility is a documented finding for C1 (benchmark section), not a blocker.
  S1 is a genuinely hard, near-chance cohort for standard decoders.

### 2026-10-20 · Gap diagnosis: S1 stroke (50) vs H1 healthy (109) (exploratory)

`scripts/diagnose_gap.py`; results in `results/summary/gap_*.csv`.
- **Setup:**
  - affected-hand MI (healthy: right hand), lesion normalised to the left
  - ERD measured 0.5–4 s vs a reference 1.5–0.5 s before MI onset
  - Mann–Whitney tests, Holm-corrected over 11 measures

| Measure | Stroke median | Healthy median | p | p (Holm) |
|---|---|---|---|---|
| ERD mu C3 (ipsilesional) | −19.2 % | −22.4 % | 0.19 | 1.0 |
| ERD mu C4 (contralesional) | −18.8 % | −15.2 % | 0.79 | 1.0 |
| Laterality mu (C3 vs C4) | −0.00 | +0.14 | 0.087 | 0.87 |
| **Trial-to-trial variability, beta C3** | **0.517** | **0.444** | **0.002** | **0.017** |
| Slow-rhythm asymmetry | −0.02 | +0.05 | 0.34 | 1.0 |

**Findings:**
- **The only difference that survives correction is variability.** Stroke beta modulation over
  the lesioned side is ~16 % more variable from trial to trial. This replicates Via et al.
  2026's "greater variability" on the same cohort.
- **Mean ERD strength does not differ**, matching Via et al.'s null band-power result.
- **There is a trend toward lost laterality:** 40 % of patients have LI < 0 vs 29 % of
  healthy subjects, and the spread is wider (IQR −0.32..0.27 vs −0.08..0.39).
- **Exploratory, uncorrected:** LI_mu rises with NIHSS (Spearman ρ = +0.39, p = 0.013). That is
  the opposite direction to Mansour 2022; S1's NIHSS range is narrow. MBI, mRS and duration
  show no relation. Cortical (n = 10) and subcortical (n = 40) lesions have similar LI.
- **Confound:** device and protocol differ between S1 and H1. HEFMI-ICH (same device for
  patients and controls) is the cleaner test.
- **Consequence for LesionSim (PLAN §4.0 rule 1):**
  - Lead with **trial-to-trial variability** (per-trial ERD gain, O4 timing jitter).
  - Add a small O2 shift.
  - Keep O1 weak, and switch O3 off for S1.

### 2026-10-20 · Gate G1, attempt 2 (authors' MATLAB pipeline, ported): FAIL

Same 50 patients, preprocessed EDF via MOABB. Results in
`results/summary/reproduce_liu2024_paper_edf.csv`.

| Pipeline | Mean (95 % CI) | Median | Paper |
|---|---|---|---|
| CSP+LDA | 51.86 % (48.6–55.2) | 49.1 % | 55.57 % |
| FBCSP+SVM | 49.18 % (46.6–51.7) | 47.5 % | 57.57 % |

- The pipeline now matches the authors' code, so the remaining gap is the **data version**: their
  code reads raw `sourcedata/*.mat`, while MOABB reads the preprocessed EDF ("artifacts
  removed").
- **Hypothesis to test:** lateralised eye or visual artifacts in the raw files, from a hand
  video shown during MI, may inflate left-vs-right decoding.
- **Next:** attempt 3 runs the same pipeline on the raw `.mat` files.

### 2026-10-20 · Gate G1, attempt 1 (MOABB defaults): FAIL

S1 Liu2024, 50 patients, within-patient. Results in
`results/summary/reproduce_liu2024_moabb.csv`.

| Pipeline | 10-fold | 10× 60/40 | Paper |
|---|---|---|---|
| CSP+LDA (8–30 Hz, 6 comps) | 51.05 % | 50.95 % | 55.57 % |
| FBCSP+SVM (9 bands 4–40 Hz, k = 8) | 54.05 % | 52.08 % | 57.57 % |

- We were 3.5–5.5 points short, so the gate failed (±3 allowed).
- Only 10–11 of 50 patients beat the 65 % chance threshold for 40 trials.
- **Cause found.** The authors' MATLAB code (figshare `code.zip`) uses different settings:
  - causal 50 Hz notch plus causal 2nd-order 4–30 Hz Butterworth
  - CSP with 2 pairs on trace-normalised covariances
  - FBCSP with 5 bands from 10 to 25 Hz and all 20 features
  - unstratified random 24/16 splits
  - test features computed as log(1 + var) while training uses log(var)
- The ported pipeline is `--mode paper`; attempt 2 follows.

### 2026-10-20 · Environment and cohorts

**Environment:**
- torch 2.6.0+cu124 and torchaudio 2.6.0+cu124 on CUDA (GTX 1650, 4 GB).
- mne 1.13.2, moabb 1.7.2, braindecode 1.8.1, pyriemann 0.12. Full pin in
  `requirements.lock.txt`.
- braindecode pulls a PyPI torchaudio (2.11) that fails to load with torch 2.6, so torchaudio
  must come from the CUDA index.

**Cohorts confirmed against MOABB 1.7.2:**
- Stroke: S1 `Liu2024` (50), S4 `Liu2025` (27, lower limb), S5 `HefmiIch2025` (20 ICH + 17
  healthy, same device).
- Healthy: H1 `PhysionetMI` (109), H3 `BNCI2014_001` (9).
- MOABB's `Wang2026` is a different, healthy study. S3 (sixth-finger stroke set) gets its own
  figshare download.
- S2 UET175 is cloned at e05a1ae, but the archive is password-protected. A request to the
  authors is drafted, not yet sent.
