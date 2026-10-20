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
