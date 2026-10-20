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
