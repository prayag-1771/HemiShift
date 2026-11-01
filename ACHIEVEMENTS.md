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

### 2026-11-01 · AUDIT RESULT 2: shuffled labels; every combination decoder falls to chance

`audit_combo.py permute` shuffles the labels within every patient-session (target, donors and
earlier sessions alike) and reruns rounds 1 and 2 unchanged: S3 dev with pair-wise updates,
and S4 (`results/summary/audit_permute_{s3,s4}.csv`). Any leak of labels, trial order or test
data would keep a decoder above 50 %.

- **S3 dev** (16 patients, 3 sessions; 27 decoders): means 47.0–50.3 % at k = 5 and
  47.1–51.7 % at k = 10. `bma` scores 48.5 / 50.1 (91.0 with the real labels).
- **S4** (27 patients, 3 sessions): means 48.0–50.7 % at k = 5 and 47.5–50.6 % at k = 10.
  `bma` scores 49.1 / 49.3 (80.3 with the real labels).
- **Per setting** (first or later sessions × k; script log): the highest decoder is 48.9–52.9 %.
  - **One small p-value:** the plain `pooled` decoder in S4's later sessions at k = 10 scores
    52.3 % (two-sided p 0.007 vs 50 %).
  - **Why it is chance:** it is the highest of 27 decoders in that setting, among 216
    decoder × setting checks (0.007 × 27 ≈ 0.19). It is also not a combination decoder. A
    single shuffle was run.
  - **Every other setting's highest decoder:** p ≥ 0.05.
- **Verdict:** every decoder is at chance. The gains come from the labelled EEG, not from
  the pipeline.

### 2026-11-01 · AUDIT RESULT 1: pair-wise updates; all six locked-test verdicts stand, H-C3's margin is corrected to +3.0

Updating the online decoders only between (rest, MI) pairs (`audit_combo.py pairs`;
`results/summary/audit_pairs_*.csv`) removes any information about a trial's position in
S3's alternation.

- **S3 dev:** the stacks are unchanged. `bma` goes 91.0 → 91.1 (first session) and
  89.7 → 89.9 (later).
- **Locked patients, raw:**
  - H-C1 +12.0 (8/0, p 0.0039)
  - H-C2 +12.0 (8/0, 0.0039)
  - H-C3 +3.0 (7/1, 0.047)
  - H-C4 +11.5 (8/0, 0.0039)
  - H-C5 +7.9 (8/0, 0.0039)
  - H-C6 +8.7 (7/1, 0.0078)
  - **All confirmed.**
- **Locked patients, ICA-cleaned:** all confirmed (p 0.004–0.039). H-C3 is +2.6 (6/0).
- **CORRECTION to H-C3's effect size:**
  - Per-trial updating had handicapped the co-adaptive own decoder (`own_sup`) in S3: it
    learned each rest label just before the paired MI trial. Updated pair-wise, `own_sup`
    reaches 86.0 % instead of 83.5 % on the locked patients.
  - The stack's honest margin over the known co-adaptive method is therefore about 3 points
    (raw +3.0, cleaned +2.6), not the 5.2 first reported. The verdict is unchanged.
  - The pair-wise numbers are the ones to cite.

### 2026-11-01 · RESULT (external test, S1 Beijing): the stack beats the own decoder; not the co-adaptive one

Pre-registered in 484842c and run once with `scripts/s1_combo_external.py`. Data: 49
patients; results in `results/summary/s1_combo_external.csv`.

| | Hypothesis (k = 5) | Result | One-sided p | Verdict |
|---|---|---|---|---|
| H-E1 | `bma` > own | 52.5 → 56.9 % (+4.4); 31 helped, 14 harmed | 0.0027 | **confirmed** |
| H-E2 | `bma` > `own_sup` | 56.2 → 56.9 % (+0.7); 24/17 | 0.17 | not confirmed |
| H-E3 | `pooled_unsup_bank` > own | 57.1 % (+4.6); 31/12 | 0.0009 (descriptive) | not tested (sequence stopped at H-E2) |

- **What the results mean:**
  - At a second hospital, on a different task and in acute patients, the stack still improves
    on the own decoder.
  - Its advantage over co-adaptive recalibration of the own decoder does not carry over:
    `own_sup_bank` alone reached 57.6 %.
  - Absolute accuracy stays low (57–59 %), because left- vs right-hand imagery in acute
    stroke carries little signal (Section "S1 deep-dive").
  - The roughly 89 % results are therefore specific to imagery-versus-rest paradigms in
    chronic patients at Tianjin.
- **Descriptive:**
  - k = 5: `pooled_bank` 56.0, `own_sup_bank` 57.6, `pooled_sup_bank` 56.6.
  - k = 10: own 56.6, `bma` 58.3.

### 2026-11-01 · DECLARED BEFORE RUNNING: combination study, round 3 (exploratory, dev only)

- **Why:** the user asked whether the stack can go further, toward 90 %+.
- **Data:** S3 dev and S4 only. The S3 locked patients are not used. They have served three
  pre-registered analyses, and testing more variants on them would be the overfitting the
  audit guards against. Any round-3 gain therefore stays exploratory until fresh data
  arrive.
- **New decoders:** all co-adaptive with cue labels, recomputed in the same run as round 1.
  S3 uses pair-wise updates (block = 2) throughout.
  - `pooled_sup_bank_fix`: donors keep the calibration's total weight as labelled trials
    accrue, instead of growing with them.
  - `pooled_sup_bank_hl20`: forgetting; labelled trials weigh 0.5^(age / 20 trials).
  - `pooled_sup_bank_all`: donors from the other patients' every session.
  - `pooled_sup_early` and `pooled_sup_late`: co-adaptive pooled banks on 0.5–2.5 s and
    2.0–4.0 s after the cue.
  - `prevoth_sup_bank28_fix` and `prevoth_sup_bank28_hl20`: the same two changes to the
    returning-patient stack.
  - `bma3`: model averaging over round 1's members and every decoder above.
- **Reading:** each new decoder is compared with round 1's `bma` from the same run. A change
  counts as promising only if it beats `bma` in both cohorts and survives cross-cohort
  selection.

### 2026-11-01 · PRE-REGISTRATION: the combination decoders at a second hospital (S1, Beijing; run once)

- **Why:** the combination was confirmed on 8 patients from one hospital, Tianjin. S1 differs
  in four ways:
  - hospital and amplifier (a portable 29-channel system);
  - stage (acute stroke);
  - task (affected- vs unaffected-hand imagery instead of imagery vs rest);
  - one session per patient.
- **Data:** all 49 usable S1 patients, with patient 13 excluded as before.
  - The combination decoders were never run on S1.
  - These patients were used for other decoders before (development, and two pre-registered
    locked-test analyses), so they are tested together, like S4.
- **Frozen decoders:** `scripts/s1_combo_external.py` with `hemishift.models.combo`, on the
  14-channel template, because S1 lacks the 28-channel strip.
  - `bma` averages the 14-channel members of the S3 system: `pooled_bank`, `own_sup_bank`,
    `pooled_sup_bank` and `prior_sup_bank`.
  - S1's trials alternate between the hands, so the online decoders and the averaging weights
    update only between pairs (block = 2).
- **Protocol:** P3, the first k trials per class; k = 5 primary, 10 descriptive.
- **Hypotheses:** one-sided Wilcoxon over patients, fixed sequence at α = 0.05.
  - H-E1: `bma` > own.
  - H-E2: `bma` > `own_sup` (the known co-adaptive decoder).
  - H-E3: `pooled_unsup_bank` > own.
- **Expectation, stated in advance:** S1's left/right signal is weak; standard decoders reach
  about 55 %. A null result is possible and will be reported as a limit of the method on
  left/right imagery in acute stroke.
- **Dry run:** 4 patients with shuffled labels, a code check only.

### 2026-11-01 · DECLARED BEFORE RUNNING: leakage and overfitting audit of the combination study

The user asked for proof that the results carry no overfitting and no data leak. Five checks
follow.

1. **Order leak in alternating paradigms.**
   - The problem: S3 alternates rest and MI. The online decoders were updated before every
     trial, so before each MI trial they had just seen one extra rest trial. Online
     re-centring could then tilt toward calling the next trial MI. This information comes
     from the trial's position, not from the brain.
   - The fix: `block = 2` updates the decoders, and the model-averaging weights, only between
     (rest, MI) pairs, so every update sees a balanced set.
   - The check: rerun every decoder this way on S3 dev and on the S3 locked patients, raw and
     cleaned. Compare with the per-trial results and with the six hypothesis tests.
   - S4's trial order is randomised, so it is not affected.
2. **Label permutation.** Every patient's labels are shuffled within each session, donors and
   earlier sessions included. Every decoder is then rerun on S3 dev (pair-wise updates) and on
   S4. Any decoder clearly above 50 % would reveal a leak.
3. **Strictly chronological calibration on S4.**
   - The problem: P3's "first k trials per class" leaves a few test trials before the last
     calibration trial when the trial order is random.
   - The check: every S4 decoder is rerun with all trials up to the one that completes k per
     class used for calibration. S3's pairs are already strictly chronological.
4. **Code checks.**
   - Unit tests (11, `tests/test_combo.py`) show that no decision changes when later trials'
     labels or recordings change, for per-trial and pair-wise updates.
   - Assertions in `combo_explore.run` check that the target patient is never among its own
     donors.
   - The split files are re-hashed, and the cached data are checked to hold only the
     intended patients.
5. **Independent recomputation** of every confirmatory p-value from the saved per-patient
   results, and a check for patients shared between S3 and S4 (same institution), using the
   participants tables.

If the pair-wise rerun changes a locked-test verdict, the change will be reported as a
correction of the 2026-11-01 result. A second external test (S1, another hospital) will be
pre-registered separately, with whichever update rule the audit supports.

### 2026-11-01 · RESULT (confirmed): the combination decoders on the S3 locked test; all six hypotheses hold

Pre-registered in ec5cd82 and run once with `scripts/s3_locked_combo.py`. Data: 8 locked-test
patients (pre 8, post 8, follow-up 6), raw recordings. Results in
`results/summary/s3_locked_combo.csv`.

| | Hypothesis (k = 5) | Result | One-sided p | Verdict |
|---|---|---|---|---|
| H-C1 | `bma` > own, first session | 77.1 → **88.7 %** (+11.6, median +11.4); 8 helped, 0 harmed | 0.0039 | **confirmed** |
| H-C2 | `bma` > own, later sessions | 75.9 → **88.1 %** (+12.2, median +9.6); 8/0 | 0.0039 | **confirmed** |
| H-C3 | `bma` > `own_sup` (known co-adaptive), first session | 83.5 → 88.7 % (+5.2, median +5.7); 7/1 | 0.0195 | **confirmed** |
| H-C4 | `prevoth_unsup_bank28` > own, later sessions | 75.9 → 87.7 % (+11.9); 8/0 | 0.0039 | **confirmed** |
| H-C5 | `prevzero_unsup_bank` > own, later sessions | 75.9 → 84.2 % (+8.4, median +5.4); 8/0 | 0.0039 | **confirmed** |
| H-C6 | `pooled_unsup_bank` > own, first session | 77.1 → 85.8 % (+8.7); 7/1 | 0.0078 | **confirmed** |

- **What the results mean:**
  - With cue labels during the session, the full stack gains about 12 points on patients it
    was never developed on, in their first session and when they return. That is about 5
    points more than the known co-adaptive decoder.
  - Without labels after calibration, returning patients still gain 12 points.
  - With no labels at all in the returning session, the gain is 8 points.
- **Descriptive (k = 10):** `bma` +7.7 (first session, 8/0) and +12.5 (later, 8/0);
  `prevoth_unsup_bank28` +12.1 (later, 8/0).
- **Caveats:**
  - There are only 8 patients, all from one site.
  - The co-adaptive decoders need the cue label of every trial, which a cue-based rehab
    session has.
  - The S3 locked set has now been used for three pre-registered analyses.
- **ICA-cleaned sensitivity analysis (declared, run once;
  `results/summary/s3_locked_combo_clean.csv`):** all six hold on the authors' cleaned
  recordings of the same patients, with dev patients' cleaned recordings as donors.

| | Cleaned result (k = 5) | One-sided p |
|---|---|---|
| H-C1 | 76.6 → 87.1 % (+10.6), 7/1 | 0.020 |
| H-C2 | 73.9 → 86.7 % (+12.8), 8/0 | 0.0039 |
| H-C3 | 82.2 → 87.1 % (+4.9), 7/0 | 0.0078 |
| H-C4 | 73.9 → 86.3 % (+12.4), 6/0 | 0.016 |
| H-C5 | 73.9 → 83.4 % (+9.5), 7/1 | 0.012 |
| H-C6 | 76.6 → 83.5 % (+6.9), 6/2 | 0.039 |

  The gains are therefore not eye or muscle artifacts that ICA removes.

### 2026-11-01 · PRE-REGISTRATION: the combination decoders on the S3 locked test patients (run once)

- **What is tested:** the combination decoders of `scripts/combo_explore.py` (rounds 1 and 2),
  frozen in this commit, on the 8 S3 locked-test patients.
  - The decoders were developed on the 16 S3 dev patients and the 27 S4 patients only.
  - None of them has been run on these 8.
- **Earlier use of these patients:**
  - Pre sessions: the pooling confirmation (2026-10-24) and the declared ICA sensitivity check
    (2026-10-31).
  - All sessions: `s3_neuro`, descriptively (own TS + LR, blocked CV, all 24 patients).
  - This is the third pre-registered use of the S3 locked set.
- **Setting:**
  - Sessions: pre (8 patients), post (8) and follow-up (6).
  - Donors: the 16 dev patients' same session. A returning patient's earlier sessions are its
    own.
  - Protocol P3, k = 5 primary and 10 descriptive.
  - "Later sessions" is each patient's mean over post and follow-up.
  - Script: `scripts/s3_locked_combo.py`, run once. Its `--dry-run` was checked on dev
    patients 2 and 3 only.
- **Hypotheses:** one-sided Wilcoxon signed-rank over patients, k = 5. Fixed sequence at
  α = 0.05: each is tested only if every earlier one was confirmed. The dev evidence is S3 /
  S4 at k = 5, with S3 ICA-cleaned last.

| | Hypothesis | Dev evidence (gain vs comparator, helped/harmed) |
|---|---|---|
| H-C1 | `bma` > own, pre session | +9.3 (14/1) / +11.2 (26/1); cleaned +6.2 (14/1) |
| H-C2 | `bma` > own, later sessions | +10.3 (16/0) / +11.6 (21/1); cleaned +7.4 (16/0) |
| H-C3 | `bma` > `own_sup` (the known co-adaptive decoder), pre session | +3.7 (13/2) / +5.1 (24/3); cleaned +2.2 |
| H-C4 | `prevoth_unsup_bank28` > own, later sessions (no labels after calibration) | +8.8 (15/0) / +9.8 (20/3) |
| H-C5 | `prevzero_unsup_bank` > own, later sessions (no labels at all in the later session) | +6.2 (15/0) / +6.2 (16/5); cleaned +3.8 (10/5) |
| H-C6 | `pooled_unsup_bank` > own, pre session (no labels after calibration) | +6.5 (13/0) / +5.3 (22/4); cleaned +3.5 (11/4) |

- **Raw and cleaned:**
  - The verdicts rest on the raw recordings, as in development.
  - A declared sensitivity analysis, also run once, applies the same tests to the authors'
    ICA-cleaned release of the same patients, with the dev patients' cleaned recordings as
    donors.
- **Power:** with 8 patients the smallest one-sided p is 1/256 = 0.004. A single harmed patient
  with a large loss can sink a test.
- **Rules:** nothing in the decoders or settings changes after this commit. Any later change
  needs fresh data.

### 2026-11-01 · RESULT (exploratory): combination study, rounds 1 and 2, S3 dev and S4

Written up after the pre-registration above. The per-patient results were committed before it
(31d1724, e6209f9, ec5cd82): `results/summary/combo_<cohort>[_r2].csv`. Mean accuracy at
k = 5, with the gain over own and helped/harmed:

| Decoder | Needs during the session | S3 pre | S4 pre | S3 later | S4 later |
|---|---|---|---|---|---|
| own (baseline) | – | 81.7 | 69.1 | 79.5 | 67.4 |
| `pooled_bank` (H-R2) | – | 86.6 (+4.9) | 73.1 (+3.9) | 83.7 | 70.9 |
| `prevoth_bank28` | – | | | 87.5 (+8.0, 14/1) | 75.4 (+8.0, 19/4) |
| `pooled_unsup_bank` | – (re-centres on unlabelled trials) | 88.2 (+6.5, 13/0) | 74.4 (+5.3, 22/4) | 85.3 | 72.4 |
| `prevoth_unsup_bank28` | – (re-centres) | | | 88.3 (+8.8, 15/0) | 77.2 (+9.8, 20/3) |
| `prevzero_unsup_bank` | nothing: no labels at all | | | 85.7 (+6.2, 15/0) | 73.6 (+6.2, 16/5) |
| `own_sup` (known co-adaptive) | cue labels | 87.3 (+5.7) | 75.2 (+6.1) | 84.5 (+5.0) | 73.0 (+5.5) |
| `pooled_sup_bank` | cue labels | 90.4 (+8.7, 13/1) | 79.5 (+10.4, 25/1) | 87.3 | 76.6 |
| `prevoth_sup_bank28` | cue labels | | | 89.9 (+10.4, 15/0) | 79.1 (+11.6, 21/2) |
| `bma` | cue labels | **91.0 (+9.3, 14/1)** | **80.3 (+11.2, 26/1)** | **89.7 (+10.3, 16/0)** | **79.1 (+11.6, 21/1)** |

- **The gains add up.** With cue labels arriving during the session, the stacks gain 9–12
  points over the own decoder in both cohorts.
- **Beyond the known co-adaptive method.** `bma` beats `own_sup` by:
  - S3 pre +3.7 (13/2), S3 later +5.3 (14/1);
  - S4 pre +5.1 (24/3), S4 later +6.1 (20/3).
  - In S4, the bank alone adds +3.4 under co-adaptation.
- **Without labels during the session:**
  - For returning patients, the stack of earlier sessions + other patients + 28 channels +
    online re-centring gains 9–10 points.
  - The no-label decoder from earlier sessions alone (`prevzero_unsup_bank`) beats the
    5-pair own decoder by 6 points in both cohorts.
- **Selection without hindsight agrees:**
  - Leave-one-patient-out picks score within 1 point of the best decoder.
  - Picks made on S3 score on S4: +11.2 (pre) and +11.6 (later) with cue labels; +5.3 and
    +9.8 without labels.
  - Picks made on S4 score on S3: +9.3 and +10.3 with cue labels.
- **Failed:**
  - `prior_bank` and `prior_sup_bank` (empirical-Bayes prior LR) were worse than pooling
    trials in every setting.
  - Sub-window crops gave no gain.
  - Label-free decoders that add other patients (`prevothzero_*`) were weak on S4 (+2 to +3).
  - Second round (`combo_posthoc.py`): unsupervised bias adaptation adds less than 1 point;
    plain probability averages do not beat `bma`.
- **ICA-cleaned S3 release** (dev; `combo_s3clean.csv`):
  - The co-adaptive stacks keep most of their gain: `bma` +6.2 (pre, 14/1) and +7.4 (later,
    16/0); `prevoth_sup_bank28` +8.6 (later, 15/1).
  - The no-label stacks keep about half: `pooled_unsup_bank` +3.5, `prevoth_bank28` +3.8,
    `prevzero_unsup_bank` +3.8.
  - Part of their raw gain is probably artifact-related.
  - The exception is round 2's `prevoth_unsup_bank28` (later sessions, no labels after
    calibration), which keeps +7.3 (15/1) of its raw +8.8.
- **Caveat:** all of this is exploratory, and S4 was used before (2026-10-31). The
  confirmation is the locked-test run pre-registered above.

### 2026-11-01 · DECLARED BEFORE RUNNING: artifact check of the combination study (S3 dev, ICA-cleaned)

- **Why:** round 1 found co-adaptive bank stacks 9–12 points above the own decoder in both
  cohorts. In S3, imagery vs rest is partly separable by eye and visual activity. A decoder
  that keeps updating could learn that artifact as well as motor activity.
- **What:** every round 1 and round 2 decoder is run on the authors' ICA-cleaned release of
  the 16 dev patients, all three sessions ("s3clean").
  - Same 14- and 28-channel groups, each re-referenced to its own average. The channels are
    taken from the 42 common to both releases.
  - Each release keeps its own trials, because cleaning cut some segments.
- **Reading:** a decoder's gain over the own decoder should stay similar on cleaned data. A
  gain that shrinks a lot is counted as partly artifact.

### 2026-11-01 · DECLARED BEFORE RUNNING: combination study, round 2

- **Why:** in round 1 on S3 (S4 still running), the best decoder without labels during the
  session was a static stack: earlier sessions + other patients, 28 channels. Separately,
  unsupervised re-centring added about 1.6 points to the pooled bank. Round 1 never combined
  the two.
- **New decoders** (both cohorts, same protocol):
  - `pooled_unsup_bank28`: `pooled_unsup_bank` on 28 channels.
  - `prevoth_unsup_bank` and `prevoth_unsup_bank28`: earlier sessions + other patients,
    labels from the calibration only, re-centred on every epoch seen so far.
  - `prevothzero_bank28` and `prevothzero_unsup_bank28`: no patient labels at all (earlier
    sessions + other patients, 28 channels), either static or re-centred online.
- **Selection:** they join round 1's candidates in the family "no labels after calibration".
  Selection without hindsight works as before.

### 2026-11-01 · DECLARED BEFORE RUNNING: combination study (exploratory; S3 dev and S4)

- **Why:** each confirmed gain is a few points. This study asks whether they add up, and
  tries new ways of combining them.
- **Data:**
  - S3 dev: 16 patients, pre / post / follow-up.
  - S4: 27 patients, pre / post / follow-up. Its confirmatory tests are spent (2026-10-31),
    so S4 now counts as development data.
  - The S1 and S3 locked tests are not touched.
- **Status:** every result here is exploratory. A combination found here can only be
  confirmed on fresh data (UET175 or another untouched cohort), under a new pre-registration.
- **Protocol:** unchanged.
  - The first k pairs (S3) or the first k trials per class (S4) calibrate; k = 5 is primary,
    10 secondary.
  - Every decoder is scored on the same test trials.
  - Online decoders take the test trials in recording order. A trial's cue label is used
    only after that trial has been decoded.
- **Candidates, fixed before running:**
  - *Static:* own, pooled and the mu / low-beta / high-beta bank, on the 14-channel template
    or the 28-channel central strip. In later sessions also: earlier sessions, earlier
    sessions + other patients, and the no-label decoder, each with the bank.
  - *Online, supervised:* co-adaptive versions of the above, refitted before each test trial.
  - *Online, unsupervised:* running re-centring on every epoch seen so far (labels from
    calibration only, or none at all).
  - *New: population-prior LR.* The patient's decoder is fitted with an L2 penalty that
    pulls its weights toward the donors' decoder, instead of pooling samples. Online, the
    prior keeps a fixed weight while the patient's own labelled trials accumulate.
  - *New: online Bayesian model averaging* over decoders. Weights are updated by each
    decoded trial's log-loss under its cue label.
  - *Crop augmentation:* covariances of 2-s sub-windows as extra calibration samples.
- **Baselines:**
  - own (8–30 Hz, 14 channels, P3);
  - the best confirmed decoder: the pooled bank, and earlier sessions in later sessions.
- **Selection without hindsight:**
  - leave-one-patient-out within each cohort;
  - across cohorts: pick on S3 dev and report on S4, and the reverse.
- **Reporting:** every candidate, with mean gain, helped/harmed and two-sided Wilcoxon p,
  Holm-corrected over the candidates of each cohort and setting.
- **Channels:** motor area only (14-channel template or 28-channel central strip). 59-channel
  numbers are descriptive only, because non-motor channels carry eye and visual signal.

### 2026-10-31 · CORRECTION: floating-point ties in the Wilcoxon tests; p-values recomputed

- **The error:**
  - Accuracy differences that are equal in exact arithmetic (e.g. four patients each gaining
    1 of 70 test trials) came out of floating point a few ulps apart. scipy then ranked them
    apart instead of as ties.
  - True zeros between two-session means (S4 H-R3, H-R4) survived as ±1e-15 and counted as
    tiny gains or losses.
  - Found while checking that figure 7's p-value matched the S4 log.
- **The fix:**
  - `stats.clean_differences` rounds paired differences to 1e-9 before ranking. Every
    Wilcoxon test now goes through `stats.signed_rank_p`.
  - A unit test pins the S3 locked-test case to the exact 12/256.
  - `scripts/recheck_stats.py` and the scripts' `--stats-only` modes recompute every logged
    value from the saved per-patient results. No decoder was re-run.

| Confirmatory test | Logged | Corrected | Verdict |
|---|---|---|---|
| S3 locked test, pooled > own, k = 5 (pre-registered) | 0.035 | **0.047** | still confirmed (< 0.05), with less margin |
| S3 locked test, k = 10 (secondary) | 0.14 | 0.17 | not significant, as before |
| S1 locked test, ens16 | 0.083 | 0.083 | unchanged |
| S4 H-R1 | 0.018 | 0.019 | confirmed |
| S4 H-R2 | 0.012 (Holm 0.021) | 0.011 (Holm 0.021) | confirmed |
| S4 H-R3 | 0.0016 (Holm 0.005); 16 helped / 5 harmed | 0.0015 (Holm 0.0045); 15 / 4 | confirmed |
| S4 H-R4 | 0.010 (Holm 0.021) | 0.010 (Holm 0.021) | confirmed |
| S3 locked-test sensitivity (six tests) | | unchanged | |

- **Development values quoted in the paper:**
  - S3 dev pooling: k = 5 p 0.004 → 0.003 (Holm 0.018 → 0.012); k = 10 0.16 → 0.20.
  - S1 ens16 dev: 0.011 → 0.012 (Holm 0.12 → 0.13).
  - S1 block re-centring: 0.021 → 0.016.
  - S1 zero-shot: 0.017 → 0.015.
  - LesionSim nested vs identity: 0.80 → 0.86.
  - SafeTransfer gate vs always transferring (two-sided): 0.033 → 0.032; k = 10 still 0.001.
- **Today's dev entries** (sweep, cross-session, cleaned release) were recomputed and are
  corrected in place below. No conclusion changes.

### 2026-10-31 · RESULT (confirmed): S4 Liu2025 replication; all four pre-registered hypotheses hold

`scripts/s4_replication.py`, run once after the download, as pre-registered (d0800e8,
amended in 0d9b750, 34f0e28 and 494d526). Results in `results/summary/s4_replication.csv` and
`results/summary/s4_descriptive.csv`. The p-values and counts below are tie-corrected (see the
correction above).
- **Data:** 27 patients, gait MI vs idle, the confirmed S3 code unchanged.
- **Exclusions, as pre-registered:**
  - The post and follow-up sessions of patients 3, 6, 8 and 11 are not in the release
    ("cannot be read").
  - Patient 26's follow-up has 10 trials per class, below the minimum of 20.
  - So H-R3 and H-R4 use 23 patients.

| | Comparison (k = 5) | Patients | Result | One-sided p | Holm | Verdict |
|---|---|---|---|---|---|---|
| H-R1 | pooled vs own, pre session | 27 | 70.8 vs 69.1 % (+1.6, 95 % CI +0.2 to +3.1; 14 helped / 6 harmed) | 0.019 | gatekeeper | **CONFIRMED** |
| H-R2 | band-bank pooled vs pooled, pre session | 27 | 73.1 vs 70.8 % (+2.3, CI +0.5 to +4.1; 17/6) | 0.011 | 0.021 | **CONFIRMED** |
| H-R3 | own earlier sessions vs own, later sessions | 23 | 70.7 vs 67.4 % (+3.2, CI +1.5 to +5.1; 15/4) | 0.0015 | 0.0045 | **CONFIRMED** |
| H-R4 | no-label earlier-session decoder vs own, margin 2 points | 23 | 69.7 vs 67.4 % (+2.3, CI −1.2 to +5.7; 15/7) | 0.010 | 0.021 | **CONFIRMED** (non-inferior) |

- **Pooling replicates on a new cohort, a new limb and a new trial design.** The trials here
  are randomised imagery and idle, not rest-then-imagery pairs. The gain is smaller than on
  S3 (+1.6 vs +4.3 points), at a lower accuracy level (own 69 vs 77 %).
- **The band bank adds +2.3 points on top: +4.0 over own.** The H-R2 decoder was chosen on S3
  dev and kept free of the theta band.
- **The patient's own earlier sessions help most at follow-up, as on S3 dev:**
  - follow-up +5.4 (17/2), post +1.5
  - at k = 10, +4.7 and +2.7
- **No labels at all:** the earlier sessions' decoder (re-centred on the calibration epochs,
  labels unused) is not worse than the own decoder with 5 labelled pairs. Mean +2.3; at
  k = 10, +0.7 vs own with 10 pairs.
- **Descriptive (no test):**
  - k = 10: pooled +2.2 over own; bank +3.2 over pooled; earlier sessions +3.5.
  - Co-adaptive decoders (refitted after every trial with its cue label): own 75.2 % (+6.1),
    pooled 75.9 %.
  - No-label decoder from the other patients only: 68.5 % vs own 69.1 % (−0.7).
  - fb4 (with theta) pooled: 74.6 % (+3.8 over the confirmed pooled).
  - 59-channel bank pooled: 77.9 % (+7.2; 21 helped / 1 harmed).
  - Confound controls (own / pooled): occipital 63.7 / 63.9 %; frontal 1–8 Hz 64.9 / 67.8 %;
    strict motor strip 67.8 / 69.7 %. Non-motor channels carry less here than in S3 (frontal
    77 %), but some.
- **Caveat:** same institution and amplifier family as S3, and some patients may be in both
  cohorts. No S3 data were used here.

### 2026-10-31 · DECLARED BEFORE RUNNING: sensitivity of the confirmed S3 result to ICA cleaning

The dev check above found that, on the authors' ICA-cleaned release, the single-band pooled
decoder loses its edge. A reviewer will ask whether the confirmed locked-test result
survives cleaning, so it is checked once on the 8 locked-test patients.
- **Status:** this is a sensitivity analysis of an existing claim, not a new hypothesis. It
  cannot add a confirmed result. It is reported whatever the outcome.
- **Data:** it is the locked-test patients' second use (pre session, cleaned release instead of
  raw). Donors are the 16 dev patients' cleaned recordings.
- **Pipeline:** the confirmed comparison unchanged: own vs pooled, 14 channels, 8–30 Hz,
  P3 with k = 5 (and 10). The band-bank pooled decoder (H-R2's) is reported alongside.
- **Script:** `scripts/s3_locked_sensitivity.py`.

**Result (run once).** Results in `results/summary/s3_locked_sensitivity.csv`. The cleaned
release keeps 52–80 epochs per patient.

| k | own 8–30 | pooled 8–30 (the confirmed decoder) | pooled band bank |
|---|---|---|---|
| 5 | 76.6 % | 78.2 % (+1.6; 5/2; one-sided p 0.29) | **80.0 %** (+3.5 vs own; 5/1; p 0.031) |
| 10 | 78.6 % | 79.2 % (+0.7; p 0.22) | **83.3 %** (+4.8 vs own; 5/0; p 0.031; +4.1 vs pooled 8–30, p 0.016) |

- **The confirmed single-band gain shrinks on cleaned data:** +4.3 on the raw recordings,
  +1.6 here (not significant).
- **The band-bank pooled decoder keeps the gain on cleaned data**, as it did on dev.
- Reading: pooling helps few-trial calibration in both versions, but on cleaned data it needs
  the band bank. This is the paper's main robustness limitation of the confirmed result. It
  is also independent support for S4's H-R2 decoder (descriptive; not a confirmation).

### 2026-10-31 · S3 artifact check on the authors' ICA-cleaned release (dev)

`scripts/s3_processed_check.py`. Results in `results/summary/s3_processed_check_dev.csv`.
- **Why the authors' release:** the raw EOG channels cannot be regressed out of the EEG.
  - In most runs they are flat (0 µV) or far outside the physiological range (hundreds to
    thousands of µV after 1–40 Hz filtering).
  - They rarely correlate with Fp1/Fp2 (scan of all 16 dev patients). MNE's EOG regression
    fails on them (singular matrix).
- **The release:** 43 channels, average reference, ICA, artifact segments cut out. Some
  patients lose trials (e.g. 48 of 80 epochs remain for patient 21).
- **Setup:** the same decoders on both versions, each on its own trials, 42 common channels.
  k = 5 shown (own / pooled):

| Decoder | Raw recordings | Cleaned release | Cleaned: pooled − own |
|---|---|---|---|
| frontal 1–8 Hz (eye control) | 77.3 / 77.2 % | **55.2 / 58.2 %** | |
| posterior 8–30 Hz (PO row) | 71.7 / 72.2 % | 72.2 / 72.8 % | |
| motor 8–30 Hz (the confirmed decoder) | 81.7 / 84.0 % | 83.6 / 84.0 % | **+0.4** (median 0, 5/5) |
| motor mu/low-beta/high-beta bank (H-R2) | 83.4 / 86.6 % | 84.1 / 86.0 % | +1.9 (median +1.5, p 0.020, 9/1) |
| motor fb4 (with theta) | 84.0 / 87.0 % | 83.7 / 86.6 % | +2.9 (median +2.9, p 0.012, 12/1) |

- **The motor decoders are not eye-driven.** Cleaning removes the eye signal (frontal
  control 77 → 55–58 %), yet motor decoding stays at 84–87 %.
- **On cleaned data the single-band pooled decoder loses its edge at k = 5.** Cleaning raises
  the own decoder (81.7 → 83.6 %) while pooled stays at 84.0 %.
  - On raw data, part of the pooling benefit may therefore be robustness to artifacts in the
    few calibration trials.
- **With the band bank, pooling still helps on cleaned data:**
  - bank: +1.9 over own (p 0.020); +2.0 over the single-band pooled decoder (p 0.034,
    tie-corrected)
  - fb4: +2.9 over own; +2.5 over the single-band pooled decoder (p 0.007)
  - So theta's contribution is not just eye activity, which ICA removed.
- Posterior alpha still separates rest from imagery after cleaning (72 %): visual or
  attentional, not ocular.
- **For the paper:** report the confirmed result as obtained on raw recordings, and this check
  as its main limitation. The band-bank pooled decoder (H-R2) is the version that is robust to
  cleaning.

### 2026-10-31 · S3 sweep repeated on the dev patients' post-treatment sessions (dev, exploratory)

`scripts/s3_sweep.py --phase post`. Results in `results/summary/s3_sweep_post_dev.csv`.
- **Setup:** the same 16 dev patients, their post-training session. Donors are the other dev
  patients' post sessions.
- **Baseline:** own 80.7 %, pooled 82.1 % at k = 5.

| Variant (vs pooled) | pre, k = 5 | post, k = 5 | post, k = 10 |
|---|---|---|---|
| 59 channels + mu/beta/8–30 bank | +2.5 | **+4.8** (p 0.007; 12/2) | +3.2 |
| fb4 (theta, mu, low and high beta), 14 channels | +3.0 | +3.7 (p 0.004; 13/2) | +3.1 (p 0.001, Holm 0.039) |
| fb4 without theta (H-R2's decoder) | +2.6 | +1.1 (p 0.32) | +1.6 (p 0.046) |
| co-adaptive pooled (online protocol) | +4.6 | +3.8 (Holm 0.027) | +2.0 |

- **The ranking replicates across sessions.** Band banks and more channels lead in both.
  Donor weighting, donor selection, stretching, ensembles and mixup do nothing in either.
- **Honest selection now gains:** 85.1 vs 82.1 % at k = 5 (median +2.1, p 0.020). The
  59-channel bank is picked in 14 of 16 folds.
- **But the larger gains lean on bands and channels outside the motor signal.**
  - Theta matters more in the post session: fb4 gains +3.7, but only +1.1 without theta.
    Theta is where eye activity separates rest from imagery.
  - The 59-channel decoder includes occipital channels, which alone decode at 77–80 % here.
- **Decision:** H-R2 stays as registered (motor channels, no theta). That is the variant a
  reviewer cannot put down to eyes or vision. fb4, the 59-channel bank and the confound
  controls are added to the S4 run as descriptive results only, before any S4 decoding.

### 2026-10-31 · AMENDMENT to the S4 pre-registration (committed before any S4 decoding)

The S3 sweep and the confound check (both below, dev only) fix the remaining hypotheses. No
Liu2025 EEG has been decoded; the download is still running.
- **H-R2:** a pooled decoder with a mu / low-beta / high-beta bank (8–13, 13–20, 20–30 Hz;
  the three bands' tangent vectors side by side; otherwise identical) beats the confirmed
  pooled decoder in the pre session at k = 5.
  - This is the sweep's best variant (fb4) without its theta band. The confound check showed
    that theta is where eye activity separates rest from imagery.
  - Dropping theta keeps 2.6 of fb4's 3.0 points on dev.
- **H-R3:** the patient's own earlier sessions as donors beat own in the later sessions at
  k = 5 (post with pre; follow-up with pre + post). The score is each patient's mean over the
  two later sessions.
- **H-R4:** the earlier sessions' decoder with no labels is not worse than own at k = 5 by
  2 points or more (non-inferiority; one-sided Wilcoxon on the difference plus 2).
  - The later session is re-centred on its calibration epochs, whose labels are not used.
  - The 2-point margin is the pre-registered negative-transfer margin.
- **Error control (replaces "Holm over H-R1 and H-R2"):** fixed sequence. H-R1 is tested at
  α = 0.05. Only if it is confirmed are H-R2 to H-R4 tested, with Holm at α = 0.05.
- **Descriptive only:** k = 10; earlier sessions plus other patients; co-adaptive own and
  pooled decoders (refitted after every trial with its cue label). Also, added before any S4
  decoding: a no-label decoder from the other patients only (re-centred on the calibration
  epochs, labels unused).

### 2026-10-31 · S3 confound check: non-motor channels also separate imagery from rest (dev)

`scripts/s3_confound_check.py`. Results in `results/summary/s3_confound_check_dev.csv`. Each
channel group is re-referenced to its own average.

| Channels | own / pooled, k = 5 |
|---|---|
| motor template, 8–30 Hz (the decoders' input) | 81.7 / 84.0 % |
| strict motor strip (FC, C, CP rows), 8–30 Hz | 81.1 / 82.1 % |
| occipital, 8–30 Hz | 74.3 / 76.1 % |
| frontal (eyes), 1–8 Hz | 77.3 / 77.7 % |
| frontal, 8–30 Hz | 60.3 / 62.4 % |
| temporal (muscle), 8–30 Hz | 66.1 / 68.1 % |

- **In S3, rest and imagery also differ outside motor cortex:** in eye activity (frontal slow
  waves, 77 %) and in posterior alpha (74–76 %). The sweep's occipital control (79 %) was
  partly the 59-channel common average spreading the motor ERD to every channel.
- **The motor decoders are unlikely to be eye-driven.** Frontal 8–30 Hz alone reaches only
  60 %. The strict motor strip, away from the eyes and visual cortex, keeps 81–82 %, and
  pooling still helps there.
- **The filter-bank gain is not the theta band:**
  - pooled fb4 87.0 %; fb4 without theta 86.6 %; theta alone 77.2 %.
  - On the strict motor strip, the bank without theta still adds 1.6 points over 8–30 Hz.
- **For the paper:** Task B accuracy is not purely motor in this paradigm. Report it, and keep
  the decoders on the motor template.

### 2026-10-31 · S3 sweep: 40 untried variants of the pooled decoder (dev, exploratory)

`scripts/s3_sweep.py`. 16 dev patients, protocol P3. Results in
`results/summary/s3_sweep_pre_dev.csv`. The generic code reproduces the confirmed pooled
decoder in 100 % of rows (own 81.7 %, pooled 84.0 % at k = 5).

| Idea | Best at k = 5 | vs pooled 84.0 % |
|---|---|---|
| Filter bank | fb4 (theta, mu, low beta, high beta): 87.0 % | +3.0 (median +1.4, p 0.083; 9 helped / 3 harmed) |
| More channels | 59 channels + mu/beta/8–30 bank: 86.5 % | +2.5 (p 0.18) |
| Single bands | beta 85.9 %; mu 79.2 % | +2.0; −4.7 (Holm 0.045) |
| Longer window (0.5–5.5 s) | no effect | factor means 85.3 vs 85.2 % |
| Donor weight 0.5–8 ×, LR C | −0.7 to +0.8; C = 0.1 hurts (−1.9) | |
| Fewer, closer donors (8 nearest, 8 best-fitting) | −0.5, −0.4 | |
| Stretching, SVM, 9-member ensemble, mixup | −0.9, +0.4, +0.6, ±0 | |
| No labels (other patients only) | 80.0 % | −3.9 (vs own −1.6) |

- **No fixed-calibration variant survives Holm** over the 31 comparisons.
- **Honest "pick the best" estimate** (leave-one-patient-out selection):
  - k = 5: 85.8 % vs 84.0 % (median +1.4, p 0.22). fb4 is picked in 14 of 16 folds.
  - k = 10: +0.5.
- **Factor means over the 12 structure combinations (k = 5):**
  - 14 / 28 / 59 channels: 84.7 / 85.3 / 85.7 %
  - window 0.5–4.0 / 0.5–5.5 s: 85.3 / 85.2 %
  - 8–30 Hz / bank: 85.0 / 85.5 %
- **Online adaptation is the only clear gain.**
  - Refitting after every pair with its cue label (co-adaptive calibration, a known technique)
    gives 88.6 % pooled and 88.5 % own: +4.6 and +4.5 over the fixed pooled decoder (Holm
    0.012 and 0.029; 12/1 and 9/1).
  - Pooling adds nothing once the decoder adapts. This is a different protocol: each trial's
    label is used after the trial is decoded.
  - Unsupervised running re-centring: 85.7 % (+1.7, p 0.062).

### 2026-10-31 · S3 cross-session transfer: the patient's own earlier session as a donor (dev, exploratory)

`scripts/s3_cross_session.py`. Results in `results/summary/s3_cross_session_dev.csv`.
- **Question:** in rehabilitation a patient returns for many sessions. Does the patient's own
  earlier recording help calibrate the next one?
- **Setup:** the target is a later session of a dev patient (post, or follow-up), calibrated
  on its first k (rest, MI) pairs (P3) and tested on the rest.
  - The donors are the patient's own earlier sessions ("previous": pre for post, pre + post for
    follow-up), the other dev patients' same-phase recordings ("others"), or both.
  - Same pooled code as confirmed; each donor group together weighs as much as the
    calibration.
  - "previous_zero" uses no labels from the later session at all: the earlier sessions'
    decoder, with the later session re-centred on its first 2k epochs.

| Target, k | own | others | previous | previous + others | previous_zero (no labels) |
|---|---|---|---|---|---|
| post, 5 (16 patients) | 80.7 % | 82.1 % (p 0.092) | 82.3 % (p 0.047, 7/1) | 82.8 % (p 0.042) | 81.3 % (p 0.41) |
| post, 10 | 84.0 % | 84.3 % | 84.8 % | 84.8 % | 81.4 % (−2.6, p 0.037) |
| follow-up, 5 (13 patients) | 77.9 % | 82.9 % (p 0.035) | 82.6 % (p 0.008, Holm 0.031; 9/1) | 84.2 % (p 0.021) | **84.5 %** (p 0.020) |
| follow-up, 10 | 82.3 % | 85.6 % (p 0.12) | 86.0 % (p 0.008, Holm 0.031; 8/0) | **87.6 %** (p 0.016, Holm 0.047) | 85.4 % (p 0.095) |

p-values are two-sided Wilcoxon vs own; Holm is over the four donor options within each row.
- **The patient's own earlier sessions help, most at follow-up,** months after the first
  recording. There the own 5-pair decoder is weakest (77.9 %), and earlier sessions add
  4.7 points (9 helped, 1 harmed). Other patients help about as much. Both together are best
  (84.2 % at k = 5, 87.6 % at k = 10).
- **A returning patient may not need labelled calibration at all.**
  - At follow-up, the earlier sessions' decoder, re-centred on 10 unlabelled epochs, scores
    84.5 %. The own decoder scores 77.9 % with 5 labelled pairs and 82.3 % with 10.
  - At post it matches the own 5-pair decoder (81.3 vs 80.7 %) but trails the 10-pair one.
- **Exploratory:** 16 and 13 dev patients, 16 comparisons. Confirmation needs fresh
  multi-session data; Liu2025 has pre, post and follow-up sessions for all 27 patients.

### 2026-10-31 · PRE-REGISTRATION: replication of patient pooling on a new cohort (S4, Liu2025)

Written and committed **before** any Liu2025 EEG is loaded (the download was still running).
- **Cohort:** S4 Liu2025 (Scientific Data 2025; figshare 10.6084/m9.figshare.27130299, re-hosted
  as Zenodo record 18987384; CC BY 4.0).
  - 27 stroke patients, 1–12 months after onset, 18 with right and 9 with left hemiplegia.
  - 60-channel Neuracle NeuSen W. Gait motor imagery vs idle, 10 + 10 trials per run in
    random order, about 4 runs per session.
  - Pre-treatment session only.
- **Independence:** same institution (Tianjin University) and amplifier family as S3.
  - Ages, sexes and paralysed sides suggest that some patients may be in both cohorts (e.g.
    S3 sub-02 and S4 sub-23: 64, female, right).
  - No S3 data are used here, so the replication is within S4. It is a new cohort and a new
    limb, not a new hospital.
- **Pipeline, frozen as confirmed on S3:**
  - 1–40 Hz; epochs 0.5–4.0 s after MI or idle onset (event codes 3 and 9); 128 Hz.
  - The 14-channel extended template with common average. Lesion moved to the left using
    participants.tsv `affected_side`.
  - 8–30 Hz OAS covariances.
  - own = `s3_transfer.own_proba`. pooled = `s3_augment.pooled_proba` with the other 26
    patients as donors, each re-centred on its own mean; all donors together weigh as much as
    the calibration.
- **Calibration (P3):** the first k trials per class in recording order calibrate (k = 5
  primary, 10 secondary). All other trials of the session test.
- **H-R1 (primary):** pooled > own at k = 5. One-sided Wilcoxon signed-rank over patients.
- **Also reported:** k = 10, the mean difference with a bootstrap CI, helped/harmed, and
  patients at or above chance.
- **Exclusions:** a patient is excluded only if the session cannot be read or a class has
  fewer than 20 trials.
- **H-R2 (secondary, added by amendment):** the best variant from the S3 sweep
  (`scripts/s3_sweep.py`, dev patients only) against pooled, at k = 5. It is fixed and
  committed before any S4 decoding. If added, Holm runs over H-R1 and H-R2.
- **Run once:** `scripts/s4_replication.py`. Whatever the outcome, it is reported.

### 2026-10-29 · S3 neuroscience analysis (PLAN §9; descriptive, all 24 patients, fixed methods)

`scripts/s3_neuro.py`. Results in `results/summary/s3_neuro_{erd,alignment,longitudinal}.csv`;
figure in `paper/figures/fig6_neuro.pdf`.
- **ERD, stroke vs healthy** (relative to rest; lesion normalised to the left):
  - Patients desynchronise more almost everywhere: mu −17 to −25 % vs −6 to −17 %
    (Holm-significant at 11 of 14 channels; not at C3, Cz or CP3); beta at all 14.
  - **Laterality reverses.** Laterality index (positive = stronger at C3): healthy +0.11, stroke
    −0.10 in mu; +0.13 vs −0.17 in beta (both p < 0.001).
  - The contralesional shift is what operator O2 imitates. The stronger ipsilesional ERD
    contradicts O1's "weaker" premise. Device and paradigm also differ between the cohorts.
- **Direction alignment** (re-centred tangent space, imagery-vs-rest class-mean direction):
  - Cosine with the other patients' pooled direction: median 0.87. With the healthy donors':
    0.78 (Wilcoxon p = 0.0002; patients closer for 22/24).
  - Within-patient split-half ceiling: 0.92.
  - This explains the pooling result: other patients give a direction almost as good as more
    of the patient's own data.
- **Longitudinal** (own TS + LR, blocked CV):
  - pre 87.8 % (n = 24), post 85.8 % (n = 24), follow-up 85.6 % (n = 19).
  - Changes: post − pre −2.0 (p = 0.18); follow − pre −2.1 (p = 0.038, Holm 0.11). Mu ERD at
    C3/C4 unchanged (p = 0.46 / 0.44).
  - Decodability is stable across the study period.

### 2026-10-29 · Phase 7: manuscript draft

`paper/` holds the LaTeX draft, the figures and the bibliography.
- **Manuscript:** `main.tex`, `results.tex`, `results_neuro.tex`.
- **Figures:** six, from `scripts/make_figures.py`, regenerated from the result CSVs.
- **Bibliography:** `refs.bib`, 32 entries, each retrieved from a registry (doi.org, arXiv
  API, OpenAlex, Europe PMC), with no hand-typed metadata.
- **Framing:** the plan's fallback (§12). The paper stands on the benchmark, SafeTransfer,
  the foundation-model arm and the confirmed pooling result, plus an honest analysis of why
  simulation did not help.
- `paper/lint.py` checks environments, braces, citations, references and figures (no LaTeX is
  installed here).

### 2026-10-25 · Phase 6: the live demo runs end to end

`demo/` (see `demo/README.md`).
- **What it does:** a locked-test patient's raw 64-channel EEG is replayed over LSL. The
  receiver calibrates on 5 rest + imagery pairs, then decodes every later trial with the own
  and the pooled decoder side by side. A FastAPI dashboard shows the stream, cue,
  calibration, gate, live accuracy and history.
- **Same method as evaluated:** the decoders live in `hemishift.models.pooling` and are
  unit-tested to give identical outputs to the experiment code.
- **Online pipeline = offline result.** Streaming all 8 locked-test patients through the
  online steps (causal filter, cut at each cue, harmonise, session) gives:
  - own 77.1 % (offline 77.1)
  - pooled 82.5 % (offline 81.4)

  Results are in `results/summary/demo_online_check.csv`.
- **Real LSL path, headless (`demo/selftest.py`):** patient 22 gives 78.6 / 84.3 %, and patient
  4 gives 90.0 / 97.1 %. Both are identical to the streaming check.
- **Latency:**
  - Processing is 12.6 ms per epoch on an idle machine (median) and 21–23 ms under full load
    (worst 54 ms).
  - Each decision is ready 4.0 s after its cue (the end of the imagery window) plus that
    processing time.
- **Server check:** page served; POST /start; 75 s of server-sent events captured: meta,
  calibration ×5, the gate, 27 decisions and 18 running scores (own 77.8 %, pooled 88.9 % at
  that point), plus about 17 EEG updates per second.

### 2026-10-25 · S3 step 5: EEG foundation model (CBraMod, frozen) on Task B (dev)

`scripts/s3_foundation.py`.
- **Setup:** CBraMod embeddings (0.5–3.5 s, three 1 s patches) and shrinkage LDA on the
  calibration epochs. Protocol P3. Results in `results/summary/s3_foundation_dev.csv`.

| Features | k = 5 | k = 10 |
|---|---|---|
| own TS + LR | 81.7 % | 84.7 % |
| fm-mean (200) | 68.0 % (13 harmed, Holm p 0.004) | 68.2 % (15 harmed, p 0.001) |
| fm-chan (14 × 200) | 58.2 % (15 harmed, p 0.002) | 62.5 % (15 harmed, p 0.001) |

- **C4 arm answered on both cohorts:** frozen foundation-model features trail the per-patient
  Riemannian decoder by 14–24 points here, and by about 5 points on S1. With 10–20
  calibration epochs, a linear probe on 200–2800 generic features cannot match 105
  covariance-based features built for exactly this signal.

### 2026-10-25 · S3 step 4: LesionSim rung 2, learned operator settings (dev): fails

`scripts/s3_theta_search.py`; Task B, k = 5, step 3's pooled decoder.
- **Search:** 27 population-mode settings: α ∈ {1, 0.6, 0.2} × β ∈ {0, 0.4, 0.8} ×
  spread ∈ {0, 0.5, 1}.
- **Honest choice:** 4 folds over patients (choose on 12, score 4).
- **Results** in `results/summary/s3_theta_search_dev.csv`:

| Donors | Mean (k = 5) |
|---|---|
| none (own) | 81.7 % |
| healthy, unmodified (identity; reproduces step 3 exactly) | 78.9 % |
| LesionSim, best setting in hindsight (α 0.6, β 0, spread 0.5) | 80.0 % |
| **LesionSim, nested choice** | **78.8 %** (vs identity: median −0.7, p 0.80; 7 helped / 8 harmed) |
| real other patients (step 3) | 84.0 % |

- **All 27 settings land within 78.9–80.0 %.**
- **The honest choice closes −2 % of the 5.1-point healthy-to-patient donor gap**; even the
  hindsight best closes only 21 %. The chosen setting differs in every fold, so there is no
  stable "stroke setting" to learn.
- **Verdict:** rung 2 fails. With rung 1 (hand-set, population and patient-matched) and the
  deep-pretraining tests, LesionSim is now negative in six forms. What makes real stroke
  donors useful is not captured by ERD attenuation, lateral shift or trial variability.
- Rung 3 (a learned generative translation) is not pursued for now. Real patient data is
  available and already helps, and no operator family moved the needle toward it.

### 2026-10-25 · S3 step 2: the SafeTransfer gate (H2), Tianjin Task B (dev)

`scripts/s3_safetransfer.py`. Results in `results/summary/s3_safetransfer_dev.csv`.
- **Candidates:** own (TS + LR), transfer (R3: healthy + LesionSim EEGNet, fine-tuned), and
  their ensemble.
- **Choice:** leave-one-pair-out CV on the calibration pairs only, then the gate (ties favour
  own; abstain unless the best beats chance, Bonferroni over 3). An abstaining patient falls
  back to own.

| | k = 5 | k = 10 |
|---|---|---|
| always own | 81.7 % | 84.7 % |
| always transfer | 76.4 % (**9/16 harmed** ≥ 2 pts) | 76.3 % (**11/16 harmed**) |
| always ensemble | 81.8 % (4 harmed) | 83.1 % (7 harmed) |
| **gate** | **80.6 % (2/16 harmed)** | **84.4 % (1/16 harmed)** |
| gate choices | own 9, ensemble 2, transfer 1, abstain 4 | own 10, ensemble 5, transfer 1 |

- **H2 is supported on dev, in both pre-registered parts:**
  1. The gate lowers the negative-transfer rate compared with always transferring: 9 → 2 of
     16 at k = 5 (exact McNemar one-sided p = 0.008; 7 vs 0 discordant) and 11 → 1 at k = 10
     (p = 0.001).
  2. It does not lower mean accuracy compared with always transferring; it raises it:
     +4.2 points at k = 5 (bootstrap CI 0.0 to +8.6, p = 0.033) and +8.1 at k = 10 (CI +1.7
     to +13.3, p = 0.001).
- **Context:**
  - The gate essentially learns, from calibration data alone, to avoid the harmful deep
    transfer. It does not beat simply always using own (80.6 vs 81.7, 84.4 vs 84.7).
  - Its value is safety when it is unknown whether a transfer model helps a given patient.
  - At k = 5 it abstained for 4 patients; the other 12 reached 83.5 %.
  - Calibration-CV and held-out accuracy correlate at ρ = 0.40 (k = 5) and 0.36 (k = 10),
    both n.s.

### 2026-10-25 · S1 step 5: patient pooling on Task A (dev only; the S1 locked test is used)

`scripts/s1_pooling.py`, the same pooled decoder as S3 step 3, with the other 33 dev patients
as donors. Protocol P3: the first k trials per class calibrate. Results in
`results/summary/s1_pooling_dev.csv`.

| k per class | own | pooled | Median diff, p | Helped / harmed |
|---|---|---|---|---|
| 5 (30 test trials) | 52.1 % | 54.4 % | +1.7, p 0.023 | 17 / 8 |
| 10 (20 test trials) | 57.1 % | 58.1 % | +2.5, p 0.29 | 17 / 10 |

- **Same pattern as S3:** pooling helps most when calibration is shortest, and the gain fades
  with more trials.
- On Task A the absolute level stays near chance at k = 5. Pooling rescues a few points, not a
  usable decoder.
- Dev-only, but consistent in direction with the S3 locked-test confirmation.

### 2026-10-24 · RESULT A (not evaluable): S1 screening on the locked test

`scripts/s1_screening.py --locked-test`, run once on the 15 S1 locked-test patients. Results in
`results/summary/s1_screening_locked_test.csv`.
- **The gate passed only 1 of 15 patients** (patient 11). On later trials that patient scored
  78.9 %; the 14 abstained patients averaged 53.9 %.
- **Primary hypothesis: NOT EVALUABLE.** The pre-registration required ≥ 2 patients per group.
- **Secondary:** calibration-CV vs test accuracy, Spearman ρ = 0.37, one-sided p = 0.088 (n.s.).
- **Always-ens16 vs always-base** in this protocol: 57.0 vs 55.6 %.
- **Reading:**
  - No false positive: the only patient passed decodes well.
  - The gate is conservative. It abstained on several patients who scored 65–74 % on later
    trials (25, 29, 37, 43), because their 20-trial calibration scores stayed below 15/20.
  - The locked-test patients are weaker overall (blocked-CV base 54.4 % vs 57.4 % on dev),
    so few could pass.
  - **Descriptive only**, dev + test pooled: 9/49 passed; passed patients score far above
    abstained ones.
  - The screen's value is specificity, not sensitivity; it is not confirmed. With ens16
    (p = 0.083), neither S1 locked-test hypothesis is confirmed.

### 2026-10-24 · RESULT B (confirmed): S3 patient pooling on the locked test

`scripts/s3_locked_test.py`, run once on the 8 S3 locked-test patients (hash 0a28a2e83d), with
the 16 dev patients as donors. Results in `results/summary/s3_locked_test.csv`.

| Calibration | own TS + LR | pooled with dev patients | Difference | Helped / harmed | One-sided Wilcoxon |
|---|---|---|---|---|---|
| **k = 5 (primary)** | 77.1 % | **81.4 %** | **+4.3 points** | 7 / 1 | **p = 0.035** |
| k = 10 (secondary) | 79.5 % | 81.6 % | +2.1 points | 5 / 1 | p = 0.14 |

**Corrected 2026-10-31 (floating-point ties, see the correction entry):** p = 0.047 at k = 5
and 0.17 at k = 10. Still confirmed at α = 0.05, with less margin than first logged.

- **The pre-registered hypothesis is CONFIRMED.** With 5 calibration pairs, pooling other
  patients' data in a re-centred Riemannian decoder beats the patient's own decoder on
  never-seen patients. At or above chance: 7/8 → 8/8.
- The gain matches dev (+2.3 there, +4.3 here) and shrinks with more calibration, where
  borrowing matters less.
- **The first locked-test-confirmed positive result of the project.** Calibration time can be
  halved without losing accuracy: pooled at k = 5 (81.4 %) ≈ own at k = 10 (79.5 %).

### 2026-10-24 · PRE-REGISTRATION: two locked-test confirmations (S1 screening, S3 pooling)

Written and committed **before** either analysis touches a locked-test patient.

**A. S1 screening (from step 3i)**
- **Patients:** the 15 S1 locked-test patients (hash 4999799d11).
  - They were used once before, for ens16 under blocked CV, and their blocked-CV accuracies
    were seen.
  - The procedure below was fixed on dev and has no free parameters, so that knowledge cannot
    steer it.
- **Procedure:** `scripts/s1_screening.py --locked-test`, otherwise unchanged.
  - The first 20 trials calibrate; the rest test.
  - base and ens16 are scored by leave-one-pair-out CV on the calibration trials.
  - The SafeTransfer gate (Bonferroni over 2, ≥ 15/20) passes the patient or abstains.
- **Primary hypothesis:** passed patients' test accuracy > abstained patients' (one-sided
  Mann–Whitney, p < 0.05). If either group has fewer than 2 patients, the test is reported as
  not evaluable.
- **Secondary:** Spearman ρ between the best calibration score and test accuracy (one-sided,
  ρ > 0).
- **Multiplicity:** Holm over the two S1 locked-test hypotheses (ens16 had p = 0.083).

**B. S3 patient pooling (from S3 step 3)**
- **Patients:** the 8 S3 locked-test patients (hash 0a28a2e83d). This is their first use.
- **Procedure:** `scripts/s3_locked_test.py`.
  - The first k (rest, MI) pairs calibrate; the rest test.
  - own TS + LR is compared with pooled TS + LR, using the 16 dev patients as donors.
  - Each domain is re-centred on its own mean, and the donors together weigh as much as the
    calibration. This is step 3's "+patients" code, unchanged.
- **Primary hypothesis:** pooled > own at k = 5 (one-sided Wilcoxon, p < 0.05; with 8 patients
  the smallest possible p is 0.004).
- **Secondary:** the same at k = 10.
- **Each analysis runs once.** Whatever the outcome, it is reported.

### 2026-10-21 · RESULT: S1 locked test, frozen ens16 pipeline (run once, as pre-registered)

`scripts/s1_locked_test.py`, 15 locked-test patients (none excluded), blocked 4-fold CV.
Results in `results/summary/s1_locked_test.csv`.

| | Base (MI 0.5–4 s, 8–30 Hz) | ens16 |
|---|---|---|
| Mean accuracy | 54.4 % | 58.0 % |
| At or above chance (67 % for 39 trials) | 2 / 15 | 4 / 15 |

- **ens16 − base:** mean +3.6 points; median 0.0 (bootstrap 95 % CI 0.0 to +7.7); helped 7,
  harmed 3, tied 5.
- **Pre-registered one-sided Wilcoxon: p = 0.083. The hypothesis is NOT confirmed at 0.05.**
- **Reading:** the effect replicates in direction and size (+3.6 on test vs +4.1 on dev), but
  15 patients, 5 of them tied, cannot confirm a 3–4 point gain.
  - It stays a reproducible, non-significant trend.
  - Confirming it would need more left/right stroke patients (UET175 once the password
    arrives).
  - No pipeline change will be evaluated on these 15 patients again.

### 2026-10-21 · PRE-REGISTRATION: S1 locked-test evaluation of the frozen ens16 pipeline

Written and committed **before** any locked-test patient is loaded.
- **Frozen pipeline (S1-ens16, from step 3e):**
  - 14 channels (EXTENDED template), 1–40 Hz, template common average, 128 Hz.
  - First 39 trials per recording (the rebound window cuts the last).
  - 16 candidates = 4 windows (MI 0.5–4, 0.5–2.5, 2–4 s; rebound 4–6 s) × (mu 8–13, beta
    13–30, 8–30 Hz, filter bank of all three).
  - Each candidate: OAS covariances, re-centring on the training mean, tangent space, LR
    (C = 1).
  - Decision = mean of the 16 P(right hand) > 0.5.
- **Comparator:** the base decoder (MI 0.5–4 s, 8–30 Hz, same model), on the same trials.
- **Evaluation:** blocked 4-fold CV per patient (as on dev), on the 15 locked-test patients
  (`configs/splits/liu2024_locked_test.json`, hash 4999799d11). No patient excluded unless the
  label-free quality rule in `liu2024_raw.EXCLUDED` already excludes them.
- **Hypothesis:** ens16 > base. One-sided Wilcoxon signed-rank on per-patient accuracy,
  p < 0.05. Also reported: mean difference with bootstrap 95 % CI, helped/harmed, patients at
  or above the chance threshold.
- **Run once.** Whatever the outcome, it is reported. No pipeline change after seeing the
  test result is evaluated on these patients again.

### 2026-10-21 · Bug fix: Euclidean alignment amplified the common-average null direction

- **Symptom:** in the first LesionSim run (step 4), every patient scored exactly 50.0 % with
  the +variability and +shift copies, i.e. the model predicted one class for everyone.
- **Cause:** the data are common-average referenced, so the channel sum carries no variance.
  `euclidean_alignment` (and the copy alignment in `s1_lesionsim.py`) used a plain inverse
  square root, which scaled that direction by about 1e6.
  - The LesionSim edits act per channel, so they break the zero sum. After alignment the
    copies reached |x| ≈ 2 × 10⁷ (99.99th percentile 6.9 × 10⁶); the originals reached 46.
  - Even on unedited data, the amplified float rounding added a noise direction to every
    EEGNet input. That affects the step 3c numbers below.
- **Fix:** `models.deep.inverse_sqrt` drops directions with variance below 1e-6 × the largest,
  and covariances are computed in float64. LesionSim and generic copies are re-referenced
  (common average) after editing, like real data. A unit test pins this down.
- **Re-runs:** step 3c (`s1_deep.py`) was re-run and its entry corrected below.
  - Held-out healthy accuracy rose from 63.0 to 67.7 %.
  - The significant negative transfer disappeared.
  - Step 4 (`s1_lesionsim.py`) is being re-run.

### 2026-10-21 · HEFMI-ICH (S5): why even healthy controls decode at chance

- **Channel order:** MOABB hard-codes it. The files store the names as MATLAB string objects
  (decoded from `__function_workspace__`), and they match MOABB's order exactly. Class names
  are "Left"/"Right"; labels alternate 0/1.
- **Data check:** each file holds 60 s of eyes-open and 60 s of eyes-closed rest.
  - Eyes-closed/eyes-open alpha peaks at O1/O2/Oz/PO3/PO4 in most files.
  - Slow eyes-open power (blinks) peaks at Fp1/Fp2/AF3/AF4.
  - So the labels are right, and the EEG itself is real.
- **Motor signal:** over the whole 37 s epoch, the healthy controls' mean 8–30 Hz power at
  C3/Cz/C4 varies by only ±0.03 log units (about 3 %). A typical MI ERD is 20–40 %. The
  left-minus-right laterality difference never exceeds 0.1 log units.
- **Verdict:** not a loading bug. The HEFMI EEG carries almost no motor-imagery ERD, even in
  healthy controls, so S5 stays a low-signal cohort. It is reported as such, not tuned.

### 2026-10-21 · S3 steps 1 and 3: transfer and LesionSim on Tianjin Task B with 5 or 10 calibration pairs (dev)

**Protocol (both steps):** P3 (chronological). The first k (rest, MI) pairs calibrate; the
remaining 70 or 60 epochs test. 16 dev patients, 14 channels.

**Step 1, deep transfer** (`scripts/s3_transfer.py`): EEGNet, fine-tuned on the calibration
epochs. Results in `results/summary/s3_transfer_dev.csv`.

| Decoder | k = 5 | k = 10 |
|---|---|---|
| own TS + LR | **81.7 %** (AUC 0.896) | **84.7 %** (0.912) |
| EEGNet from scratch | 58.5 % | 59.9 % |
| healthy-pretrained, zero-shot | 58.5 % | 60.2 % |
| healthy-pretrained + fine-tuned | 76.6 % (10 harmed) | 76.3 % (12 harmed) |
| + LesionSim copy | 76.3 % | 76.5 % |
| + generic copy (control) | 74.5 % | 72.5 % |
| + other dev patients (real stroke) | 77.2 % | 76.3 % |

- Every deep variant trails the own Riemannian decoder by 5–10 points (Holm p ≤ 0.06).
- Fine-tuning barely uses more calibration data (76.6 → 76.3 % from k = 5 to 10), so the
  deep route under-adapts.
- LesionSim = healthy-only, and both beat the generic control.

**Step 3, LesionSim as extra data for the own Riemannian decoder** (`scripts/s3_augment.py`).
- **Pooling:** donors are pooled with the calibration epochs. Each domain is re-centred on its
  own mean, the tangent space is taken at the identity, and all donors together weigh as much
  as the calibration. Results in `results/summary/s3_augment_dev.csv`.

| Decoder | k = 5 | vs own (median, p, Holm) | k = 10 | vs own |
|---|---|---|---|---|
| own | 81.7 % | — | 84.7 % | — |
| + healthy donors | 78.9 % | −2.9, p 0.16 | 84.3 % | −1.7, p 0.82 |
| + LesionSim donors (population) | 79.7 % | 0.0, p 0.33 | 84.2 % | +1.7, p 0.64 |
| + LesionSim donors (patient-matched) | 78.6 % | −2.1, p 0.10 | 83.5 % | 0.0, p 0.53 |
| **+ other dev patients** | **84.0 %** | **+1.4, p 0.004 (Holm 0.018); 11 helped, 0 harmed ≥ 2 pts** | 86.1 % | +0.8, p 0.16 |

- **Real stroke data from other patients improves the own decoder at k = 5:** +2.3 points
  mean, 11 of 16 helped, none harmed by ≥ 2 points. At k = 10 the gain shrinks (+1.4, n.s.).
- **LesionSim donors, population or patient-matched, do no better than unmodified healthy
  donors.** Neither beats own.
- **Why patient matching cannot work here:**
  - Patient-matched settings were mostly clipped at α = 1, β = 1 (median). The patients'
    relative ipsilesional mu ERD is mostly *not* weaker than healthy subjects' (healthy:
    −13 % ipsilateral, −9 % contralateral).
  - This chronic cohort does not show the "weaker lesioned-side ERD" that operator O1
    assumes. That matches the gap diagnosis, where mean power did not differ after
    correction and only variability did.
- **Verdict on C2 / H1 (dev):** LesionSim gives no measurable benefit in any of the four ways
  tested:
  1. deep pretraining on S1
  2. deep pretraining on S3
  3. population-mode donors for the Riemannian decoder
  4. patient-matched donors for the Riemannian decoder

  Real other-patient data does help, but only Riemannian pooling at k = 5 on Task B.

### 2026-10-21 · S1 step 3i: calibration-only screening finds who can use a left/right BCI (dev)

`scripts/s1_screening.py`.
- **Protocol P3 (chronological):** each patient calibrates on its first 20 trials and is
  tested on the next 19 (20 for patient 20).
- **Gate:** base and ens16 are scored by leave-one-pair-out CV on the 20 calibration trials.
  The SafeTransfer gate picks the better one, or abstains unless it beats chance (exact
  binomial, Bonferroni over the 2: ≥ 15/20).
- **Results** in `results/summary/s1_screening_dev.csv`:

| | Patients | Test accuracy (mean) |
|---|---|---|
| Gate passes ("usable signal") | 8 / 34 | **72.6 %** (60–89 %) |
| Gate abstains | 26 / 34 | 51.8 % (only 1 above chance) |

- **Passed vs abstained:** one-sided Mann–Whitney p < 0.001.
- **Calibration CV vs test accuracy:** Spearman ρ = 0.48 (p = 0.004).
- **Always-ens16 vs always-base** in this protocol: 59.7 vs 55.8 %. The step 3e gain holds
  under chronological calibration too.
- **Clinical scores do not predict decodability.** ens16 test accuracy vs NIHSS: ρ = −0.05;
  vs days since stroke: −0.13; vs rhythm peak: −0.22 (all n.s.). Only a short EEG calibration
  tells who will succeed.
- **Reading:**
  - This is the first clinically usable S1 result. Twenty calibration trials (about 3
    minutes) separate the quarter of patients whose left/right MI decodes well on later
    trials (72.6 %) from those who would get chance-level, misleading feedback (51.8 %).
  - That is the abstention arm of C3 working as designed.
  - Exploratory (dev); confirmation on the locked test is to be pre-registered.

### 2026-10-21 · S1 step 3h: EEG foundation model (CBraMod, frozen, linear probe), dev

`scripts/s1_foundation.py`.
- **Model:** CBraMod pretrained on TUEG (braindecode "braindecode/cbramod-pretrained";
  weights verified loaded), with a frozen encoder.
- **Input and classifier:** 200 Hz, 100 µV units, MI 0.5–4.5 s (4 patches). Shrinkage LDA on
  patch-averaged embeddings. Blocked 4-fold CV.

| Variant | Mean | ≥ 65 % | vs base 57.4 % (median, p) |
|---|---|---|---|
| fm-mean (200 features) | 52.1 % | 3 | −5.0, p 0.040 |
| fm-chan (14 × 200) | 52.7 % | 7 | −8.7, p 0.10 |
| fm + ERD (mean probability) | 52.9 % | 7 | −7.5, p 0.12 |

- **The foundation-model features decode left/right worse than covariances.** The C4 arm
  (foundation models as a baseline) is answered for S1: no gain.
- The fm + ERD row equals fm-chan because LDA's overconfident probabilities dominate the
  average (see step 3j).

### 2026-10-21 · S1 step 3j: phase coupling (lagged coherence), dev

`scripts/s1_coherence.py`.
- **Features:** lagged coherence over MI 0.5–4 s, averaged within mu and beta; the 91
  channel pairs per band go to shrinkage LDA. Blocked 4-fold CV, first 39 trials. Results in
  `results/summary/s1_coherence_dev.csv`.

| Variant | Mean | ≥ 65 % | vs base (median, p, Holm) |
|---|---|---|---|
| base | 57.4 % | 8 | — |
| coh-mu | 52.4 % | 2 | −3.8, p 0.046 |
| coh-beta | 46.7 % | 1 | −10.3, p < 0.001 |
| coh-both | 51.4 % | 2 | −5.1, p 0.004 |
| ens16 | 61.5 % | 13 | +2.6, p 0.011 |
| ens16 + coh (mean probability) | 52.4 % | 2 | −5.1, p 0.012 |

- **Lagged coherence carries no left/right information here.** It sits at or below chance;
  below-chance means with no signal come from drift across blocks.
- **Lesson for combining models:** averaging probabilities lets an overconfident member
  dominate. LDA on many features outputs near-0/1 probabilities, so ens16 + coh drops to the
  coherence model's accuracy. The CBraMod probe's "fm+erd" shows the same pattern.
  Combinations need calibrated scores (e.g. standardised logits) or rank averaging.

### 2026-10-21 · S1 step 4: LesionSim pretraining (dev, after the alignment fix)

`scripts/s1_lesionsim.py`.
- **Setup:** EEGNet pretrained on each source, then fine-tuned per development patient
  (blocked 4-fold CV). The LesionSim and generic copies are re-referenced after editing; the
  aligned copies' scale is now 7–10, like the originals (the generic copy is 34, because white
  noise fills directions the data lacks).
- **Results** in `results/summary/s1_lesionsim_dev.csv`:

| Pretraining data | Fine-tuned mean | vs own TS + LR (57.7 %) | Helped / harmed | vs healthy-only |
|---|---|---|---|---|
| healthy (PhysioNet, 109) | 56.5 % | median −2.5, p 0.60 | 13 / 18 | — |
| + variability copy (O4b) | 54.5 % | −2.5, p 0.12 | 10 / 20 | −2.5, p 0.068 |
| + contralesional-shift copy (O2) | 55.6 % | −2.5, p 0.32 | 15 / 18 | 0.0, p 0.26 |
| + weaker-ERD copy (O1) | 56.2 % | −2.5, p 0.52 | 13 / 19 | 0.0, p 0.53 |
| + LesionSim, all three | 55.4 % | −2.5, p 0.23 | 12 / 19 | −2.5, p 0.43 |
| + generic copy (control) | 54.4 % | −5.0, p 0.10 | 11 / 21 | −2.5, p 0.19 |
| **+ the other dev patients (real stroke)** | **56.0 %** | −2.5, p 0.44 | 14 / 19 | 0.0, p 0.73 |

- **LesionSim does not improve healthy-to-stroke transfer on S1 Task A.** No operator, alone
  or combined, beats healthy-only pretraining.
- **Even real stroke data does not help.** Pretraining on the other 33 patients' trials gives
  the same as healthy-only (56.0 vs 56.5 %). That bounds what any healthy-to-stroke
  simulation can achieve here: the pretraining data is not the bottleneck; each patient's own
  weak signal is.
- **Pre-registered H1 (R3 > R2):** no support on S1 at the dev stage.

### 2026-10-21 · S1 step 3g: individual frequency band and a long window (dev)

`scripts/s1_individual.py`; blocked 4-fold CV, first 39 trials; Holm over 4 variants.
- **Rhythm peak:** found above the 1/f background (C3/C4/Pz, whole recording) in 24/34
  patients. Patients without one keep 8–13 Hz.
- **Results** in `results/summary/s1_individual_dev.csv`:

| Variant | Mean | ≥ 65 % | vs base (median, p, Holm) | Helped / harmed |
|---|---|---|---|---|
| base | 57.4 % | 8 | — | — |
| iband: mu band at the patient's peak ± 2 Hz | 53.4 % | 2 | −5.1, p 0.016 (Holm 0.049) | 10 / 20 |
| long: one 0.5–6 s window, 8–30 Hz | 55.8 % | 7 | −2.6, p 0.44 | 14 / 18 |
| ens16 (step 3e, reproduced exactly) | 61.5 % | 13 | +2.6, p 0.011 (Holm 0.043) | 20 / 9 |
| ens16-i: ens16 with the individual band | 60.1 % | 11 | +2.6, p 0.10 | 19 / 12 |

- **No gain from individual bands.** A narrow band at the peak is significantly worse: the
  class information is not concentrated at the alpha peak.
- **No gain from one long window.** Separate windows in an ensemble beat one window spanning
  both MI and rebound (61.5 vs 55.8 %). The ensemble's strength is averaging over windows,
  not simply more data per covariance.

### 2026-10-21 · S1 step 3e: classical decoder upgrades (34 dev patients)

`scripts/s1_classical.py`.
- **Evaluation:** blocked 4-fold CV. Every filter, reference and model is fitted on training
  trials only. All variants use the first 39 trials (40 for patient 20), so the baseline equals
  step 3d's "fixed" decoder.
- **Statistics:** Holm across the 11 variants.
- **Results** in `results/summary/s1_classical_dev.csv`:

| Variant | Mean | ≥ 65 % | vs base (median, p, Holm) | Helped / harmed |
|---|---|---|---|---|
| base: TS + LR, MI 0.5–4 s, 8–30 Hz | 57.4 % | 8 | — | — |
| **ens16: mean probability of the 16 step-3d candidates** | **61.5 %** | **13** | **+2.6, p 0.011 (Holm 0.12)** | **20 / 9** |
| motor8: 8 motor-strip channels | 58.3 % | 9 | +2.6, p 0.25 | 19 / 13 |
| ens12: the 12 MI-window candidates only | 58.2 % | 10 | 0.0, p 0.69 | 15 / 13 |
| full29: all 29 channels | 57.2 % | 10 | +1.3, p 0.81 | 17 / 14 |
| potato: drop outlier training trials | 56.6 % | 6 | −2.6, p 0.39 | 10 / 18 |
| crops2s: five 2 s crops per trial | 56.5 % | 9 | −2.5, p 0.79 | 14 / 18 |
| mdm | 56.2 % | 7 | −2.6, p 0.33 | 11 / 19 |
| C = 0.1 | 55.9 % | 6 | −1.3, p 0.21 | 12 / 17 |
| lda (shrinkage) | 55.8 % | 10 | −2.6, p 0.14 | 8 / 20 |
| fgmdm | 55.8 % | 9 | −2.5, p 0.10 | 8 / 18 |
| csp (3 pairs) + LDA | 55.7 % | 6 | −2.6, p 0.10 | 8 / 20 |

- **Best candidate so far: the window/band ensemble with the rebound window** (+4.1 points
  mean). The gain needs the rebound window (4–6 s after MI onset, the break): ens12 without
  it gives only +0.8. So the post-imagery period adds information that the MI period lacks.
- Averaging all candidates (61.5 %) beats picking one per patient (step 3d nested, 59.2 %).
  With 30 training trials, averaging is more robust than selecting.
- **Caveats:**
  - Exploratory and not Holm-significant, so it needs the locked test set.
  - Trials alternate L/R, so in the break a patient may already prepare the next (opposite)
    hand. Rebound-window information can mix post-imagery beta rebound with that
    anticipation. Both are motor signals, but a real BCI's sequence is not predictable.
- Everything else (regularisation, LDA, MDM/FgMDM, CSP, channel sets, crops, outlier removal)
  gives no gain. The baseline decoder is not the bottleneck.

### 2026-10-21 · S1 step 3f: slow cortical potentials (MRCP) as a second source (dev)

`scripts/s1_slow.py`, first run.
- **Features:** 0.1–3 Hz, the 8 motor-strip channels, mean amplitude in 0.5 s bins over
  MI 0.5–4 s, shrinkage LDA. Blocked 4-fold CV, 40 trials.
- **Results** in `results/summary/s1_slow_dev.csv`:

| Variant | Mean | ≥ 65 % | vs ERD base (median, p) |
|---|---|---|---|
| ERD base (TS + LR) | 57.7 % | 8 | — |
| slow | 57.2 % | 10 | −1.3, p 0.98 |
| slow after EOG regression | 57.4 % | 10 | −1.3, p 0.94 |
| EOG channels only (control) | 55.4 % | 8 | +1.2, p 0.48 |
| hybrid (mean of ERD + slow) | 57.7 % | 8 | −2.5, p 0.88 |
| hybrid + EOG regression | 59.4 % | 9 | 0.0, p 0.59 |

- **Slow potentials decode as well as ERD, but in different patients.**
  - Spearman ρ between per-patient ERD and slow accuracy is 0.00 (0.13 after EOG regression).
  - Above chance: ERD {7, 15, 20, 22, 23, 28, 40, 45}; slow {8, 9, 12, 15, 18, 20, 22, 28,
    30, 44}.
  - Plain averaging dilutes whichever source is noise for a given patient.
- **Eye movements decode the class in 8 patients** (EOG-only control), so the slow features
  need confound controls.
- **Second run:** combiners that learn per patient which source to trust (inner blocked CV on
  the training trials), plus two controls:

| Variant | Mean | ≥ 65 % | vs ERD base (median, p, Holm over 9) |
|---|---|---|---|
| select (ERD, slow or mean; best inner CV) | 60.1 % | 12 | 0.0, p 0.59 |
| stack (LR on the two sources' out-of-fold logits) | 58.5 % | 15 | −2.5, p 0.89 |
| occipital slow features (visual control) | 53.6 % | 9 | −5.0, p 0.32 |
| C3 − C4 only (LRP) | 52.1 % | 6 | −8.8, p 0.033 (Holm 0.30) |

- **Verdict:** slow potentials add no reliable gain on S1.
  - Selection and stacking help some patients (stack puts 15 above chance) but hurt others, so
    the mean does not move.
  - The pure motor LRP is weak (52 %), and the visual control is weak too (54 %). The
    slow-feature information is spread out and patient-specific, not a clean lateralised
    motor potential.

### 2026-10-21 · S1 step 3d: per-patient window and band tuning, done honestly (nested CV, dev)

`scripts/s1_nested.py`.
- **Menu:** 16 candidates, from 4 windows (MI 0.5–4, 0.5–2.5, 2–4 s; rebound 4–6 s) ×
  (mu, beta, 8–30 Hz, or a filter bank of all three).
- **Selection:** each patient picks from its own training trials only. Inner loop: each of the
  3 training blocks is held out once. The pick is then scored on the outer test block
  (blocked 4-fold).
- **Model and data:** TS + LR, 14 channels. The rebound window loses each recording's last
  trial (it runs past the end of the file), so all candidates use the first 39 trials (40 for
  patient 20).
- **Results** in `results/summary/s1_nested_dev.csv`:

| Method | Mean | ≥ 65 % (chance threshold) |
|---|---|---|
| Fixed (MI 0.5–4 s, 8–30 Hz) | 57.4 % | 8 / 34 |
| Nested per-patient choice | 59.2 % | 10 / 34 |
| Oracle (best candidate in hindsight) | 72.7 % | 26 / 34 |

- **Nested vs fixed:** median +2.6 points, p = 0.40, helped 20 / harmed 13. **Not significant.**
- **Picks** (136 = 34 × 4 folds): rebound 47, MI 0.5–2.5 s 35, MI 0.5–4 s 29, MI 2–4 s 25. By
  band: beta 46, 8–30 Hz 36, mu 30, filter bank 24. The default was picked only 7 times, and
  picks change from fold to fold within a patient. **There is no stable "best setting" per
  patient.**
- **Methodological finding:** "best window/band per subject" looks like 72.7 % but is really
  59.2 %. The 13.5-point gap is selection luck on 40 trials. Per-subject tuning without nested
  CV inflates small-data MI results this much.

### 2026-10-21 · S1 step 3c: deep learning, CORRECTED after the alignment fix (dev)

`scripts/s1_deep.py`.
- **Pretraining:** EEGNet on 4,409 PhysioNet healthy trials (14 channels, per-subject
  Euclidean alignment, random 2 s crops, 60 epochs). Held-out healthy subjects reach 67.7 %
  (63.0 % before the fix), so pretraining works.
- **Patients:** blocked 4-fold CV per development patient (lesion normalised, alignment from
  training folds). Results in `results/summary/s1_deep_dev.csv`.

| Method | Mean | ≥ 65 % | vs own TS + LR | Helped / harmed |
|---|---|---|---|---|
| Own TS + LR (reference) | 57.7 % | 8 | — | — |
| EEGNet from scratch (30 trials) | 57.4 % | 6 | median 0.0, p 0.94 | 16 / 15 |
| Healthy-pretrained, zero-shot | 51.2 % | 1 | **median −6.2, p 0.017** | 10 / 21 |
| Healthy-pretrained + fine-tuned | 55.7 % | 6 | median 0.0, p 0.26 | 15 / 16 |

- **Deep learning does not beat the simple per-patient Riemannian decoder** on S1 Task A.
- **Retracted:** "healthy pretraining causes significant negative transfer". Before the fix,
  fine-tuning gave 53.4 % with 22/34 harmed (p 0.034). That was mostly the alignment bug: the
  amplified null direction fed noise into every input.
  - After the fix, fine-tuned transfer is no better and no worse than the own decoder
    (15 helped / 16 harmed, n.s.).
  - Zero-shot healthy transfer is still clearly worse than a per-patient decoder.
  - Via et al. 2026's 21/50 harmed remains the literature reference. Our pipeline shows
    transfer that *does not help* rather than transfer that hurts.

### 2026-10-21 · S1 step 3b: within-session drift correction, the first significant gain (dev, exploratory)

`scripts/s1_drift.py`: blocked 4-fold CV, own decoder, 14 channels. In the "block" variant,
each block of 10 consecutive trials is re-centred on its own Riemannian mean. This is
label-free; each block holds 5 left + 5 right trials. Results in
`results/summary/s1_drift_dev.csv`.

| Window | Global re-centring | Block re-centring | Median gain | p | Helped / harmed |
|---|---|---|---|---|---|
| MI 0.5–4 s | 57.7 % (8 > chance, 2 < chance) | 59.0 % (12 >, 1 <) | +2.5 | 0.106 | 18 / 12 |
| **Rebound 4–6 s** | 58.6 % (10 >, 1 <) | **61.1 % (11 >, 0 <)** | **+5.1** | **0.0205** | **21 / 8** |

- **This is the first significant improvement on S1 Task A.** Signals drift during the session,
  and re-centring per block removes the drift.
- **Exploratory:** it is the best of ~12 dev comparisons today, so it is a candidate, not a
  claim.
- **Caveat:** block re-centring uses the whole test block (unlabelled, including later trials).
  A real-time BCI needs the causal version, a running mean of past trials only.
- **To do before any claim:**
  1. Implement and test causal adaptive re-centring.
  2. Freeze the candidate pipeline (rebound window + causal re-centring).
  3. Evaluate it once on the 15 locked test patients.
- **Causal check** (`s1_drift.py --causal`): calibrate on the first 10 + 10, then decode the last
  20 in order, re-centring each trial on the 10 trials before it. Results in
  `results/summary/s1_drift_causal_dev.csv`.

  | Window | Static | Causal adaptive | Median change | p | Helped / harmed |
  |---|---|---|---|---|---|
  | MI 0.5–4 s | 57.1 % | 55.1 % | −5.0 | 0.11 | 9 / 18 |
  | Rebound 4–6 s | 55.3 % | 56.7 % | +5.3 | 0.11 | 18 / 10 |

  The gain only partly survives causally: the rebound trend is positive but not significant, and
  the MI window gets worse. A likely cause is that training trials were normalised by the
  calibration mean while test trials used a sliding mean. The consistent variant (both
  normalised by recent trials) comes next.
- **Consistent causal variant** (calibration trials also normalised by their preceding trials):
  MI 55.1 % (median −5.0, p 0.11); rebound 55.7 % (median 0.0, p 0.57).
- **Verdict:** **drift correction gives no gain under real-time (causal) conditions.** The block
  gain came from re-centring with *future* trials of the same block, a transductive look-ahead
  that a BCI cannot use. It is dropped as an improvement, and noted as a pitfall: offline
  re-centring can inflate results.

### 2026-10-21 · S1 step 3a: information outside the classic window? (34 dev patients)

`scripts/s1_information.py`: own decoder, 14 channels, blocked 4-fold CV (all 40 trials scored,
30 trained). Results in `results/summary/s1_information_dev.csv`.

| Variant | Mean | ≥ 65 % (chance threshold, 40 trials) | vs MI (Holm) |
|---|---|---|---|
| MI 0.5–4 s | 57.7 % | 8 | — |
| Rebound 4–6 s (break) | 58.6 % | 10 | p 0.84 (1.0) |
| Healthy side + midline, MI | 55.7 % | 7 | p 0.46 (1.0) |
| MI + rebound 0.5–6 s | 55.8 % | 7 | p 0.36 (1.0) |
| Healthy side, MI + rebound | 55.8 % | 7 | p 0.25 (0.99) |

- **No source beats the classic window on average.** Training on 30 trials instead of 20 lifts
  the mean to ~58 %, so data quantity matters.
- **Two patients (1, 16) are significantly below chance** (≤ 30 %). That points to within-session
  non-stationarity (drift flips the decision rule) and motivates adaptive re-centring.
- **The best variant differs by patient:** 18/34 pass 65 % in at least one, but picking after
  scoring is optimistic. Per-patient selection must use training folds only.

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
