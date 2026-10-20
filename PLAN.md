# HemiShift — Project Plan

*Working name. Teaching a stroke-rehab BCI to read damaged brains, using only public data.*

Written 2026-10-20. Evidence for every claim about prior work is in [LITERATURE.md](LITERATURE.md).

---

## 0. In plain words

**The problem.** After a stroke, a patient can do rehab by *imagining* moving the weak hand. An
EEG cap picks up the attempt, and a robotic glove moves the hand. Done hundreds of times, this
helps the brain rewire. The weak link is the AI that reads the EEG. It's usually built from
healthy brains, and on stroke brains it guesses badly. In one recent study, borrowing knowledge
from healthy people made **21 of 50 patients worse**.

**What we do:**
1. **Merge 4 public stroke datasets** (about 130 patients, 4 hospitals, 3 countries, 4 EEG
   devices) into one clean benchmark. Nobody has done this. Every paper so far tests on one
   hospital only.
2. **Make healthy brains look like stroke brains.** We take healthy people's EEG and apply the
   changes a stroke causes: a weaker signal on the damaged side, the healthy side taking over,
   more scattered timing. That gives hundreds of extra "stroke-like" training examples. We found
   no paper doing this. The best version tunes the edits to each new patient from their first
   ~10 practice tries, so the extra examples match *that* patient's damage.
3. **Add a safety check.** With a few practice tries from a new patient, the system decides
   whether the borrowed knowledge helps *this* patient. If nothing works, it says so instead of
   giving wrong feedback.
4. **Run the hardest test:** train on some hospitals and test on a hospital the AI has never seen.
5. **Demo:** a real patient's recorded brain signal plays back, and a virtual hand closes when
   the AI detects the attempt.

**A realistic margin.** Even healthy people average only about 71 % on this kind of task, so a
patient's own decoder tops out around 75–80 %. The real gap is **transfer**: a decoder built from
other people drops a new patient to about 50 %, a coin flip. Our margin is closing part of that
50 → 70 % gap with few practice tries, while harming fewer patients. A consistent gain of a few
points across patients and hospitals is publishable. A 20-point jump would be suspicious, so check
for leakage before celebrating (§7). The evidence on why accuracy is low is in LITERATURE.md §5.

---

## 1. What the novelty check changed

| Original idea | Status (2026-10-20) | Role now |
|---|---|---|
| EEG foundation models on stroke | **Taken.** Nguyen et al., arXiv 2609.00282 (Aug 2026), UET175 only | A baseline arm (C4), not the headline |
| Healthy-to-stroke transfer | **Taken.** Nagarajan JNE 2024; Via et al. bioRxiv 2026 | The problem we improve on |
| Multi-site stroke benchmark | **Open** | **C1** |
| Physiology-based "lesion simulation" augmentation | **Open.** Closest are JBHI 2025 (cross-side generation) and WalkBCI 2026 (from the patient's own rest EEG) | **C2, the core method** |
| Per-patient negative-transfer gate | **Open.** The problem was measured (42 % of patients), not solved | **C3** |
| Train on some hospitals, test on an unseen one | **Open** | Headline protocol **P4** |

## 2. Contributions (what the paper claims)

- **C1 — StrokeMI-Bench.** A harmonised, leakage-safe benchmark built from 4 public stroke MI
  cohorts plus healthy cohorts. It has fixed patient-level and site-level splits and is fully
  reproducible from raw downloads.
- **C2 — LesionSim.** A physiology-informed augmentation that turns healthy MI EEG into
  stroke-like MI EEG. Its parameters are calibrated on training-fold patients only.
- **C3 — SafeTransfer gate.** Using a few calibration trials, choose patient-specific, transferred
  or ensemble decoding, or abstain. Target: fewer patients harmed by transfer, without lowering the
  mean.
- **C4 — Honest comparison.** Classical, Riemannian, specialist deep and foundation models (LoRA),
  all under the same leave-patient-out and leave-site-out protocols.

## 3. Data

### Stroke cohorts

| ID | Dataset | Patients | Device · channels · Hz | Task | Watch out for |
|---|---|---|---|---|---|
| S1 | Liu 2024 (Sci Data 11:131), Beijing | 50 acute (1–30 d), mild (NIHSS ≈ 4) | ZhenTec NT1 semi-dry · 29–30 + 2 EOG · 500 | Left vs right hand | Paper inconsistent on channels and trials. Data **CC BY 4.0**, 2.3 GB |
| S2 | UET175 (Front Neurosci 2025), Ho Chi Minh City | 30 | Emotiv Flex saline · 22 · 128 | Left/right arm and leg | **Trials pre-selected for ≥30 % ERD**, so optimistic; report separately. **Archive is password-protected**: request it from the authors |
| S3 | Wang 2026 (Sci Data 13:1109), Tianjin | 24 chronic (3–14 mo), 3 time points | Neuracle · 64 (40 kept) · 250 | Rest / affected hand / sixth finger | Data **CC BY 4.0** (only the article is NC-ND), 8.7 GB (5.2 raw + 3.5 processed). No MOABB loader: MOABB's `Wang2026` is a different, healthy study |
| S4 | Liu 2025 (Sci Data), lower limb | 27, 3 time points | to check | Lower-limb MI | Original figshare is **CC BY**. **21.8 GB, so download deferred** until a phase needs it |
| S5 | HEFMI-ICH (Sci Data 2025), intracerebral haemorrhage | **20 ICH + 17 healthy controls**, 1–6 sessions each | g.HIamp · 32 · 256 (EEG; fNIRS skipped) | Left vs right hand grasp; 2 s cue, 10 s MI, **15 s rest** | **Same device for patients and controls**, so disease and device effects separate. CC BY 4.0. MOABB `HefmiIch2025` |

### Healthy cohorts

- **H1 — PhysioNet EEGMMI** (109). Used by both 2026 competitors, so results line up directly.
- **H2 — OpenBMI / Lee2019** (54). Nagarajan's source; very large download, so it's optional.
- **H3 — BCI IV-2a** (9). A sanity check.
- **H4 — Cho2017** (52). Optional.

Loaders, confirmed 2026-10-20 against MOABB 1.7.2:
- MOABB: S1 (`Liu2024`), S4 (`Liu2025`), S5 (`HefmiIch2025`), H1, H3.
- Our figshare downloader: S3.
- GitHub (password pending): S2.

Even without S2, S1 + S3 + S5 give three stroke cohorts from three hospitals.

### Harmonised tasks

- **Task A, "which hand":** affected-hand MI vs unaffected-hand MI. This is left/right
  relabelled by hemiplegia side. Available in S1, S2 (arms) and the healthy sets (with a virtual
  affected side).
- **Task B, "is the patient trying":** affected-hand MI vs rest. This is the decision a rehab
  glove actually needs, and it's likely easier when the damaged side's signal is weak: it can use
  whichever hemisphere carries the signal, while left-vs-right needs both. Make it the primary
  task wherever rest windows exist. Available in S3 and H1. S1, S2 and S5 have it only if rest windows of
  the same length exist; check in week 1.

### Harmonisation recipe (decided in week 2, after checking UET175's channel names)

- **Channels:** a motor-strip template of roughly F3 Fz F4 FC3 FCz FC4 C3 Cz C4 CP3 CP4 P3 Pz P4.
  CPz is the reference in S1 and S3, so leave it out. Re-reference to the common average of the
  template.
- **Resampling and filtering:** resample to **128 Hz** (UET175's rate) and store 1–40 Hz.
  Models choose their own band.
- **Epochs:** 0.5–4.0 s after MI onset. Rest windows are the same length.
- **Lesion-side mirroring:** swap C3↔C4, FC3↔FC4 and so on for left-hemiplegia patients, so the
  lesion is always on the same side. Run an ablation with mirroring on and off.
- **Artifacts:** amplitude-threshold trial rejection, plus EOG regression where EOG exists (S1).
  Log every rejection.

## 4. LesionSim (C2), the core method

### 4.0 How we make it work (five rules)

1. **Diagnose before building.** Phase 1a measures how real stroke EEG differs from healthy EEG in
   *our* cohorts, per hemisphere and per patient: ERD size, laterality, timing spread,
   trial-to-trial variance and noise. Operators are prioritised by the measured gap, not by
   textbook. Prior evidence (Via 2026, S1 vs EEGMMI): mean band power does *not* differ after FDR,
   but stroke EEG shows **more variability and more diffuse time–frequency modulation**. So for S1,
   O4 (timing/variance) and O2 (diffuse topography) come first; O1 and O3 matter more for the
   chronic S3 cohort. Severity decides *where* the signal is: in more impaired patients it moves
   to the healthy hemisphere (Mansour 2022, 136 patients). That makes O2 and patient-matched mode
   essential, not optional.
2. **Patient-matched mode (the strongest version).** In few-shot use (P3/P4), estimate *this*
   patient's θ (ERD weakness, laterality, spread) from their k calibration trials, then build a
   synthetic training set from healthy donors that matches that patient. It's leakage-safe because
   it uses calibration trials only, never test trials. Population mode (θ from training-fold
   patients) is the fallback and also covers k = 0.
3. **Domain randomisation.** Randomise everything we can't measure (device noise, electrode
   dropout, impedance drift, timing) over wide ranges. This is the sim-to-real trick from robotics,
   and it stops the decoder overfitting to one hospital.
4. **Fail fast.** A week-3 pilot (§11) tests a minimal LesionSim on S1 + H1 before the full build.
5. **A ladder, not one bet.** If a rung fails, climb to the next one instead of stopping:
   - **Rung 1:** hand-written operators O1–O5, calibrated in population and patient-matched modes.
   - **Rung 2, learned θ:** tune the operator settings automatically to maximise inner-validation
     accuracy on training patients (augmentation search).
   - **Rung 3, learned translation:** a conditional generative model (cGAN, diffusion or flow
     matching, reusing DiffIDS code) maps healthy → stroke, with the measured statistics as
     constraints.

   All three are "healthy-to-stroke simulation", so C2's claim survives whichever rung works.

**Where to judge it.** Gains should be largest where data is scarce: few-shot k = 5–10, more
impaired patients, and unseen hospitals. Report per-patient results and the negative-transfer
rate, not only means. Fewer patients harmed is a win even if the mean barely moves.

**Combine, then ablate.** The final system stacks LesionSim with proven tricks (EA, fine-tuning,
ensembling) to beat the best existing recipe. Ablations then show how much of the margin
LesionSim itself contributes, and the paper must report that honestly.

### 4.1 Operators

These operators apply to a healthy MI trial. The "ipsilesional" channels are on the hemisphere
opposite the virtual affected hand.

| Op | What it mimics | How |
|---|---|---|
| O1 | Weaker movement signal on the damaged side | Per band (mu 8–13, beta 13–30): Hilbert amplitude A(t) → A₀ + α·(A(t) − A₀), with α ∈ [0, 1] and A₀ the pre-cue level. Phase is kept. |
| O2 | Healthy side taking over, and blurred topography | Transplant the relative ERD time course to the homologous contralesional channels with weight β, then apply spatial blur σ |
| O3 | EEG slowing in the damaged hemisphere | Scale theta (and delta) up and alpha down on ipsilesional channels by γ |
| O4 | More scattered timing | Random time-warping of the ERD envelope (onset jitter, stretching) and per-trial α variance |
| O5 | Device differences | Noise at the target device's SNR, channel dropout, resampling to device rate and back |

**Calibration (leakage-safe).** Fit θ = (α distribution, β, σ, γ, jitter) by matching summary
statistics of real stroke patients: **training-fold patients only** in population mode, or the
**target patient's calibration trials only** in patient-matched mode. The statistics are:
- per-hemisphere ERD size
- laterality index
- ERD latency
- background asymmetry (BSI / delta-alpha ratio)
- trial-to-trial variance

Start with simple moment matching; try MMD minimisation only if needed. If calibration says an
operator isn't needed (for example γ ≈ 1 because S1 strokes are mild), it switches itself off.
Via et al. found no band-power difference surviving FDR between EEGMMI and S1, so expect small
O3 effects there.

**Validation:**
- **Realism:** a discriminator's AUC for real stroke vs simulated, compared with real stroke vs raw
  healthy.
- **Utility:** downstream accuracy.
- Report both, but decide on utility only, because realism does not predict usefulness (lesson
  from DiffIDS).

**Ablations that make the paper:**
1. Each operator on and off.
2. **Calibrated θ vs random θ vs generic augmentation with the same distortion budget.** This
   proves the physiology matters, not just "more noise".
3. Number of healthy donors (10 / 50 / 109 / 200+).
4. Ratio of synthetic to real data.

**Generative models play two roles:** a comparison baseline (generic generation with no
physiology), and Rung 3 (physiology-constrained translation) if Rungs 1–2 fail.

## 5. Models and training regimes

- **Classical (CPU):** CSP+LDA and FBCSP+SVM (to reproduce S1's paper); Riemannian TS+LR and MDM
  with re-centering/RPA.
- **Deep (GPU):** EEGNet, ShallowConvNet; EEG-Conformer optional. All through braindecode.
- **Foundation models (C4):** LaBraM-base and CBraMod, plus REVE-base if the weights and memory
  allow. Each as head-only, LoRA, and full fine-tuning where it fits.
- **Alignment:** Euclidean Alignment on and off for every deep model.

| Regime | Trained on | Role |
|---|---|---|
| R0 | Target patient only | Patient-specific baseline |
| R1 | Other stroke patients, same site | Stroke-pooled |
| R2 | Healthy pretrain → stroke fine-tune | **Nagarajan-style recipe, the main competitor** |
| R3 | Healthy + LesionSim → stroke fine-tune | **Ours** |
| R4 | All other sites + healthy + LesionSim | For leave-site-out |

## 6. Evaluation protocols

- **P1 Within-patient.** Liu 2024's protocol (60/40 + 10-fold), used **only** to compare with
  their 55.57 / 57.57 / 72.21 %. It isn't deployment-realistic, so say so.
- **P2 Leave-patient-out** within a cohort, with zero calibration.
- **P3 Few-shot calibration:** k ∈ {0, 2, 5, 10, 20} trials per class from the new patient, 5
  seeds. The calibration trials are the patient's **first k trials per class in recording
  order**, as in a real session. Random draws sit next to test trials in time and flatter the
  decoder, so they are used only as a sensitivity check (`splits.calibration_split`).
- **P4 Leave-site-out (headline):** train on other hospitals and test on an unseen one,
  k ∈ {0, 5, 10}.
- **Floor:** healthy-only zero-shot. Nguyen 2026 got 0.464 on UET175.

**Metrics:**
- balanced accuracy (primary), Cohen's κ, AUC for Task B
- per-patient distribution
- **negative-transfer rate:** the share of patients where the transferred model is ≥ 2 points
  below R0/R1 at the same k (pre-registered margin)
- worst-10 % patient mean
- gate calibration (ECE)
- share of patients significantly above chance, using exact binomial thresholds for each
  patient's test-trial count (16 test trials need ≥ 75 %; 40 need ≥ 65 %). Single-patient
  differences sit within noise (± 7–12 points), so claims are made across patients only.

**Statistics:**
- per-patient paired Wilcoxon signed-rank tests, with Holm correction
- bootstrap 95 % CIs over patients
- effect sizes
- 5 seeds

**Pre-registered hypotheses.** Write them into ACHIEVEMENTS.md *before* running:
- **H1:** R3 > R2 on P4, Task A, k = 10, in ≥ 2 of the stroke cohorts (Holm-corrected p < 0.05).
- **H2:** SafeTransfer lowers the negative-transfer rate compared with always transferring,
  without lowering mean balanced accuracy.

## 7. Leakage rules (non-negotiable)

1. Split by **patient** (and by **site** for P4) before anything learns from data.
2. Everything that is fitted uses training patients only: normalisation, CSP filters, Riemannian
   references, **LesionSim θ**, hyperparameters.
3. For a test patient, the EA/RPA reference comes from that patient's **calibration trials
   only**. Declare this in the paper.
4. Calibration trials never appear in the test set.
5. Early stopping and tuning use inner-validation patients. Test patients are scored once per
   final config.
6. Every number in ACHIEVEMENTS.md records its config and split hash.
7. **Sanity controls:** shuffled labels must give ≈ 50 %. Any cross-patient stroke result above
   90 % triggers a leakage audit.

## 8. SafeTransfer gate (C3)

- **Candidates:** R0 (patient-specific, trained on the k calibration trials), R3 fine-tuned on
  the same k, and their ensemble.
- **Choice:** estimate each candidate's accuracy with leave-one-out on the k calibration trials
  and pick the best. If every candidate's CI includes chance, **abstain** ("collect more
  calibration first"). Wrong feedback can teach the patient the wrong thing.
- **Who abstain is for:** about 1 in 8 patients may produce no usable signal at all (6 of 46 at
  chance in Ang 2011). No decoder can fix that, so abstaining is the honest output for them.
- **k:** use k ≥ 5 per class; at k = 2 the estimates are too noisy (report that too).
- **Report:**
  - negative-transfer rate (within our pipeline; cite Via's 21/50 as context, since their
    decoder differs)
  - correlation between predicted and true held-out accuracy
  - abstention rate
- **Exploratory:** do NIHSS, MBI, days since stroke or lesion side predict who benefits? S1 has
  this metadata.

## 9. Neuroscience analysis (makes it more than an ML paper)

- Attribution topomaps (Integrated Gradients; Haufe patterns for linear models). Do decoders rely
  on the *healthy* hemisphere in more impaired patients?
- Per-patient ERD laterality vs decodability vs NIHSS/MBI.
- S3 longitudinal: does decodability change from pre to post to follow-up rehab?
- Real stroke vs LesionSim topographies side by side.

## 10. Demo, "HemiShift Rehab"

- **Live replay:** a held-out patient's EEG streams over LSL.
- **Decoder:** 2 s window, 250 ms hop, ONNX Runtime on the laptop CPU. Target under 100 ms per
  window.
- **Virtual hand:** closes after N consecutive positive windows (debounced).
- **Calibration wizard:** "10 imagined grasps", then SafeTransfer picks a model or abstains.
- **Clinician panel:** live ERD topomap, confidence, and **"old decoder vs ours" side by side**.
  This is the strongest moment in the demo.
- **Optional:** an ESP32 servo "finger" that physically closes.

## 11. Phases (about 18 weeks at 15–20 h/week)

DiffIDS's 16-week plan ends around mid-November. Weeks 1–4 here are light enough to overlap
with it.

| Week | Phase | Deliverable | Gate |
|---|---|---|---|
| 1 | **0 · Setup + data access** | Env with CUDA torch; all cohorts downloaded; inventory (subjects, channels, Hz, trials/class, rest windows, **licence from the original repo**); S1 inconsistencies resolved; 6 must-read papers read; [P] authors filled in LITERATURE.md | **G0:** ≥ 3 stroke cohorts usable. UET175's archive needs a password from its authors; if it hasn't arrived, download Liu 2025 (21.8 GB) as the third cohort. Otherwise scope to S1+S3 with a 2-site P4 |
| 2 | **1a · Fast path: S1 + H1 + gap diagnosis** | S1 and EEGMMI harmonised first; stroke-vs-healthy gap measured (§4.0 rule 1); FBCSP/EEGNet leave-patient-out baselines on S1; ACHIEVEMENTS.md started with H1/H2 written *before* any run | **G1:** reproduce S1's CSP+LDA / FBCSP+SVM within ±3 points |
| 3 | **Pilot (fail fast)** | R2 vs R2 + minimal LesionSim (the operators the diagnosis ranks highest, in population and patient-matched modes) on S1, k = 10, 5 seeds | **GP:** more patients helped than harmed and a mean gain of ≥ 1–2 points → full speed. Nothing → Phase 3 starts at Rung 2/3 |
| 4–5 | **1b · StrokeMI-Bench (C1)** | Add S2–S5; `build_benchmark.py` goes raw → harmonised arrays + metadata; fixed split files; data card with QC plots per cohort | Rebuilds bit-identical from scratch |
| 5–6 | **2 · Baselines** | R0–R2 × classical/Riemannian/EEGNet/Shallow × P1–P4 on all cohorts | **G1b:** best P4 baseline < 85 % (headroom exists) |
| 6–9 | **3 · LesionSim ladder (C2)** | Rungs 1→3 as needed; unit tests (identity at α=1, β=0, γ=1); calibration; realism + utility; ablations | **G2:** H1 holds on P2 or P4 in ≥ 2 cohorts. Otherwise use the fallback (§12) |
| 9–11 | **4 · Foundation models (C4)** | LaBraM/CBraMod (± REVE) × head/LoRA/full × ± LesionSim; one config reproduces Nguyen 2026's UET175 setup | — |
| 11–12 | **5 · Gate + analysis (C3)** | SafeTransfer results; attribution maps; clinical correlations | — |
| 12–14 | **6 · Demo** | ONNX + LSL replay + UI + latency numbers + 2-minute video | Demo runs end to end |
| 14 | **Patent (optional)** | The repo went public on 2026-10-20, which counts as public disclosure. Indian and European novelty is likely lost. The US allows 1 year after an inventor's own disclosure, so a US provisional, if wanted, must be filed well before **2027-10-20**. Confirm with the VIT IPR cell (not legal advice) | Decided |
| 14–18 | **7 · Paper** | Draft → supervisor review → re-run LITERATURE §4 searches → submit | Submitted |

## 12. Risks and fallbacks

| Risk | Fallback |
|---|---|
| LesionSim doesn't beat R2 (GP or G2 fails) | Climb the ladder (§4.0): Rung 2 (learned θ), then Rung 3 (learned translation). Also try LesionSim for pretraining only. If every rung fails, the paper stands on C1 + C3 + C4 plus an honest analysis of *why* simulation didn't help. **Change the method, not the paper type** (same rule as DiffIDS). |
| S1 strokes are mild, so the stroke-healthy gap is small | S3 (chronic) and S5 may show bigger effects. State the severity range as a limitation. |
| UET175 is optimistic (ERD-selected) | Report separately and never pool it silently. This is also a finding worth stating. |
| Licences | Checked 2026-10-20: S1, S3 and S4 data are CC BY, so processed derivatives can be released with attribution. UET175's licence must be read from its repo. If any cohort turns out NC/ND, release code plus the build script only. |
| FMs don't fit 4 GB | Kaggle T4 (16 GB). Drop REVE before dropping LaBraM/CBraMod. GTX 1650 has **no bf16**, so use fp16 autocast with GradScaler. |
| Results look too good | Run the §7 sanity controls first. |

**Limitations to state in the paper:**
- offline, retrospective data only, with no closed-loop online trial
- severity range limited to the cohorts available
- devices and sites confounded within cohorts

## 13. Paper and venues

**Draft title:** *Simulating the Lesioned Brain: Physiology-Informed Augmentation for
Cross-Hospital Motor-Imagery Decoding after Stroke.*

**Outline:**
1. Intro (rehab BCIs, calibration burden, negative transfer)
2. Related work
3. StrokeMI-Bench
4. LesionSim
5. SafeTransfer
6. Experiments (P1–P4)
7. Results
8. Neuro analysis
9. Limitations
10. Conclusion

**Venues:**
- **Q1:** Journal of Neural Engineering, IEEE TNSRE, Journal of NeuroEngineering and
  Rehabilitation, IEEE JBHI
- **Fallback:** Biomedical Signal Processing and Control, Frontiers in Neuroscience / Human
  Neuroscience, IEEE Access

**Odds, same as DiffIDS:** Q1 on a first attempt is uncertain (roughly a quarter to a third).
Publication somewhere is very likely if the work is finished and submitted down the tiers.
**Biggest lever:** a supervisor plus a neurologist or physiatrist co-author.

## 14. Repo layout and environment

```
B_i_g/
  PLAN.md  LITERATURE.md  ACHIEVEMENTS.md (results log, from week 2)
  requirements.txt
  configs/      datasets.yaml · splits/ · experiments/*.yaml
  src/hemishift/
    data/       loaders.py · harmonise.py · mirror.py · splits.py · datacard.py
    augment/    lesionsim.py · classic.py · calibrate.py
    models/     classical.py · deep.py · fm.py (LoRA via peft)
    train/      run.py           (one config-driven entry point)
    eval/       protocols.py · metrics.py · stats.py · gate.py
    demo/       replay_lsl.py · server.py · ui/
  scripts/      download_all.py · build_benchmark.py · make_tables.py
  notebooks/    exploration only, never the source of truth
  data/         (gitignored) raw/ · harmonised/
  results/      (gitignored except summary CSVs)
```

- **Python packages (3.11 venv):** torch (CUDA build; install from the cu124 index; uninstall
  any +cpu build first; `--retries 10 --timeout 120`), mne, moabb, braindecode, pyriemann,
  scikit-learn, scipy, pandas, peft, captum, onnx, onnxruntime, pylsl, pyyaml, matplotlib.
- **Download location:** point MOABB downloads at `data/raw` with `moabb.utils.set_download_dir`
  so C: doesn't fill up.
- **Device switch:** `HEMISHIFT_DEVICE=cpu|cuda`. scikit-learn and pyriemann run on CPU only.
- **Git:** own repo at https://github.com/prayag-1771/HemiShift, nested inside `G:\VSC_NEW`,
  branch `main`. It's public by the owner's choice from 2026-10-20 and will be made private
  later; that won't undo the disclosure. Commit after every meaningful change, with the owner as
  sole author and no co-author trailers.

## 15. What this shows on a resume

- multi-source data engineering (4 hospitals, 4 devices → one benchmark)
- signal processing
- augmentation design grounded in physiology
- classical ML, deep learning and foundation-model fine-tuning (LoRA)
- domain adaptation
- leakage-safe evaluation with proper statistics
- explainability
- real-time deployment (ONNX, streaming)
- a working demo, a patent filing and a journal paper

## 16. This week

1. Done: repo created at github.com/prayag-1771/HemiShift.
2. Set up the environment and confirm `torch.cuda.is_available()` is True.
3. Write and run `scripts/download_all.py`, then fill in the inventory table and licences.
4. Read Nagarajan 2024, Via 2026, Nguyen 2026, Mutual Generation (JBHI 2025), WalkBCI 2026 and
   CLEAN-MI.
5. Ask a supervisor (and, through them, a neurologist or physiatrist) to join.
