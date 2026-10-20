# HemiShift — Literature and Novelty Check

Checked 2026-10-20. "Not found" means it didn't turn up in targeted searches,
not that it's proven not to exist. Re-run the searches in §4 right before
submission, because this field moves monthly.

Verification status:
- **[V]** I confirmed the title, authors and venue from the publisher page or PubMed/Europe PMC.
- **[P]** I confirmed the title and venue, but the author list still needs checking.
  Never cite a [P] entry until you've opened it yourself.

---

## 1. Verdict on the three questions

| Question | Answer | Consequence for us |
|---|---|---|
| Has anyone fine-tuned EEG foundation models on stroke MI? | **Yes, as of Aug 2026.** Nguyen et al. (arXiv 2609.00282) ran LaBraM and REVE with LoRA on UET175 only. | It can't be our headline anymore. It becomes a baseline arm, and their numbers are a reference. |
| Has anyone built a harmonised multi-site stroke MI benchmark? | **Not found.** Every paper below uses one stroke cohort. CLEAN-MI only *lists* Liu2024 in a review table and runs no stroke experiments. | **Open. This is contribution C1.** |
| Has anyone simulated stroke-like EEG from healthy EEG, with physiology-based edits, as augmentation? | **Not found.** The closest work generates EEG across hemiplegic sides (JBHI 2025) or from the patient's own resting EEG (WalkBCI, IJNS 2026). | **Open. This is contribution C2.** |
| Extra: cross-hospital generalisation (train on hospital A, test on hospital B)? | **Not found.** | **Open. This is the headline test (protocol P4).** |
| Extra: predicting or preventing negative transfer per patient? | A 2026 preprint *measured* it: 21 of 50 patients got worse. Nobody has fixed it. | **Open. This is contribution C3.** |

---

## 2. Papers you must cite and beat

### Healthy-to-stroke transfer (the problem itself)

- **[V] Nagarajan A, Robinson N, Ang KK, Chua KSG, Chew E, Guan C.** "Transferring a deep learning
  model from healthy subjects to stroke patients in a motor imagery brain–computer interface."
  *Journal of Neural Engineering* 21(1):016007, 2024. doi:10.1088/1741-2552/ad152f
  - Healthy OpenBMI to **71 stroke patients (private data)**, MI-vs-rest detection.
  - Results: **71.15 ± 12.46 %**, +18.15 % over the pretrained model, beats subject-specific
    Deep ConvNet by 7.64 % and FBCSP by 5.55 %.
  - **Closest prior work.** Their stroke data isn't public, so we can't compare head-to-head.
    We can reproduce their *recipe* on public data.

- **[V] Via Z, Kruse A, Thapa BR, Bae J.** "Healthy-to-Stroke Translation of EEG-Based BMIs: EEG
  Characterization and Reinforcement Learning–Based Decoder Evaluation." *bioRxiv*, posted
  2026-06-29. doi:10.64898/2026.06.23.733831
  - Uses **109 healthy subjects (= PhysioNet EEGMMI) and 50 acute stroke patients (= Liu 2024)**,
    exactly our data pairing.
  - Q-KTD decoder; mean first-epoch success went from **49.46 % to 51.82 %**.
  - **21/50 patients had negative transfer.** Stroke EEG showed "more diffuse time–frequency
    modulation", and no channel-level band-power difference survived FDR correction.
  - This number (42 % negative transfer) is the target for contribution C3.

- **[V] Nguyen AT, Sun Z, Johnson MJ.** "Lightweight Adaptation of EEG Foundation Models for Stroke
  Motor Imagery Decoding: Domain Shift and Subject-Level Robustness." *arXiv* 2609.00282,
  submitted 2026-08-31.
  - LaBraM-base, REVE-base and REVE-large; head-only vs LoRA; healthy EEGMMI, stroke **UET175 only**.
  - Head-only on UET175 is near chance. LoRA: LaBraM 0.499, REVE-base 0.847 ± 0.194,
    REVE-large 0.806. LOOCV REVE-base LoRA 0.952 mean (patient range 0.586–1.000).
  - **Zero-shot EEGMMI to UET175: 0.464**, at chance.
  - No augmentation, no synthetic data, a single stroke dataset. Those three gaps are ours.

- **[P]** "Motor Imagery Classification Based on Temporal-Spatial Domain Adaptation for Stroke
  Patients." *Cognitive Computation*, 2025. doi:10.1007/s12559-025-10477-3
  - Healthy-trained model plus alignment losses; reported about 51.7 %. Authors still to verify.

### Generative and augmentation work for stroke (closest to contribution C2)

- **[V] Xu F, Sun Y, Lun Z, Zhao P, Zhang Z, Zhang C, Zhang A, Feng C, Zhang Y, Jung TP, Leng J.**
  "Enhancing Motor Imagery Decoding for Stroke Patients Using Data Augmentation and Transfer
  Learning." *International Journal of Neural Systems* 36(13):2650051, 2026.
  doi:10.1142/S0129065726500516
  - WalkBCI: an RM-GAN creates MI features from the **target patient's own resting EEG**.
  - Offline +3.3 % accuracy; online up to 72.5 %; calibration time −58 %.
  - Difference from us: they generate from the patient's rest data. We transform **healthy
    donors' MI** into stroke-like MI using physiology.

- **[P]** "Mutual Generation for Cross-Domain Challenge in Stroke Patients' Motor Imagery
  Classification and Functional Recovery Prediction." *IEEE JBHI*, 2025.
  doi:10.1109/JBHI.2025.3646871
  - Generative model across **left- vs right-hemiplegic** domains. **Must read in full before
    finalising C2** (IEEE page didn't load). Authors still to verify.

- **[P]** "Fail closed trust gated synthetic augmentation governs tail risk under subject shift in
  EEG." *bioRxiv*, 2026. doi:10.64898/2026.01.26.701638
  - Synthetic data should only be used when it helps validation, because it can hurt under subject
    shift. Relevant to the safety gate (C3). Authors still to verify.

### Foundation-model caution (why FMs are one arm, not the hero)

- **[P]** "EEG Foundation Models: Progresses, Benchmarking, and Open Problems." *arXiv* 2601.17883,
  2026.
  - 12 open FMs; specialist models trained from scratch are often competitive.
- **[P]** "Matched-Input Estimates Differ in Sign Across Architectures: Auditing EEG Foundation
  Models on Motor Imagery." *arXiv* 2609.23924, 2026.
  - The best fine-tuning recipe differs between LaBraM and CBraMod.
- **[P]** CBraMod, ICLR 2025, arXiv 2412.07236. Code and weights: github.com/wjq-learning/cbramod

### Harmonisation tooling

- **[V] Liu D, Chen Z, Wu D.** "CLEAN-MI: A Scalable and Efficient Pipeline for Constructing
  High-Quality Neurodata in Motor Imagery Paradigm." *arXiv* 2506.11830, 2025.
  - Band filtering, channel templates, subject screening, marginal alignment across MI datasets.
  - Full text checked 2026-10-20: it reviews 18 MI datasets including **Liu2024**, but only lists
    it in a table. Experiments use Weibo2014, Cho2017 and BNCI2015001. No stroke harmonisation and
    no cross-site stroke evaluation, so **C1 stands**. Cite it as the closest harmonisation tool.
- **MOABB** (`moabb.datasets`) has loaders for `Liu2024`, `Liu2025`, `Wang2026`,
  `HefmiIch2025`, `PhysionetMI`, `Lee2019_MI`, `BNCI2014_001`, `Cho2017` and `Stieger2021`
  (checked against the develop-branch `__init__.py`). Confirmed in MOABB 1.7.2 on 2026-10-20:
  `Liu2025` and `HefmiIch2025` are the stroke sets below. **`Wang2026` is NOT the sixth-finger
  stroke set.** It's a healthy 39-subject "sensory-guided joint-learning" study (Nature
  Communications 2026, doi:10.1038/s41467-026-75435-5), so S3 needs its own loader.

---

## 3. Datasets (details from the data papers)

- **[V] Liu H, Wei P, Wang H, Lv X, Duan W, Li M, Zhao Y, Wang Q, Chen X, Shi G, Han B, Hao J.**
  "An EEG motor imagery dataset for brain computer interface in acute stroke patients."
  *Scientific Data* 11:131, 2024. doi:10.1038/s41597-023-02787-8
  - 50 acute patients (1–30 days), Xuanwu Hospital, Beijing; 27 left / 23 right hemiplegia.
  - ZhenTec NT1 wireless semi-dry, 500 Hz, ref CPz.
  - **Paper is inconsistent: 29 vs 30 EEG channels; 40 trials vs "20 used".** Verify on download.
  - Left vs right hand MI; 8 s trials (4 s MI).
  - Metadata: age, sex, side, NIHSS, MBI, mRS, days since stroke.
  - Data: figshare doi:10.6084/m9.figshare.21679035.v5. Dataset licence not stated in the article;
    check figshare.
  - Paper's baselines (within-patient, 60/40 split + 10-fold): CSP+LDA **55.57 %**,
    FBCSP+SVM **57.57 %**, TSLDA+DGFMDRM 61.20 %, their TWFB+DGFMDRM **72.21 %**.
- **[V] Ma Thi C, Nguyen The H-A, Nguyen Minh K, Vu Thanh L, Nguyen Dinh H, Huynh Thi N-Y, Ha Thi T-H,
  Hoang Tien T-N, Au Dao D-T, Nguyen Hoang K-L, Huynh Kha V, Le Hoang T-L.** "UET175: EEG dataset of
  motor imagery tasks in Vietnamese stroke patients." *Frontiers in Neuroscience* 19:1580931, 2025.
  doi:10.3389/fnins.2025.1580931
  - 30 patients; Emotiv EPOC Flex saline, 22 motor channels, 128 Hz.
  - 4 classes (left/right arm lift, left/right leg lift); 2 s relax, 2 s cue, 4 s MI.
  - Data: github.com/nmk-k66-uet/UET175 (434 MB). No baselines in the paper.
  - **⚠ Trials were kept only if they showed ≥30 % ERD.** That's a selection bias that makes
    UET175 easier than real use, and it likely explains the high FM numbers above. Always report
    it separately and say so.
- **[V] Wang Z, Liu Y, Huang S, Wu W, Huang H, Gui Z, Li Z, Guo J, Zhang H, Xu M, Ming D.** "A
  multi-paradigm and longitudinal EEG dataset including the 'sixth-finger' and 'affected-hand'
  motor imagery of stroke patients." *Scientific Data* 13:1109, 2026. doi:10.1038/s41597-026-07787-y
  - 24 chronic patients (3–14 months), Tianjin Huanhu Hospital; 15 right / 9 left hemiplegia.
  - Neuracle 64-ch (40 kept), 250 Hz.
  - Classes: **rest / affected-hand MI / sixth-finger MI**, at 3 time points.
  - Data: figshare doi:10.6084/m9.figshare.30509165.
  - **Article is CC BY-NC-ND 4.0.** If the data is too, publish code but not processed derivatives.
- **[P]** Liu et al. "Lower limb motor imagery EEG dataset based on the multi-paradigm and
  longitudinal-training of stroke patients." *Scientific Data*, 2025. doi:10.1038/s41597-025-04618-4
  - 27 patients, lower-limb MI, 3 time points.
  - **Licence conflict between mirrors (CC BY vs CC BY-NC-ND)**; check the original figshare.
- **[P] Shi J, Chen D, et al.** "HEFMI-ICH: a hybrid EEG-fNIRS motor imagery dataset for
  brain-computer interface in intracerebral hemorrhage." *Scientific Data*, 2025.
  doi:10.1038/s41597-025-06100-7. Full author list still to verify.
  - 37 subjects (17 healthy, 20 ICH); g.HIamp 32-ch EEG, 256 Hz.
  - Left/right hand grasp MI, 15 trials/class per session; 2 s cue + 10 s MI + 15 s rest;
    1–6 sessions.
  - Figshare doi:10.6084/m9.figshare.28955456.v4, **CC BY 4.0**, 13.2 GB total (EEG epochs
    ≈ 8.5 GB). MOABB `HefmiIch2025`.
  - **Healthy and patients on the same device**, which separates the disease effect from the
    device effect.
  - Related method paper on ICH (13 ICH + 17 controls): *Advanced Science* 2025,
    doi:10.1002/advs.202505426.
- Healthy (well known; numbers from memory, verify the data-paper citations before writing):
  - PhysioNet EEGMMI: 109 subjects, 64 ch, 160 Hz.
  - OpenBMI / Lee2019: 54 subjects, 62 ch, 1000 Hz, 2 sessions.
  - BCI IV-2a: 9 subjects, 22 ch, 250 Hz.
  - Cho2017: 52 subjects, 64 ch, 512 Hz.

---

## 4. Searches to re-run before submission

- "stroke motor imagery" + (foundation model | LoRA | LaBraM | CBraMod | REVE)
- "stroke" + EEG + (simulation | synthetic | augmentation | generative | translation) + healthy
- "multi-site" OR "cross-dataset" + stroke + motor imagery
- "negative transfer" + stroke + BCI
- patient-specific / personalised simulation of stroke EEG for BCI calibration. Check this before
  claiming the patient-matched mode (PLAN §4.0); it wasn't searched separately.

---

## 5. Why accuracy is low: evidence (checked 2026-10-20)

**Headline.** About 70 % is roughly the ceiling of non-invasive MI decoding for an average user,
healthy or not. What stroke mainly breaks is **transfer between people**.

| Cause | Evidence |
|---|---|
| MI-EEG is hard for everyone | **[P]** Lee M-H et al. "EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy." *GigaScience* 8(5), 2019. doi:10.1093/gigascience/giz002. 54 healthy subjects: mean MI accuracy **71.1 % (± 15)**; MI "BCI illiteracy" (< 70 %) in **53.7 %** of them. |
| Resting mu-rhythm strength predicts who can use MI-BCI | **[P]** Blankertz B et al. "Neurophysiological predictor of SMR-based BCI performance." *NeuroImage* 51(4):1303–1309, 2010. Uses a 2-min rest recording and 2 Laplacian channels; r = 0.53, N = 80; about 20 % don't reach criterion. |
| Stroke ≈ healthy when the decoder is trained on the patient | **[V]** Ang KK, Guan C, Chua KSG, Ang BT, Kuah CWK, Wang C, Phua KS, Chin ZY, Zhang H. "A large clinical study on the ability of stroke patients to use an EEG-based motor imagery brain-computer interface." *Clinical EEG and Neuroscience* 42(4), 2011. doi:10.1177/155005941104200411. 46 patients: MI detection **0.74** vs healthy **0.78** (p = 0.23); finger tapping 0.87; **6/46 at chance**; no correlation with Fugl-Meyer (p = 0.29); 11 retested patients went 0.72 → 0.76, and 0.82 online. |
| The signal moves to the healthy hemisphere as severity rises | **[V]** Mansour S, Giles J, Ang KK, Nair KPS, Phua KS, Arvaneh M. "Exploring the ability of stroke survivors in using the contralesional hemisphere to control a brain-computer interface." *Scientific Reports* 12:16223, 2022. doi:10.1038/s41598-022-20345-x. 136 patients. Ipsilesional accuracy < 60 % goes with **more impairment and higher contralesional accuracy**; ipsilesional > 80 % goes with poorer contralesional accuracy. (It cites Kaiser et al.: more impaired patients show more contralesional ERD. The original isn't checked.) |
| Stroke EEG is more variable and more diffuse | Via et al. 2026 (§2): no band-power difference survives FDR, but stroke shows greater variability and more diffuse time–frequency modulation. |
| MI ability is impaired after stroke in some patients | **[P]** "Assessment of Motor Imagery and Its Relationship to Functional Impairment in Post-Stroke Patients: A Descriptive Cross-Sectional Study," 2024 (PMC11477988). 36 patients; implicit MI ability tracks physical function. |
| Transfer collapses to chance | Nguyen 2026: zero-shot EEGMMI → UET175 scores 0.464. Via 2026: 49.46 % → 51.82 %, with 21/50 negative transfer. |
| Few trials make per-patient numbers noisy | Exact one-sided binomial thresholds (p < 0.05), computed here. Test trials → accuracy needed to beat chance (± 1 SE at 70 %): 16 → **75 %** (± 11.5 pts); 20 → 75 %; 24 → 70.8 %; 40 → 65 % (± 7.2); 100 → 59 % (± 4.6). For background, cite **[P]** Combrisson & Jerbi 2015 and **[P]** Müller-Putz et al. 2008 ("Better than random?"), after checking both. |
| Device and trial count matter | Wang 2026 (64-ch lab cap, many trials, sixth-finger vs affected-hand MI): CSP ≈ 85 % within-patient. Liu 2024 (portable semi-dry cap, 40 trials, left vs right): 55–72 %. The tasks differ, so this only illustrates the point. |
| Inflated 90 %+ claims | **[P]** Brookshire G et al. "Data leakage in deep learning studies of translational EEG." *Frontiers in Neuroscience*, 2024. doi:10.3389/fnins.2024.1373515. Segment-based splits put the same subject in train and test, which inflates accuracy. UET175's ERD-based trial selection (§3) inflates it the same way. |

**Implications (already in PLAN.md):**
- Target new-patient transfer: ≈ 50 % up toward the 70–80 % within-patient ceiling.
- O2 and patient-matched mode are core, not optional.
- Prefer Task B where it's available.
- Report per-patient binomial significance.
- The gate's abstain covers the roughly 1 in 8 patients with no usable signal.
- Google Scholar "cited by" on Liu 2024 (Sci Data) and UET175. Anyone using these datasets is a
  potential competitor.
