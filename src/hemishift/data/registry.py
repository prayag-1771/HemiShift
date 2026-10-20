"""Every cohort HemiShift uses, in one place.

Licences were checked at the original repositories on 2026-10-20 (PLAN.md §3).
`kind` says how to fetch it:
  moabb    -> `source` is a MOABB dataset class name
  figshare -> `article` is a figshare article id; `files` lists what to download
  git      -> `source` is a git URL, pinned to `commit`
Order matters: download_all.py fetches cohorts in this order (week-2 fast path first).
`deferred` cohorts are only downloaded when named explicitly (disk budget).
"""

COHORTS = {
    "liu2024": dict(
        kind="moabb", source="Liu2024", role="stroke",
        site="Xuanwu Hospital, Beijing", licence="CC BY 4.0",
        doi="10.6084/m9.figshare.21679035.v5", size_gb=2.3,
    ),
    "physionet": dict(
        kind="moabb", source="PhysionetMI", role="healthy",
        site="PhysioNet EEGMMI (BCI2000)", licence="ODC-By 1.0 (confirm)", size_gb=2.0,
    ),
    "hefmi_ich": dict(
        # 20 intracerebral-haemorrhage patients (subjects 18-37) + 17 healthy controls (1-17)
        # on the same g.HIamp 32-channel system: separates disease effects from device effects.
        kind="moabb", source="HefmiIch2025", role="stroke+healthy",
        site="HEFMI-ICH (Shi et al., Sci Data 2025)", licence="CC BY 4.0",
        doi="10.6084/m9.figshare.28955456.v4", size_gb=8.5,
    ),
    "wang2026sf": dict(
        # Sixth-finger / affected-hand stroke dataset. Not MOABB's `Wang2026`, which is a
        # different (healthy, Nature Communications 2026) study.
        # From its BIDS metadata (checked 2026-10-20): Neuracle NeuSen W, 250 Hz, 40 electrodes
        # listed (task json says 60), ground AFz. Events (ms): 1 = rest starts (8 s),
        # 2 = affected-hand MI starts (6 s), 3 = sixth-finger MI starts (6 s), so each 14 s trial
        # is rest then MI. participants.tsv: age, sex, Hemiplegia_side (L/R), duration (months),
        # Collection_phase (pre/post/follow-up). The json also documents a `diagnosis` column
        # that the tsv lacks.
        kind="figshare", article=30509165, role="stroke",
        site="Tianjin Huanhu Hospital", licence="CC BY 4.0 (data; article is CC BY-NC-ND)",
        doi="10.6084/m9.figshare.30509165.v11", size_gb=8.7,
        files=[
            "sourcedata.zip", "processeddata.zip", "README.md", "dataset_description.json",
            "participants.tsv", "participants.json", "events.tsv", "events.json",
            "electrodes.tsv", "task-MotorImagery_eeg.json",
        ],
    ),
    "bnci2014_001": dict(
        kind="moabb", source="BNCI2014_001", role="healthy",
        site="BCI Competition IV-2a, Graz", licence="confirm", size_gb=0.5,
    ),
    "uet175": dict(
        kind="git", source="https://github.com/nmk-k66-uet/UET175",
        commit="e05a1ae12421514d2392db169f20437b1c60673d", role="stroke",
        site="Hospital 175, Ho Chi Minh City", licence="CC BY (README; no LICENSE file)",
        doi="10.3389/fnins.2025.1580931", size_gb=0.5,
        # Data_UET175.rar is encrypted. The password comes from the authors on request
        # (chaumt@vnu.edu.vn, kienprb@gmail.com). Layout: ID01-ID30/<session>/RunN_{Raw Signals.csv,
        # Action label.txt, Event timestamp.txt, Session setup.json}, plus Info.json per patient.
        needs_password=True,
    ),
    "liu2025": dict(
        kind="moabb", source="Liu2025", role="stroke",
        site="Tianjin (lower limb; gait imagery vs rest)", licence="CC BY",
        doi="10.6084/m9.figshare.27130299", size_gb=21.8, deferred=True,
    ),
}
