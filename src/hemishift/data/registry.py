"""Every cohort HemiShift uses, in one place.

Licences were checked at the original repositories on 2026-10-20 (PLAN.md §3).
`source` is a MOABB class name when kind == "moabb", and a git URL when kind == "git".
`deferred` cohorts are only downloaded when named explicitly (disk budget).
"""

COHORTS = {
    # Stroke cohorts
    "liu2024": dict(
        kind="moabb", source="Liu2024", role="stroke",
        site="Xuanwu Hospital, Beijing", licence="CC BY 4.0",
        doi="10.6084/m9.figshare.21679035.v5", size_gb=2.3,
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
    "wang2026": dict(
        kind="moabb", source="Wang2026", role="stroke",
        site="Tianjin Huanhu Hospital", licence="CC BY 4.0 (data; article is CC BY-NC-ND)",
        doi="10.6084/m9.figshare.30509165", size_gb=8.7,
    ),
    "liu2025": dict(
        kind="moabb", source="Liu2025", role="stroke",
        site="Tianjin (lower limb)", licence="CC BY",
        doi="10.6084/m9.figshare.27130299", size_gb=21.8, deferred=True,
    ),
    # Healthy cohorts
    "physionet": dict(
        kind="moabb", source="PhysionetMI", role="healthy",
        site="PhysioNet EEGMMI (BCI2000)", licence="ODC-By 1.0 (confirm)", size_gb=2.0,
    ),
    "bnci2014_001": dict(
        kind="moabb", source="BNCI2014_001", role="healthy",
        site="BCI Competition IV-2a, Graz", licence="confirm", size_gb=0.5,
    ),
}
