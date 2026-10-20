"""Inventory the downloaded MOABB cohorts: what is actually inside each one.

For every cohort it records subjects, sessions/runs, channel names, sampling rate, recording
length and event labels with counts, so the harmonisation choices in PLAN.md §3 rest on the
data rather than on the papers (S1's paper disagrees with itself on channels and trials).

Usage, from the project root:
    .venv/Scripts/python -I scripts/inventory.py                   # all downloaded MOABB cohorts
    .venv/Scripts/python -I scripts/inventory.py liu2024 --subjects 3
Writes results/summary/inventory_<cohort>.json and prints a short table.
"""

import argparse
import json
from collections import Counter

import mne

from hemishift.data.registry import COHORTS
from hemishift.paths import RAW_DIR, RESULTS_DIR


def count_events(raw, dataset_event_id):
    """Event label counts, whether MOABB stored markers in a stim channel or as annotations."""
    stim_channels = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "stim"]
    if stim_channels:
        events = mne.find_events(raw, stim_channel=stim_channels[0], shortest_event=0, verbose=False)
        code_to_label = {code: label for label, code in dataset_event_id.items()}
        return Counter(code_to_label.get(int(code), f"code_{int(code)}") for code in events[:, 2])
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_to_label = {code: label for label, code in event_id.items()}
    return Counter(code_to_label[code] for code in events[:, 2])


def describe_raw(raw, dataset_event_id):
    counts = count_events(raw, dataset_event_id)
    return {
        "n_channels": len(raw.ch_names),
        "channels": raw.ch_names,
        "channel_types": dict(Counter(raw.get_channel_types())),
        "sfreq": raw.info["sfreq"],
        "duration_s": round(raw.times[-1], 1),
        "events": dict(counts),
    }


def inventory_moabb(name, cohort, n_subjects):
    import moabb.datasets as moabb_datasets
    from moabb.utils import set_download_dir

    set_download_dir(str(RAW_DIR / "moabb"))
    dataset = getattr(moabb_datasets, cohort["source"])()
    report = {
        "cohort": name,
        "moabb_class": cohort["source"],
        "moabb_doc": (dataset.__doc__ or "").strip().splitlines()[0],
        "n_subjects": len(dataset.subject_list),
        "dataset_event_id": dataset.event_id,
        "subjects": {},
    }
    for subject in dataset.subject_list[:n_subjects]:
        sessions = dataset.get_data(subjects=[subject])[subject]
        report["subjects"][str(subject)] = {
            f"{session}/{run}": describe_raw(raw, dataset.event_id)
            for session, runs in sessions.items()
            for run, raw in runs.items()
        }
    return report


def print_summary(report):
    print(f"\n== {report['cohort']} ({report['moabb_class']}): {report['moabb_doc']}")
    print(f"   subjects: {report['n_subjects']} | dataset event_id: {report['dataset_event_id']}")
    for subject, runs in report["subjects"].items():
        for run_key, info in runs.items():
            print(
                f"   sub {subject} {run_key}: {info['n_channels']} ch {info['channel_types']}, "
                f"{info['sfreq']:g} Hz, {info['duration_s']} s, events {info['events']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Inventory downloaded MOABB cohorts.")
    parser.add_argument("names", nargs="*", help="cohort keys (default: all MOABB cohorts)")
    parser.add_argument("--subjects", type=int, default=2, help="subjects to open per cohort")
    args = parser.parse_args()

    names = args.names or [n for n, c in COHORTS.items() if c["kind"] == "moabb"]
    out_dir = RESULTS_DIR / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        cohort = COHORTS[name]
        if cohort["kind"] != "moabb":
            print(f"[{name}] not a MOABB cohort; inspect it with the cohort-specific loader")
            continue
        try:
            report = inventory_moabb(name, cohort, args.subjects)
        except Exception as exc:  # report and continue with the other cohorts
            print(f"[{name}] FAILED: {exc!r}")
            continue
        (out_dir / f"inventory_{name}.json").write_text(json.dumps(report, indent=2))
        print_summary(report)


if __name__ == "__main__":
    main()
