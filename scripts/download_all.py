"""Download HemiShift cohorts into data/raw.

Usage, from the project root:
    .venv/Scripts/python -I scripts/download_all.py                 # all non-deferred cohorts
    .venv/Scripts/python -I scripts/download_all.py liu2024 uet175  # selected cohorts

Downloads are untrusted data: nothing under data/raw is ever imported or executed.
"""

import argparse
import subprocess
import sys

from hemishift.data.registry import COHORTS
from hemishift.paths import RAW_DIR


def download_moabb(name, cohort):
    import moabb.datasets as moabb_datasets
    from moabb.utils import set_download_dir

    dataset_cls = getattr(moabb_datasets, cohort["source"], None)
    if dataset_cls is None:
        print(f"[{name}] skipped: installed MOABB has no {cohort['source']} loader")
        return False
    set_download_dir(str(RAW_DIR / "moabb"))
    dataset = dataset_cls()
    print(f"[{name}] {cohort['source']}: {len(dataset.subject_list)} subjects, ~{cohort['size_gb']} GB")
    dataset.download(accept=True)
    return True


def download_git(name, cohort):
    target = RAW_DIR / name
    if target.exists():
        print(f"[{name}] already present at {target}")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", cohort["source"], str(target)], check=True)
    # Pin the exact snapshot so every rebuild sees the same files.
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", cohort["commit"]], check=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download HemiShift cohorts into data/raw.")
    parser.add_argument("names", nargs="*", help=f"cohort keys; known: {', '.join(COHORTS)}")
    args = parser.parse_args()

    names = args.names or [n for n, c in COHORTS.items() if not c.get("deferred")]
    unknown = [n for n in names if n not in COHORTS]
    if unknown:
        sys.exit(f"unknown cohorts: {unknown}")

    failed = []
    for name in names:
        cohort = COHORTS[name]
        fetch = download_moabb if cohort["kind"] == "moabb" else download_git
        try:
            ok = fetch(name, cohort)
        except Exception as exc:  # keep going so one bad mirror doesn't block the rest
            print(f"[{name}] FAILED: {exc!r}")
            ok = False
        if not ok:
            failed.append(name)

    print("finished; failed or skipped:", ", ".join(failed) or "none")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
