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


def download_figshare(name, cohort):
    import json
    import urllib.request
    import zipfile

    import pooch

    target = RAW_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    api_url = f"https://api.figshare.com/v2/articles/{cohort['article']}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        listing = {entry["name"]: entry for entry in json.load(response)["files"]}
    missing = [f for f in cohort["files"] if f not in listing]
    if missing:
        raise RuntimeError(f"not in figshare article {cohort['article']}: {missing}")

    for file_name in cohort["files"]:
        entry = listing[file_name]
        print(f"[{name}] {file_name} ({entry['size'] / 1e9:.2f} GB)")
        # figshare redirects to S3 links that expire after ~10 s, so a slow hand-off fails
        # with 403; a fresh request gets a fresh link.
        for attempt in range(1, 4):
            try:
                path = pooch.retrieve(
                    url=entry["download_url"],
                    known_hash=f"md5:{entry['computed_md5']}",  # re-downloads on mismatch
                    fname=file_name,
                    path=target,
                    progressbar=True,
                )
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"[{name}] {file_name}: attempt {attempt} failed ({exc!r:.80}), retrying")
        if file_name.endswith(".zip"):
            extract_dir = target / file_name.removesuffix(".zip")
            if not extract_dir.exists():
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(extract_dir)  # zipfile drops absolute and ".." member paths
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

    fetchers = {"moabb": download_moabb, "figshare": download_figshare, "git": download_git}
    failed = []
    for name in names:
        cohort = COHORTS[name]
        fetch = fetchers[cohort["kind"]]
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
