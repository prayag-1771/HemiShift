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


PHYSIONET_S3 = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/"
PHYSIONET_RUNS = (1, 2, 4, 6, 8, 10, 12, 14)  # what MOABB's PhysionetMI.data_path fetches


def prefetch_physionet_s3(n_subjects=109, workers=8):
    """Fetch EEGMMI from PhysioNet's AWS open-data mirror into MOABB's expected layout.

    physionet.org served ~5 kB/s here and kept timing out. The S3 copies are byte-identical,
    and each one's ETag is its MD5, so every file is verified before it is put in place.
    That matters because MOABB later trusts any file it finds there without re-checking.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    root = RAW_DIR / "moabb" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"

    def fetch(subject_run):
        """True once the file is in place. Never raises, so one file can't stop the batch."""
        subject, run = subject_run
        rel = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
        dest = root / rel
        if dest.exists():  # files are only renamed into place after verification
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            for attempt in range(1, 6):
                try:
                    head = requests.head(PHYSIONET_S3 + rel, timeout=60)
                    head.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 5:
                        raise
                    time.sleep(5 * attempt)
            etag = head.headers["ETag"].strip('"')
            md5 = None if "-" in etag else etag  # multipart ETags are not MD5s
            fetch_resumable(PHYSIONET_S3 + rel, dest, int(head.headers["Content-Length"]),
                            md5, label=f"[physionet] {rel}", max_attempts=10)
            return True
        except Exception as exc:
            print(f"[physionet] {rel}: gave up for now ({type(exc).__name__})")
            return False

    jobs = [(s, r) for s in range(1, n_subjects + 1) for r in PHYSIONET_RUNS]
    missing = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, ok in enumerate(pool.map(fetch, jobs), start=1):
            missing += not ok
            if done % 80 == 0 or done == len(jobs):
                print(f"[physionet] S3 mirror: {done}/{len(jobs)} files checked, {missing} missing")
    if missing:  # the caller's retry loop comes back for just these
        raise RuntimeError(f"{missing} EEGMMI files still missing")


def prefetch_liu2024():
    """Fetch the files MOABB's Liu2024 loader needs, resumably, into the exact spots it reads.

    That is edffile.zip (463 MB) plus the electrodes and events tables, which get_data()
    requests on every load. MOABB fetches them with pooch, which cannot resume (two zip
    attempts died at 323 and 404 MB) and times out at 30 s on a busy link. MOABB's data_dl
    trusts a file already in place, so each one is MD5-checked here first. Destinations come
    from MOABB's own path helpers (pinned 1.7.2).
    """
    import json
    import urllib.request
    from pathlib import Path

    from moabb.datasets.download import _normalize_destination, _sanitize_path, get_dataset_path
    from moabb.datasets.liu2024 import LIU2024_ELECTRODES, LIU2024_EVENTS, LIU2024_URL

    with urllib.request.urlopen("https://api.figshare.com/v2/articles/21679035", timeout=60) as r:
        listing = {str(f["id"]): f for f in json.load(r)["files"]}
    root = Path(get_dataset_path("Liu2024", None)) / "MNE-liu2024-data"
    for url in (LIU2024_URL, LIU2024_ELECTRODES, LIU2024_EVENTS):
        entry = listing[url.rsplit("/", 1)[-1]]
        dest = _sanitize_path(_normalize_destination(url, root))
        dest.parent.mkdir(parents=True, exist_ok=True)
        fetch_resumable(url, dest, entry["size"], entry["computed_md5"],
                        label=f"[liu2024] {entry['name']}")


def download_moabb(name, cohort, max_attempts=30):
    import time

    import moabb.datasets as moabb_datasets

    from hemishift.data.loaders import configure_moabb

    dataset_cls = getattr(moabb_datasets, cohort["source"], None)
    if dataset_cls is None:
        print(f"[{name}] skipped: installed MOABB has no {cohort['source']} loader")
        return False
    configure_moabb()
    dataset = dataset_cls()
    print(f"[{name}] {cohort['source']}: {len(dataset.subject_list)} subjects, ~{cohort['size_gb']} GB")
    # Files arrive atomically and finished subjects are skipped, so after a network drop
    # (or a network change) calling download() again picks up where it stopped.
    for attempt in range(1, max_attempts + 1):
        try:
            # Prefetch big or slow files resumably; download() then finds them in place.
            if name == "physionet":
                prefetch_physionet_s3()
            elif name == "liu2024":
                prefetch_liu2024()
            dataset.download(accept=True)
            return True
        except Exception as exc:
            print(f"[{name}] interrupted ({type(exc).__name__}); retry {attempt} shortly")
            time.sleep(min(120, 10 * attempt))
    return False


def fetch_resumable(url, dest, size, md5, label, max_attempts=60):
    """Download `url` to `dest`, resuming after stalls; verify size and MD5 at the end.

    Writes to `<dest>.part` and continues from its length with an HTTP Range request, so a
    timeout at 900 MB costs seconds, not the 900 MB. Each attempt re-requests `url`, which
    matters for figshare: it redirects to S3 links that expire after ~10 s.
    md5=None checks the size only.
    """
    import hashlib
    import time

    import requests

    if dest.exists() and dest.stat().st_size == size:
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, max_attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset < size:
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:  # server ignored Range
                        offset = 0
                    step = max(size // 20, 50_000_000)  # every 5 %, never for small files
                    next_report = offset + step
                    with open(part, "ab" if offset else "wb") as out:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            out.write(chunk)
                            offset += len(chunk)
                            if offset >= next_report:
                                print(f"{label}: {offset / 1e6:,.0f} / {size / 1e6:,.0f} MB")
                                next_report += step
            except (requests.RequestException, OSError) as exc:
                print(f"{label}: attempt {attempt} stalled at {offset / 1e6:,.0f} MB "
                      f"({type(exc).__name__}); resuming")
                time.sleep(min(60, 5 * attempt))
                continue
        digest = hashlib.md5()
        with open(part, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 23), b""):
                digest.update(block)
        if part.stat().st_size == size and md5 in (None, digest.hexdigest()):
            part.replace(dest)
            return dest
        print(f"{label}: checksum mismatch, starting this file again")
        part.unlink()
        time.sleep(5)
    raise RuntimeError(f"{label}: gave up after {max_attempts} attempts")


def download_figshare(name, cohort):
    import json
    import urllib.request
    import zipfile

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
        path = fetch_resumable(entry["download_url"], target / file_name, entry["size"],
                               entry["computed_md5"], label=f"[{name}] {file_name}")
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
