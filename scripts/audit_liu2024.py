"""Step 1 of the S1 deep-dive: what is actually in each development patient's recording?

Development patients only (configs/splits/liu2024_locked_test.json); the locked test
patients are never opened here. Per patient:
  technical      channels to interpolate: flat, broadband-noisy, or decorrelated from their
                 neighbours (hemishift.preprocess.quality)
  slow           channels with excess 1-6 Hz power: possible lesion slowing, kept, not repaired;
                 slow_on_lesion_side counts those on the hemisphere opposite the paralysis
  line_noise     50 Hz power relative to 45-55 Hz neighbours (median over channels)
  bad_trials     trials whose peak amplitude has robust |z| > 3.5
  eog_class_d    |Cohen's d| of HEOL/VEOR power between left- and right-hand MI trials
                 (eye movements that differ by class could fake decodability)
  lat_<band>_d   Cohen's d (left vs right trials) of log power C3 - C4, MI 0.5-4 s:
                 the classic lateralised signal; lat_<band>_p is its Mann-Whitney p
  iaf_hz         rhythm peak above the 1/f background, 6-14 Hz, at C3/C4/Pz over the whole
                 recording (NaN when there is no peak)
Writes results/summary/audit_liu2024_dev.csv. Descriptive only, not a decoder.

Usage: .venv/Scripts/python -I scripts/audit_liu2024.py
"""

import json

import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

from hemishift.data.channels import hemisphere
from hemishift.data.liu2024_raw import EXCLUDED, SFREQ, load_raw, participants
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.quality import aperiodic_peak, channel_quality

BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}


def robust_z(values):
    values = np.asarray(values, dtype=float)
    mad = np.median(np.abs(values - np.median(values))) * 1.4826
    return (values - np.median(values)) / (mad if mad > 0 else 1.0)


def cohens_d(a, b):
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                     / (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / pooled


def band_log_power(epochs, picks, band):
    freqs, psd = welch(epochs.get_data(picks=picks), fs=SFREQ, nperseg=int(SFREQ))
    mask = (freqs >= band[0]) & (freqs < band[1])
    return np.log(psd[..., mask].mean(-1))  # (n_trials, n_picks)


MOTOR = {"F3", "Fz", "F4", "FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "P3", "Pz", "P4"}


def audit(subject, paralysis_side):
    raw = load_raw(subject)
    data = raw.get_data(picks="eeg")
    freqs, psd = welch(data, fs=SFREQ, nperseg=int(2 * SFREQ))
    line = psd[:, np.argmin(np.abs(freqs - 50))] / psd[:, (np.abs(freqs - 50) > 2) & (np.abs(freqs - 50) < 5)].mean(1)

    filtered = raw.copy().filter(1.0, 40.0, verbose="error")
    quality = channel_quality(filtered)
    technical = quality.loc[quality["interpolate"], "channel"].tolist()
    slow = quality.loc[quality["slow"], "channel"].tolist()
    lesion = "right" if paralysis_side == "left" else "left"  # lesion is opposite the paralysis
    events, event_id = mne.events_from_annotations(filtered, verbose="error")
    mi = mne.Epochs(filtered, events, event_id, tmin=0.5, tmax=4.0, baseline=None,
                    picks=["eeg", "eog"], preload=True, verbose="error")
    pre = mne.Epochs(filtered, events, event_id, tmin=-2.0, tmax=0.0, baseline=None,
                     picks="eeg", preload=True, verbose="error")
    is_left = mi.events[:, 2] == event_id["left_hand"]
    peaks = np.abs(mi.get_data(picks="eeg")).max(axis=(1, 2))

    row = {"subject": subject, "n_technical": len(technical), "technical": " ".join(technical),
           "n_technical_motor": len(MOTOR & set(technical)),
           "n_slow": len(slow), "slow": " ".join(slow),
           "slow_on_lesion_side": sum(hemisphere(ch) == lesion for ch in slow),
           "line_noise": float(np.median(line)), "n_bad_trials": int((robust_z(peaks) > 3.5).sum())}
    eog = band_log_power(mi, ["HEOL", "VEOR"], (1.0, 40.0)).mean(1)
    row["eog_class_d"] = abs(cohens_d(eog[is_left], eog[~is_left]))
    for name, band in BANDS.items():
        power = band_log_power(mi, ["C3", "C4"], band)
        lateral = power[:, 0] - power[:, 1]
        row[f"lat_{name}_d"] = cohens_d(lateral[is_left], lateral[~is_left])
        row[f"lat_{name}_p"] = stats.mannwhitneyu(lateral[is_left], lateral[~is_left]).pvalue
    # Rhythm peak above the 1/f background, from the whole recording at 0.25 Hz resolution.
    f, p = welch(filtered.get_data(picks=["C3", "C4", "Pz"]), fs=SFREQ, nperseg=int(4 * SFREQ))
    row["iaf_hz"] = aperiodic_peak(f, p.mean(0))
    return row


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    clinical = participants()
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            print(f"sub {subject:2d}: excluded ({EXCLUDED[subject]})")
            continue
        row = audit(subject, clinical.loc[subject, "ParalysisSide"])
        row.update(clinical.loc[subject, ["ParalysisSide", "NIHSS", "StrokeLocation"]].to_dict())
        rows.append(row)
        print(f"sub {subject:2d}: technical {row['technical'] or '-'} | slow {row['slow'] or '-'} "
              f"(lesion side {row['slow_on_lesion_side']}) | bad trials {row['n_bad_trials']} | "
              f"lateral d mu {row['lat_mu_d']:+.2f} beta {row['lat_beta_d']:+.2f} | IAF {row['iaf_hz']:.2f} Hz")
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / "audit_liu2024_dev.csv"
    table.to_csv(out, index=False)

    print(f"\n{len(table)} development patients (results in {out})")
    print(f"technical faults (to interpolate): {(table.n_technical > 0).sum()} patients, "
          f"{(table.n_technical_motor > 0).sum()} with a faulty motor channel")
    slow = table[table.n_slow > 0]
    print(f"pathological-looking slowing: {len(slow)} patients; slow channels on the lesion side: "
          f"{slow.slow_on_lesion_side.sum()} of {slow.n_slow.sum()}")
    print(f"median bad trials: {table.n_bad_trials.median():.0f} | median line-noise ratio: "
          f"{table.line_noise.median():.1f}")
    print(f"EOG differs by class (|d| > 0.5): {(table.eog_class_d > 0.5).sum()} patients")
    for band in BANDS:
        print(f"lateralised {band} signal at p < 0.05: {(table[f'lat_{band}_p'] < 0.05).sum()} patients "
              f"(expected by chance: {0.05 * len(table):.1f}); median |d| {table[f'lat_{band}_d'].abs().median():.2f}")
    peaks = table.iaf_hz.dropna()
    print(f"rhythm peak above 1/f: found in {len(peaks)}/{len(table)} patients; median "
          f"{peaks.median():.2f} Hz, range {peaks.min():.2f}-{peaks.max():.2f}")


if __name__ == "__main__":
    main()
