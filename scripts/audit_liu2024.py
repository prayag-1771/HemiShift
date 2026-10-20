"""Step 1 of the S1 deep-dive: what is actually in each development patient's recording?

Development patients only (configs/splits/liu2024_locked_test.json); the locked test
patients are never opened here. Per patient:
  bad_channels   EEG channels with robust |z| > 3.5 in log-variance or in 30-40 Hz noise
  line_noise     50 Hz power relative to 45-55 Hz neighbours (median over channels)
  bad_trials     trials whose peak amplitude has robust |z| > 3.5
  eog_class_d    |Cohen's d| of HEOL/VEOR power between left- and right-hand MI trials
                 (eye movements that differ by class could fake decodability)
  lat_<band>_d   Cohen's d (left vs right trials) of log power C3 - C4, MI 0.5-4 s:
                 the classic lateralised signal; lat_<band>_p is its Mann-Whitney p
  iaf_hz         individual alpha (mu) peak, 6-14 Hz, at C3/C4/Pz before MI onset
Writes results/summary/audit_liu2024_dev.csv. Descriptive only, not a decoder.

Usage: .venv/Scripts/python -I scripts/audit_liu2024.py
"""

import json

import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

from hemishift.data.liu2024_raw import SFREQ, load_raw, participants
from hemishift.paths import CONFIG_DIR, RESULTS_DIR

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


def audit(subject):
    raw = load_raw(subject)
    data = raw.get_data(picks="eeg")
    freqs, psd = welch(data, fs=SFREQ, nperseg=int(2 * SFREQ))
    hf = psd[:, (freqs >= 30) & (freqs < 40)].mean(1) / psd[:, (freqs >= 1) & (freqs < 40)].mean(1)
    line = psd[:, np.argmin(np.abs(freqs - 50))] / psd[:, (np.abs(freqs - 50) > 2) & (np.abs(freqs - 50) < 5)].mean(1)
    eeg_names = np.array(raw.copy().pick("eeg").ch_names)
    bad = eeg_names[(np.abs(robust_z(np.log(data.var(1)))) > 3.5) | (robust_z(hf) > 3.5)]

    filtered = raw.copy().filter(1.0, 40.0, verbose="error")
    events, event_id = mne.events_from_annotations(filtered, verbose="error")
    mi = mne.Epochs(filtered, events, event_id, tmin=0.5, tmax=4.0, baseline=None,
                    picks=["eeg", "eog"], preload=True, verbose="error")
    pre = mne.Epochs(filtered, events, event_id, tmin=-2.0, tmax=0.0, baseline=None,
                     picks="eeg", preload=True, verbose="error")
    is_left = mi.events[:, 2] == event_id["left_hand"]
    peaks = np.abs(mi.get_data(picks="eeg")).max(axis=(1, 2))

    row = {"subject": subject, "n_bad_channels": len(bad), "bad_channels": " ".join(bad),
           "line_noise": float(np.median(line)), "n_bad_trials": int((robust_z(peaks) > 3.5).sum())}
    eog = band_log_power(mi, ["HEOL", "VEOR"], (1.0, 40.0)).mean(1)
    row["eog_class_d"] = abs(cohens_d(eog[is_left], eog[~is_left]))
    for name, band in BANDS.items():
        power = band_log_power(mi, ["C3", "C4"], band)
        lateral = power[:, 0] - power[:, 1]
        row[f"lat_{name}_d"] = cohens_d(lateral[is_left], lateral[~is_left])
        row[f"lat_{name}_p"] = stats.mannwhitneyu(lateral[is_left], lateral[~is_left]).pvalue
    f, p = welch(pre.get_data(picks=["C3", "C4", "Pz"]).mean(axis=1), fs=SFREQ, nperseg=int(SFREQ))
    window = (f >= 6) & (f <= 14)
    row["iaf_hz"] = float(f[window][np.argmax(p.mean(0)[window])])
    return row


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    clinical = participants()
    rows = []
    for subject in split["dev"]:
        row = audit(subject)
        row.update(clinical.loc[subject, ["ParalysisSide", "NIHSS", "StrokeLocation"]].to_dict())
        rows.append(row)
        print(f"sub {subject:2d}: bad ch {row['n_bad_channels']} | bad trials {row['n_bad_trials']} | "
              f"EOG class d {row['eog_class_d']:.2f} | lateral d mu {row['lat_mu_d']:+.2f} "
              f"(p {row['lat_mu_p']:.3f}) beta {row['lat_beta_d']:+.2f} (p {row['lat_beta_p']:.3f}) | "
              f"IAF {row['iaf_hz']:.0f} Hz")
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / "audit_liu2024_dev.csv"
    table.to_csv(out, index=False)

    print(f"\n{len(table)} development patients (results in {out})")
    print(f"patients with any bad channel: {(table.n_bad_channels > 0).sum()} | median bad trials: "
          f"{table.n_bad_trials.median():.0f} | median line-noise ratio: {table.line_noise.median():.1f}")
    print(f"EOG differs by class (|d| > 0.5): {(table.eog_class_d > 0.5).sum()} patients")
    for band in BANDS:
        print(f"lateralised {band} signal at p < 0.05: {(table[f'lat_{band}_p'] < 0.05).sum()} patients "
              f"(expected by chance: {0.05 * len(table):.1f}); median |d| {table[f'lat_{band}_d'].abs().median():.2f}")
    print(f"individual mu peak: median {table.iaf_hz.median():.0f} Hz, range {table.iaf_hz.min():.0f}-{table.iaf_hz.max():.0f}")


if __name__ == "__main__":
    main()
