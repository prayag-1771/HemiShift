"""S3 cross-session transfer: the patient's own earlier session as a donor (dev, exploratory).

In rehabilitation a patient comes back for many sessions, so from the second session on the
patient's own earlier recording is available. The target is a later session (post, or
follow-up), calibrated on its first k (rest, MI) pairs as in P3 and tested on the rest.
Every decoder uses the confirmed pooled code (14 channels, 8-30 Hz, each domain re-centred on
its own mean, tangent space at the identity, LR):
  own              the later session's calibration only
  others           + the other dev patients' recordings of the same phase (the confirmed setup)
  previous         + the patient's own earlier session(s) instead
  previous+others  + both; each donor group together weighs as much as the calibration
  previous_zero    no labels in the later session: the earlier session's decoder, with the
                   later session re-centred on its first 2k epochs
Results: results/summary/s3_cross_session_dev.csv.

Usage: .venv/Scripts/python -I scripts/s3_cross_session.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.linear_model import LogisticRegression

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.eval.stats import holm, paired_comparison
from hemishift.paths import RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _script("s3_transfer")
augment = _script("s3_augment")
sweep = _script("s3_sweep")
TARGETS = (("post", ("pre",)), ("follow", ("pre", "post")))
OUT = RESULTS_DIR / "summary" / "s3_cross_session_dev.csv"


def pooled_groups(C_cal, y_cal, C_test, groups):
    """Pooled TS + LR where each donor group (re-centred covariances, labels) together weighs
    as much as the calibration. With one group this is s3_augment.pooled_proba."""
    W = invsqrtm(mean_riemann(C_cal))
    identity = np.eye(C_cal.shape[-1])
    X = tangent_space(np.concatenate([W @ C_cal @ W] + [C for C, _ in groups]), identity,
                      metric="riemann")
    y = np.concatenate([y_cal] + [labels for _, labels in groups])
    w = np.concatenate([np.ones(len(y_cal))]
                       + [np.full(len(labels), len(y_cal) / len(labels)) for _, labels in groups])
    model = LogisticRegression(C=1.0, max_iter=1000).fit(X, y, sample_weight=w)
    return model.predict_proba(tangent_space(W @ C_test @ W, identity, metric="riemann"))[:, 1]


def donor_only(C_ref, C_test, donor_C, donor_y):
    """Decoder trained on donors only; the target is re-centred on C_ref (labels unused)."""
    W = invsqrtm(mean_riemann(C_ref))
    identity = np.eye(C_ref.shape[-1])
    model = LogisticRegression(C=1.0, max_iter=1000).fit(
        tangent_space(donor_C, identity, metric="riemann"), donor_y)
    return model.predict_proba(tangent_space(W @ C_test @ W, identity, metric="riemann"))[:, 1]


def main():
    sessions = {phase: sweep.load("dev", phase, EXTENDED_TEMPLATE, sweep.WINDOWS["std"])
                for phase in ("pre", "post", "follow")}
    C = {phase: {s: s3.premise.covariances(X) for s, (X, _) in data.items()}
         for phase, data in sessions.items()}
    centred = {phase: {s: augment.recentred_by_domain(c, np.zeros(len(c))) for s, c in data.items()}
               for phase, data in C.items()}

    rows = []
    for target, previous in TARGETS:
        for s, (X, y) in sessions[target].items():
            if any(s not in sessions[p] for p in previous):
                continue
            prev = (np.concatenate([centred[p][s] for p in previous]),
                    np.concatenate([sessions[p][s][1] for p in previous]))
            others = [o for o in sessions[target] if o != s]
            oth = (np.concatenate([centred[target][o] for o in others]),
                   np.concatenate([sessions[target][o][1] for o in others]))
            Cs = C[target][s]
            for k in s3.KS:
                cal, test = s3.calibration_split(len(y), k)
                probs = {
                    "own": s3.own_proba(X, y, cal, test),
                    "others": augment.pooled_proba(Cs[cal], y[cal], Cs[test], *oth),
                    "previous": pooled_groups(Cs[cal], y[cal], Cs[test], [prev]),
                    "previous+others": pooled_groups(Cs[cal], y[cal], Cs[test], [prev, oth]),
                    "previous_zero": donor_only(Cs[cal], Cs[test], *prev),
                }
                rows.append({"target": target, "subject": s, "k": k,
                             **{n: 100 * np.mean((p > 0.5).astype(int) == y[test])
                                for n, p in probs.items()}})
            print(f"{target} sub {s:2d} k=5: " + " ".join(
                f"{n} {v:.1f}" for n, v in rows[-2].items() if n not in ("target", "subject", "k")),
                flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    names = ["others", "previous", "previous+others", "previous_zero"]
    for target, _ in TARGETS:
        for k in s3.KS:
            t = table[(table["target"] == target) & (table["k"] == k)]
            print(f"\n{target}, k = {k} ({len(t)} patients): own {t['own'].mean():.1f} %")
            tests = {n: paired_comparison(t[n] / 100, t["own"] / 100) for n in names}
            adjusted = dict(zip(names, holm([tests[n]["p"] for n in names])))
            for n in names:
                r = tests[n]
                line = (f"  {n:16s} {t[n].mean():5.1f} % | vs own median {100 * r['median_diff']:+.1f}, "
                        f"p {r['p']:.3f} (Holm {adjusted[n]:.3f}), helped {(t[n] > t['own']).sum()}, "
                        f"harmed {(t[n] < t['own']).sum()}")
                if n != "others":
                    o = paired_comparison(t[n] / 100, t["others"] / 100)
                    line += f" | vs others median {100 * o['median_diff']:+.1f}, p {o['p']:.3f}"
                print(line)
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
