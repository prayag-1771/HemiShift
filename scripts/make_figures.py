"""Paper figures, regenerated from results/summary/*.csv (no number is typed by hand).

Writes paper/figures/fig{1..6}_*.pdf (and .png previews).
  fig1  S3 patient pooling: dev and pre-registered locked test (the confirmed result)
  fig2  S3 transfer strategies at k = 5 and 10 (deep, foundation model, Riemannian pooling)
  fig3  LesionSim rung 2: the 27 learned operator settings vs healthy and real-patient donors
  fig4  SafeTransfer gate: patients harmed and mean accuracy
  fig5  S1 left vs right: selection inflation, window ensemble, calibration screening
  fig6  Neurophysiology: ERD maps, donor-direction alignment, longitudinal decodability

Usage: .venv/Scripts/python -I scripts/make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hemishift.eval.stats import chance_threshold  # noqa: E402
from hemishift.paths import RESULTS_DIR, ROOT  # noqa: E402

SUMMARY = RESULTS_DIR / "summary"
OUT = ROOT / "paper" / "figures"
OWN, POOLED, HEALTHY, LESION, GENERIC, DEEP, PATIENT = (
    "#8c96a8", "#2563eb", "#d97706", "#7c3aed", "#9ca3af", "#0f766e", "#2563eb")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
})


def read(name):
    return pd.read_csv(SUMMARY / f"{name}.csv")


def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {name}")


def paired(ax, a, b, labels, colors, title):
    """Per-patient lines from a to b, with the group means."""
    for u, v in zip(a, b):
        ax.plot([0, 1], [u, v], color="#c7ccd4" if v >= u else "#f2b8b5", lw=0.9, zorder=1)
    for i, (values, color) in enumerate(((a, colors[0]), (b, colors[1]))):
        ax.scatter(np.full(len(values), i), values, s=10, color=color, zorder=2)
        ax.plot([i - 0.18, i + 0.18], [np.mean(values)] * 2, color=color, lw=2.4, zorder=3)
        ax.text(i + (0.22 if i else -0.22), np.mean(values), f"{np.mean(values):.1f}", va="center",
                ha="left" if i else "right", fontsize=7, color=color, fontweight="bold")
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-0.55, 1.55)
    ax.set_title(title)


def fig1():
    dev = read("s3_augment_dev")
    test = read("s3_locked_test")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    d5, t5 = dev[dev.k == 5], test[test.k == 5]
    paired(axes[0], d5["own"], d5["+patients"], ["own", "pooled"], (OWN, POOLED),
           f"a  Development, k = 5 (n = {len(d5)})")
    paired(axes[1], t5["own"], t5["pooled"], ["own", "pooled"], (OWN, POOLED),
           f"b  Locked test, k = 5 (n = {len(t5)})")
    axes[1].text(0.5, 0.02, "pre-registered, run once\none-sided Wilcoxon p = 0.035", transform=axes[1].transAxes,
                 ha="center", va="bottom", fontsize=6.5, color="#374151")
    for ax in axes[:2]:
        ax.set_ylabel("held-out accuracy (%)")
        ax.set_ylim(48, 101)
    ax = axes[2]
    for offset, (name, table, own_col, pooled_col) in enumerate(
            (("development", dev, "own", "+patients"), ("locked test", test, "own", "pooled"))):
        for color, col, shift in ((OWN, own_col, -0.06), (POOLED, pooled_col, 0.06)):
            means = [table[table.k == k][col].mean() for k in (5, 10)]
            cis = [boot_ci(table[table.k == k][col]) for k in (5, 10)]
            x = np.array([0, 1]) + shift + offset * 0.0
            ax.errorbar(x, means, yerr=[[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]],
                        color=color, marker="o" if offset == 0 else "s", ms=4, lw=1.2, capsize=2,
                        ls="-" if offset == 0 else "--", label=f"{'own' if color == OWN else 'pooled'}, {name}")
    ax.set_xticks([0, 1], ["k = 5", "k = 10"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("mean accuracy (%), 95 % CI")
    ax.set_title("c  Calibration pairs")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3)
    fig.tight_layout()
    save(fig, "fig1_pooling")


def fig2():
    transfer, fm, augment = read("s3_transfer_dev"), read("s3_foundation_dev"), read("s3_augment_dev")
    rows = [("own decoder (calibration only)", transfer, "own", OWN, "Own"),
            ("EEGNet from scratch", transfer, "scratch", DEEP, "Deep transfer (EEGNet)"),
            ("healthy-pretrained, zero-shot", transfer, "zeroshot", DEEP, None),
            ("healthy-pretrained, fine-tuned", transfer, "healthy", HEALTHY, None),
            ("  + LesionSim copy", transfer, "+lesionsim", LESION, None),
            ("  + generic copy (control)", transfer, "+generic", GENERIC, None),
            ("  + other patients", transfer, "+patients", PATIENT, None),
            ("CBraMod probe, channel-mean", fm, "fm-mean", DEEP, "Foundation model (frozen)"),
            ("CBraMod probe, per channel", fm, "fm-chan", DEEP, None),
            ("+ healthy donors", augment, "+healthy", HEALTHY, "Riemannian pooling"),
            ("+ LesionSim donors (population)", augment, "+lesionsim", LESION, None),
            ("+ LesionSim donors (patient-matched)", augment, "+matched", LESION, None),
            ("+ other patients' data", augment, "+patients", PATIENT, None)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True)
    positions, current = [], 0.0
    for row in rows:  # leave a gap above each group for its heading
        current += 1.6 if row[4] else 1.0
        positions.append(current)
    y = -np.array(positions)
    for ax, k in zip(axes, (5, 10)):
        own_mean = transfer[transfer.k == k]["own"].mean()
        ax.axvline(own_mean, color=OWN, lw=1, ls=":")
        for yi, (label, table, col, color, group) in zip(y, rows):
            values = table[table.k == k][col]
            lo, hi = boot_ci(values)
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.6)
            ax.scatter([values.mean()], [yi], color=color, s=18, zorder=3)
            ax.text(hi + 0.6, yi, f"{values.mean():.1f}", va="center", fontsize=6.5, color="#374151")
        ax.set_title(f"{'a' if k == 5 else 'b'}  k = {k} calibration pairs (n = 16)")
        ax.set_xlabel("held-out accuracy (%), mean and 95 % CI")
        ax.set_xlim(50, 95)
    axes[0].set_yticks(y, [r[0] for r in rows])
    for yi, (_, _, _, _, group) in zip(y, rows):
        if group:
            axes[0].text(-0.02, yi + 0.62, group, transform=axes[0].get_yaxis_transform(), ha="right",
                         va="center", fontsize=6.8, fontweight="bold", color="#111827")
    fig.tight_layout()
    save(fig, "fig2_transfer_strategies")


def fig3():
    grid = read("s3_theta_search_dev").set_index("subject")
    means = grid.mean()
    augment = read("s3_augment_dev")
    a5 = augment[augment.k == 5]
    refs = {"own": a5["own"].mean(), "healthy donors": a5["+healthy"].mean(), "real patients": a5["+patients"].mean()}
    alphas, betas, spreads = (1.0, 0.6, 0.2), (0.0, 0.4, 0.8), (0.0, 0.5, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.2), sharey=True)
    vmin, vmax = 77.0, 85.0
    for ax, spread in zip(axes, spreads):
        values = np.array([[means[f"{a}/{b}/{spread}"] for b in betas] for a in alphas])
        im = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if values[i, j] < 81 else "black")
        ax.set_xticks(range(3), [f"{b:g}" for b in betas])
        ax.set_yticks(range(3), [f"{a:g}" for a in alphas])
        ax.set_xlabel("β (shift to healthy side)")
        ax.set_title(f"spread = {spread:g}")
    axes[0].set_ylabel("α (ERD kept)")
    axes[0].add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec="#f97316", lw=1.6))  # unmodified donors
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("mean accuracy (%), k = 5, n = 16")
    cbar.ax.axhline(refs["own"], color="white", lw=1.6, ls="--")
    cbar.ax.axhline(refs["real patients"], color="#f97316", lw=1.8)
    fig.text(0.06, -0.06, f"Orange box: unmodified healthy donors ({refs['healthy donors']:.1f} %). Colour-bar marks: "
             f"own decoder {refs['own']:.1f} % (white, dashed); real other-patient donors {refs['real patients']:.1f} % "
             "(orange), above every setting.", fontsize=6.5, ha="left")
    save(fig, "fig3_lesionsim_settings")


def fig4():
    t = read("s3_safetransfer_dev")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    names = [("transfer", "always transfer", HEALTHY), ("ensemble", "always ensemble", GENERIC), ("gate", "SafeTransfer gate", POOLED)]
    width = 0.26
    for i, (col, label, color) in enumerate(names):
        harmed = [int((t[t.k == k][col] <= t[t.k == k]["own"] - 2).sum()) for k in (5, 10)]
        bars = axes[0].bar(np.arange(2) + (i - 1) * width, harmed, width, color=color, label=label)
        axes[0].bar_label(bars, fontsize=6.5, padding=1)
        means = [t[t.k == k][col].mean() for k in (5, 10)]
        axes[1].plot([5, 10], means, marker="o", color=color, label=label, lw=1.4, ms=4)
    axes[1].plot([5, 10], [t[t.k == k]["own"].mean() for k in (5, 10)], marker="o", color=OWN, label="always own", lw=1.4, ms=4, ls=":")
    axes[0].set_xticks([0, 1], ["k = 5", "k = 10"])
    axes[0].set_ylabel("patients harmed (≥ 2 points below own)")
    axes[0].set_title("a  Negative transfer (n = 16)")
    axes[0].set_ylim(0, 13)
    axes[1].set_xticks([5, 10], ["k = 5", "k = 10"])
    axes[1].set_ylabel("mean accuracy (%)")
    axes[1].set_title("b  Accuracy")
    axes[1].legend(frameon=False, fontsize=6.2)
    fig.tight_layout()
    save(fig, "fig4_safetransfer")


def fig5():
    nested, classical, locked = read("s1_nested_dev"), read("s1_classical_dev"), read("s1_locked_test")
    screen_dev, screen_test = read("s1_screening_dev"), read("s1_screening_locked_test")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1.25, 1.2]})
    ax = axes[0]
    for i, (col, label) in enumerate((("fixed", "fixed"), ("nested", "nested\n(honest)"), ("oracle", "best in\nhindsight"))):
        values = nested[col]
        ax.scatter(np.full(len(values), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(values)), values,
                   s=6, color="#9ca3af", zorder=1)
        ax.plot([i - 0.25, i + 0.25], [values.mean()] * 2, color="#111827" if col != "oracle" else "#dc2626", lw=2.2)
        ax.text(i, values.mean() + 2, f"{values.mean():.1f}", ha="center", fontsize=6.8, fontweight="bold")
    ax.set_xticks(range(3), ["fixed", "nested\n(honest)", "best in\nhindsight"])
    ax.axhline(100 * chance_threshold(39), color="#6b7280", lw=0.8, ls=":")
    ax.text(-0.5, 100 * chance_threshold(39) + 0.8, "chance threshold", fontsize=5.8, color="#6b7280", ha="left", va="bottom")
    ax.set_ylabel("accuracy (%), 34 dev patients")
    ax.set_title("a  Selection inflation")
    ax = axes[1]
    for x0, table, title in ((0, classical, "dev (34)"), (2.3, locked, "locked test (15)")):
        for u, v in zip(table["base"], table["ens16"]):
            ax.plot([x0, x0 + 1], [u, v], color="#c7ccd4" if v >= u else "#f2b8b5", lw=0.8)
        for dx, col, color in ((0, "base", OWN), (1, "ens16", POOLED)):
            ax.plot([x0 + dx - 0.2, x0 + dx + 0.2], [table[col].mean()] * 2, color=color, lw=2.4)
            ax.text(x0 + dx, table[col].mean() + 2.5, f"{table[col].mean():.1f}", ha="center", fontsize=6.8,
                    color=color, fontweight="bold")
        ax.text(x0 + 0.5, 101, title, ha="center", fontsize=7)
    ax.set_xticks([0, 1, 2.3, 3.3], ["base", "ens16", "base", "ens16"])
    ax.set_ylim(15, 106)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("b  Window ensemble")
    ax.text(0.5, 18, "p = 0.011", ha="center", fontsize=6.3, color="#374151")
    ax.text(2.8, 18, "p = 0.083 (one-sided,\npre-registered)", ha="center", fontsize=6.3, color="#374151")
    ax = axes[2]
    for table, marker, label in ((screen_dev, "o", "dev"), (screen_test, "^", "locked test")):
        best = table[["cal_base", "cal_ens16"]].max(axis=1)
        passed = table["choice"] != "abstain"
        jitter = np.random.default_rng(1).uniform(-1.2, 1.2, len(best))
        ax.scatter(best[~passed] + jitter[~passed], table["gate"][~passed], marker=marker, s=14, color="#9ca3af",
                   label=f"abstained, {label}")
        ax.scatter(best[passed] + jitter[passed], table["gate"][passed], marker=marker, s=18, color=POOLED,
                   label=f"passed, {label}")
    ax.axvline(75, color="#6b7280", lw=0.8, ls=":")
    ax.set_xlabel("calibration CV accuracy (%), first 20 trials")
    ax.set_ylabel("accuracy on later trials (%)")
    ax.set_title("c  Screening")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=5.8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_beijing")


def fig6():
    import mne

    from hemishift.data.channels import EXTENDED_TEMPLATE

    erd, align, longi = read("s3_neuro_erd"), read("s3_neuro_alignment"), read("s3_neuro_longitudinal")
    info = mne.create_info(list(EXTENDED_TEMPLATE), 128.0, "eeg")
    info.set_montage("standard_1005")
    fig = plt.figure(figsize=(7.2, 2.6))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.08, 1.15, 1.15], hspace=0.35, wspace=0.45)
    vmax = 30
    for row, band in enumerate(("mu", "beta")):
        for col, group in enumerate(("healthy", "stroke")):
            ax = fig.add_subplot(grid[row, col])
            values = 100 * erd[(erd.group == group) & (erd.band == band)][list(EXTENDED_TEMPLATE)].mean().to_numpy()
            im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax),
                                         contours=0, sensors=True)
            if row == 0:
                ax.set_title(f"{group}\n(n = {erd[(erd.group == group)].subject.nunique()})", fontsize=7.5)
            if col == 0:
                ax.text(-0.25, 0.5, band, transform=ax.transAxes, rotation=90, va="center", fontsize=7.5, fontweight="bold")
    cax = fig.add_subplot(grid[:, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("relative ERD (%), MI vs rest; lesion left", fontsize=6.5)
    ax = fig.add_subplot(grid[:, 3])
    paired(ax, align["cos_healthy"], align["cos_patients"], ["healthy\ndonors", "other\npatients"], (HEALTHY, PATIENT),
           "b  Direction alignment")
    ax.set_ylabel("cosine with the patient's own direction")
    ax = fig.add_subplot(grid[:, 4])
    wide = longi.pivot(index="subject", columns="phase", values="accuracy")[["pre", "post", "follow"]]
    for _, r in wide.iterrows():
        ax.plot(range(3), r.to_numpy(), color="#d1d5db", lw=0.8, marker="o", ms=2)
    ax.plot(range(3), wide.mean().to_numpy(), color="#111827", lw=2.2, marker="o", ms=4)
    for i, m in enumerate(wide.mean()):
        ax.text(i, m + 3, f"{m:.1f}", ha="center", fontsize=6.8, fontweight="bold")
    ax.set_xticks(range(3), ["pre", "post", "follow-up"])
    ax.set_ylabel("own decoder accuracy (%)")
    ax.set_title("c  Across the study period")
    fig.text(0.02, 0.97, "a  ERD maps", fontsize=8.5, fontweight="bold", va="top")
    save(fig, "fig6_neuro")


def main():
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    if (SUMMARY / "s3_neuro_longitudinal.csv").exists():
        fig6()


if __name__ == "__main__":
    main()
