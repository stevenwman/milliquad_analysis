"""Plot pitch + x-position time series for all selected step trials.

Shows windowing cutoffs for both methods:
  - Spatial gate (current sim): full window from STEP_START_X to 90% STEP_END_X
  - q60-on-gated (experimental-style): spatially gate first, then [0.45*n_gated : 0.75*n_gated]

Left panel: pitch (deg) vs time
Right panel: pos_x (mm) vs time
Shading + lines mark the window boundaries.

Usage:
    uv run python -m analysis.temp.plot_pitch_windows
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NPZ_PATH = Path(__file__).resolve().parent.parent.parent / (
    "results/20260228T230022_step_q60_rk-warm/20260302T181206_validation_trajectories.npz"
)
CSV_PATH = NPZ_PATH.parent / "20260302T181206_validation_trials.csv"
OUT_DIR = Path(__file__).resolve().parent

# Step geometry
STEP_START_X = 0.05
STEP_END_X = 0.05 + 7 * 0.0045 + 0.02  # 0.1015
CUTOFF_X = STEP_START_X + 0.9 * (STEP_END_X - STEP_START_X)
SIM_DT = 1.0 / 2000.0


def main():
    d = np.load(str(NPZ_PATH), allow_pickle=True)

    # Get selected trials from CSV
    selected = set()
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["selected"] == "True":
                selected.add(f"{row['ref_id']}_t{row['trial']}")

    # Group by ref_id
    from collections import defaultdict
    refs = defaultdict(list)
    for key in sorted(selected):
        if f"{key}_pitch" not in d:
            continue
        rid = "_".join(key.split("_")[:-1])  # e.g. scene1_f10
        refs[rid].append(key)

    for rid, trials in sorted(refs.items()):
        n_trials = len(trials)
        fig, axes = plt.subplots(n_trials, 2, figsize=(16, 3.5 * n_trials), squeeze=False)
        fig.suptitle(rid, fontsize=16, fontweight="bold")

        for i, tkey in enumerate(trials):
            pitch = d[f"{tkey}_pitch"]
            pos_x = d[f"{tkey}_pos_x"] * 1000  # mm
            time = d[f"{tkey}_time"]
            n = len(pitch)

            ax_p, ax_x = axes[i]

            # --- Spatial gate indices ---
            enter_idx = int(np.searchsorted(d[f"{tkey}_pos_x"], STEP_START_X))
            exit_indices = np.where(d[f"{tkey}_pos_x"] >= CUTOFF_X)[0]
            exit_idx = int(exit_indices[0]) if len(exit_indices) else n - 1

            # --- q60 on spatially-gated subset ---
            n_gated = exit_idx - enter_idx + 1
            lo_q60 = enter_idx + int(0.45 * n_gated)
            hi_q60 = enter_idx + int(0.75 * n_gated)

            # Pitch panel
            ax_p.plot(time, pitch, "k-", linewidth=0.5, alpha=0.7)
            ax_p.axvspan(time[enter_idx], time[min(exit_idx, n-1)],
                         color="tab:blue", alpha=0.10, label="spatial gate (full)")
            ax_p.axvspan(time[lo_q60], time[min(hi_q60, n-1)],
                         color="tab:red", alpha=0.20, label="q60 on gated")
            ax_p.axvline(time[enter_idx], color="tab:blue", ls="--", lw=1)
            ax_p.axvline(time[min(exit_idx, n-1)], color="tab:blue", ls="--", lw=1)
            ax_p.axvline(time[lo_q60], color="tab:red", ls=":", lw=1.2)
            ax_p.axvline(time[min(hi_q60, n-1)], color="tab:red", ls=":", lw=1.2)
            ax_p.set_ylabel("Pitch (deg)")
            ax_p.set_title(f"{tkey}  |  pitch", fontsize=10)
            if i == 0:
                ax_p.legend(fontsize=8, loc="upper right")

            # Pos_x panel
            ax_x.plot(time, pos_x, "k-", linewidth=0.5, alpha=0.7)
            ax_x.axhline(STEP_START_X * 1000, color="tab:blue", ls="--", lw=1, label="step_start_x")
            ax_x.axhline(CUTOFF_X * 1000, color="tab:blue", ls="-", lw=1, label="90% step_end_x")
            ax_x.axvline(time[enter_idx], color="tab:blue", ls="--", lw=0.8, alpha=0.5)
            ax_x.axvline(time[min(exit_idx, n-1)], color="tab:blue", ls="--", lw=0.8, alpha=0.5)
            ax_x.axvline(time[lo_q60], color="tab:red", ls=":", lw=1, alpha=0.7)
            ax_x.axvline(time[min(hi_q60, n-1)], color="tab:red", ls=":", lw=1, alpha=0.7)
            ax_x.set_ylabel("pos_x (mm)")
            ax_x.set_title(f"{tkey}  |  pos_x", fontsize=10)
            if i == 0:
                ax_x.legend(fontsize=8, loc="upper left")

            for ax in (ax_p, ax_x):
                ax.set_xlabel("Time (s)")
                ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out = OUT_DIR / f"pitch_windows_{rid}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
