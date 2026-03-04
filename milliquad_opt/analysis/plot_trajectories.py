"""Plot x-position vs time for all validation trials, grouped by ref_id.

Called automatically by validate_params.py when --csv is used,
or standalone:
    uv run python -m analysis.plot_trajectories results/20260228T202903_rough_spatial_rk4
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETTLE_TIME = 0.3  # seconds
NCOLS = 4

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "TeX Gyre Pagella",
    "font.size": 10,
})

TRIAL_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
COLOR_UNSELECTED = "#bbbbbb"


def plot_trajectory_overview(
    run_dir: Path,
    step_start_x: float | None = None,
    step_end_x: float | None = None,
    npz_path: Path | None = None,
    csv_path: Path | None = None,
) -> Path:
    """Generate trajectory overview plot. Returns output path."""
    npz_path = npz_path or run_dir / "validation_trajectories.npz"
    csv_path = csv_path or run_dir / "validation_trials.csv"
    out_path = run_dir / "trajectory_overview.png"

    traj = np.load(npz_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Build lookup: (ref_id, trial) -> row
    row_lookup = {}
    for r in rows:
        row_lookup[(r["ref_id"], int(r["trial"]))] = r

    # Ordered ref_ids (natural sort: scene1 < scene2 < scene4 < scene_wheel,
    # then by frequency)
    ref_ids = sorted(
        set(r["ref_id"] for r in rows),
        key=lambda s: (
            0 if "scene1" in s else 1 if "scene2" in s else 2 if "scene4" in s else 3,
            int(s.split("_f")[1]),
        ),
    )

    nrefs = len(ref_ids)
    nrows = (nrefs + NCOLS - 1) // NCOLS

    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(NCOLS * 4.2, nrows * 3.0),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, ref_id in enumerate(ref_ids):
        ax = axes[idx // NCOLS][idx % NCOLS]
        trial_rows = sorted(
            [r for r in rows if r["ref_id"] == ref_id],
            key=lambda r: int(r["trial"]),
        )

        # Plot non-selected first (behind), then selected on top
        for is_sel_pass in (False, True):
            for tr in trial_rows:
                trial_idx = int(tr["trial"])
                selected = tr["selected"] == "True"
                if selected != is_sel_pass:
                    continue

                key_time = f"{ref_id}_t{trial_idx}_time"
                key_pos = f"{ref_id}_t{trial_idx}_pos_x"
                if key_time not in traj:
                    continue
                t = traj[key_time]
                x_mm = traj[key_pos] * 1000.0  # m -> mm

                if selected:
                    color = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
                    lw, alpha, zorder = 1.4, 1.0, 3
                else:
                    color = COLOR_UNSELECTED
                    lw, alpha, zorder = 0.8, 0.4, 2
                ax.plot(
                    t, x_mm,
                    color=color, lw=lw, alpha=alpha, zorder=zorder,
                    label=f"t{trial_idx}",
                )

        # Settle time marker
        ax.axvline(SETTLE_TIME, color="k", ls="--", lw=0.8, alpha=0.6, zorder=1)

        # Spatial gate bounds (horizontal lines on x-position axis)
        if step_start_x is not None:
            ax.axhline(step_start_x * 1000, color="tab:blue", ls=":", lw=1.0,
                        alpha=0.7, zorder=1, label="gate start")
        if step_end_x is not None:
            ax.axhline(step_end_x * 1000, color="tab:red", ls=":", lw=1.0,
                        alpha=0.7, zorder=1, label="gate end")

        # Annotate min_window_vx for selected trials
        sel_annotations = []
        for tr in trial_rows:
            if tr["selected"] == "True":
                mwv_str = tr.get("min_window_vx", "")
                if mwv_str:
                    vx_mw = float(mwv_str) * 1000.0  # m/s -> mm/s
                    sel_annotations.append(f"t{tr['trial']}: {vx_mw:.1f} mm/s")
        if sel_annotations:
            annotation_text = "\n".join(sel_annotations)
            ax.text(
                0.97, 0.03, annotation_text,
                transform=ax.transAxes, fontsize=7.5,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
            )

        ax.set_title(ref_id, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("x position (mm)")
        ax.legend(fontsize=6.5, loc="upper left", ncol=2, framealpha=0.7)

    # Hide unused subplots
    for idx in range(nrefs, nrows * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Trajectory overview: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m analysis.plot_trajectories <run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    # Auto-detect spatial bounds from terrain type
    step_start_x = None
    step_end_x = None
    name = run_dir.name
    if "rough" in name:
        from config_rough import FLAT_LEAD, _X_HALF
        step_start_x = FLAT_LEAD
        step_end_x = FLAT_LEAD + 2 * _X_HALF
    elif "step" in name:
        import importlib
        config_mod = importlib.import_module("config_step")
        step_start_x = getattr(config_mod, "STEP_START_X", None)
        step_end_x = getattr(config_mod, "STEP_END_X", None)
    plot_trajectory_overview(run_dir, step_start_x, step_end_x)
