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

SETTLE_TIME = 0.1  # seconds — must match config.SETTLE_TIME
NCOLS = 4

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "TeX Gyre Pagella",
    "font.size": 10,
})

TRIAL_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]


def _compute_avg_vx(time_arr, pos_x_arr, settle_time=SETTLE_TIME):
    """Compute average forward velocity (mm/s) after settle time."""
    mask = time_arr >= settle_time
    if mask.sum() < 2:
        return 0.0
    t_start = time_arr[mask][0]
    t_end = time_arr[mask][-1]
    x_start = pos_x_arr[mask][0]
    x_end = pos_x_arr[mask][-1]
    dt = t_end - t_start
    if dt < 1e-6:
        return 0.0
    return (x_end - x_start) / dt * 1000.0  # m/s -> mm/s


def plot_trajectory_overview(
    run_dir: Path,
    step_start_x: float | None = None,
    step_end_x: float | None = None,
    npz_path: Path | None = None,
    csv_path: Path | None = None,
    trial_duration_map: dict[str, float] | None = None,
) -> Path:
    """Generate trajectory overview plot. Returns output path."""
    if npz_path is None:
        candidates = sorted(run_dir.glob("*validation_trajectories.npz"))
        npz_path = candidates[-1] if candidates else run_dir / "validation_trajectories.npz"
    if csv_path is None:
        candidates = sorted(run_dir.glob("*validation_trials.csv"))
        csv_path = candidates[-1] if candidates else run_dir / "validation_trials.csv"
    out_path = run_dir / "trajectory_overview.png"

    traj = np.load(npz_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

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

        for tr in trial_rows:
            trial_idx = int(tr["trial"])
            crash = tr.get("crash", "False") == "True"

            key_time = f"{ref_id}_t{trial_idx}_time"
            key_pos = f"{ref_id}_t{trial_idx}_pos_x"
            if key_time not in traj:
                continue

            t = traj[key_time]
            x_mm = traj[key_pos] * 1000.0  # m -> mm

            color = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
            avg_vx = _compute_avg_vx(t, traj[key_pos])
            label = f"t{trial_idx}: {avg_vx:.0f} mm/s"

            ax.plot(t, x_mm, color=color, lw=1.4, alpha=1.0, zorder=3, label=label)

        # Settle time marker
        ax.axvline(SETTLE_TIME, color="k", ls="--", lw=0.8, alpha=0.6, zorder=1)

        # Spatial gate bounds (step/rough: horizontal lines on x-position axis)
        if step_start_x is not None:
            ax.axhline(step_start_x * 1000, color="tab:blue", ls=":", lw=1.0,
                        alpha=0.7, zorder=1, label="gate start")
        if step_end_x is not None:
            ax.axhline(step_end_x * 1000, color="tab:red", ls=":", lw=1.0,
                        alpha=0.7, zorder=1, label="gate end")

        # Time gate (flat_tg: vertical line at SETTLE_TIME + trial_duration)
        if trial_duration_map and ref_id in trial_duration_map:
            t_gate = SETTLE_TIME + trial_duration_map[ref_id]
            ax.axvline(t_gate, color="tab:red", ls=":", lw=1.0,
                        alpha=0.7, zorder=1, label=f"gate {t_gate:.2f}s")

        ax.set_title(ref_id, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("x position (mm)")
        ax.legend(fontsize=6.5, loc="upper left", ncol=1, framealpha=0.7)

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

    # Auto-detect time gate from saved config in run dir
    trial_duration_map: dict[str, float] = {}
    saved_config = run_dir / "config_flat_tg.py"
    if not saved_config.exists():
        saved_config = run_dir / "config_flat_tg_no20.py"
    if saved_config.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("_cfg", saved_config)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        from config import _make_ref_id
        for rd in cfg.REFERENCE_DATA:
            if "trial_duration" in rd:
                rid = _make_ref_id(rd["scene"], rd["ctrl_freq"])
                trial_duration_map[rid] = rd["trial_duration"]

    plot_trajectory_overview(run_dir, step_start_x, step_end_x,
                             trial_duration_map=trial_duration_map or None)
