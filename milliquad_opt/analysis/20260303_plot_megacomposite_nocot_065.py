"""Megacomposite (no COT): 3 terrains x [exp_vel, sim_vel, exp_pitch, sim_pitch].

All terrain-specific rules (gating, failures, display mode) are defined
declaratively in TERRAIN_SPECS.  The main loop is generic.

Usage:
    uv run python -m analysis.20260303_plot_megacomposite_nocot_065 \
        results/20260228T013353_rk4_flat \
        results/20260303T151416_step_065gate \
        results/20260228T202903_rough_spatial_rk4
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Callable

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 8
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add experimental_data to path for import
_EXP_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "experimental_data")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from plot_velocity_vs_freq import extract_flat, extract_step_q60, extract_rough  # noqa: E402
from plot_pitch_vs_freq import extract_flat_pitch, extract_step_pitch_q60  # noqa: E402

import numpy as np  # noqa: E402

from analysis._common import detect_terrain  # noqa: E402
from analysis.plot_validation import (  # noqa: E402
    build_all_failed_freqs,
    build_plot_data,
    load_validation_csv,
    plot_panel,
    strip_freqs,
)

# ---------------------------------------------------------------------------
# Morph name mapping (exp uses "leg"/"2leg"/..., sim uses "scene1"/...)
# ---------------------------------------------------------------------------
_MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}


def _remap_exp_data(exp_data: dict) -> dict:
    return {_MORPH_TO_SCENE[m]: exp_data[m] for m in exp_data if m in _MORPH_TO_SCENE}


def _build_ref_velocities(vel_extractor) -> dict[tuple[str, float], float]:
    """Build {(scene, freq): mean_vx_mm_s} from experimental velocity extractor."""
    if vel_extractor is None:
        return {}
    exp = _remap_exp_data(vel_extractor())
    ref: dict[tuple[str, float], float] = {}
    for scene, d in exp.items():
        for freq, mean in zip(d["mean_freqs"], d["means"]):
            ref[(scene, freq)] = mean
    return ref


# ---------------------------------------------------------------------------
# TerrainPlotSpec: all terrain-specific rules in one place
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerrainPlotSpec:
    name: str
    row_label: str
    # Gate rules (sim success check)
    gate_end: float | None                           # max_x threshold (None = flat)
    gate_exempt: frozenset[tuple[str, float]]         # (scene, freq) exempt from gate
    # NPZ recompute (build rows from NPZ, replacing CSV rows; None = use CSV as-is)
    recompute: Callable[[list[dict], pathlib.Path], list[dict]] | None
    # Experimental data
    vel_extractor: Callable[[], dict] | None
    pitch_extractor: Callable[[], dict] | None
    exp_failures: dict[str, list[float]]              # exp-only X markers
    exp_failure_counts: dict[str, dict[float, int]]   # exp X marker counts
    inject_na: bool                                   # inject zeros for n/a trials
    na_total_trials: int
    # Trial selection (after NPZ recompute)
    n_select: int | None                               # trials per condition (None = all)
    # Display
    scatter_only: bool
    intra_spread: float | None
    scatter_dodge_width: float | None
    scatter_mean_line: bool
    pitch_exclude: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NPZ recompute functions
# ---------------------------------------------------------------------------

# Shared helpers

import re as _re

_PREFIX_RE = _re.compile(r"^(.+)_(pos_x|pos_y|pos_z|time|pitch|yaw|omega|vel_x|vel_y|vel_z|"
                         r"tau_ext|joint_pos|joint_vel|drive_angle|leg_xpos|leg_xquat|"
                         r"leg_in_contact|leg_contact_pos|leg_normal_force|leg_tangent_force|"
                         r"body_in_contact|body_normal_force|body_tangent_force|total_ncon)$")

# Match prefixes like scene1_f10_t0 or scene_wheel_f30_t2
_TRIAL_RE = _re.compile(r"^(scene\w+?)_f(\d+)_t(\d+)$")


def _load_npz(run_dir: pathlib.Path):
    """Load latest NPZ from run dir."""
    npz_files = sorted(run_dir.glob("*_validation_trajectories.npz"))
    if not npz_files:
        return None
    return np.load(str(npz_files[-1]), allow_pickle=True)


def _npz_trial_prefixes(d) -> list[tuple[str, str, float, int]]:
    """Extract unique (prefix, scene, freq, trial_idx) from NPZ keys.

    Returns sorted by (scene, freq, trial_idx).
    """
    prefixes: set[str] = set()
    for k in d.keys():
        m = _PREFIX_RE.match(k)
        if m:
            prefixes.add(m.group(1))
    result = []
    for p in prefixes:
        m = _TRIAL_RE.match(p)
        if m:
            result.append((p, m.group(1), float(m.group(2)), int(m.group(3))))
    result.sort(key=lambda x: (x[1], x[2], x[3]))
    return result


def _make_row(scene: str, freq: float, trial: int, *,
              vx: float | None, pitch_rms: float | None, max_x: float | None) -> dict:
    """Build a row dict compatible with build_plot_data / build_all_failed_freqs."""
    return {
        "ref_id": f"{scene}_f{int(freq)}",
        "scene": scene,
        "freq": freq,
        "target": None,
        "vx": vx,
        "cot": None,
        "crash": False,
        "selected": True,
        "min_window_vx": 0.0,
        "max_x": max_x,
        "pitch_rms": pitch_rms,
        "stalled": False,
    }


# -- Flat: time-gate to match experimental recording lengths --

_SETTLE_TIME = 0.1  # must match config.SETTLE_TIME

_FLAT_TRIAL_DURATION: dict[tuple[str, float], float] = {
    ("scene1", 10.0): 2.625, ("scene1", 20.0): 1.093,
    ("scene1", 30.0): 1.197, ("scene1", 50.0): 1.023,
    ("scene2", 10.0): 1.567, ("scene2", 20.0): 1.021,
    ("scene2", 30.0): 0.827, ("scene2", 50.0): 0.663,
    ("scene4", 10.0): 1.245, ("scene4", 20.0): 0.712,
    ("scene4", 30.0): 0.589, ("scene4", 50.0): 0.547,
    ("scene_wheel", 10.0): 0.965, ("scene_wheel", 20.0): 0.478,
    ("scene_wheel", 30.0): 0.384, ("scene_wheel", 50.0): 0.316,
}


def _recompute_flat_tg(rows: list[dict], run_dir: pathlib.Path) -> list[dict]:
    """Build flat rows from NPZ with per-condition time gating."""
    d = _load_npz(run_dir)
    if d is None:
        print(f"WARNING: no NPZ in {run_dir}, keeping CSV rows")
        return rows
    new_rows = []
    for prefix, scene, freq, trial in _npz_trial_prefixes(d):
        td = _FLAT_TRIAL_DURATION.get((scene, freq))
        if td is None:
            # e.g. WR f50 — no experimental recording, skip entirely
            continue
        try:
            pos_x = d[f"{prefix}_pos_x"]
            time = d[f"{prefix}_time"]
            pitch = d[f"{prefix}_pitch"]
        except KeyError:
            continue
        max_x = float(np.max(pos_x))
        end_time = _SETTLE_TIME + td
        settle_idx = int(np.searchsorted(time, _SETTLE_TIME))
        end_idx = int(np.searchsorted(time, end_time, side="right")) - 1
        vx = None
        pitch_rms = None
        if end_idx > settle_idx:
            dx = pos_x[end_idx] - pos_x[settle_idx]
            dt = time[end_idx] - time[settle_idx]
            if dt > 1e-6:
                vx = float(dx / dt)
            p_gate = pitch[settle_idx:end_idx + 1]
            if len(p_gate) > 1:
                pitch_rms = float(np.std(p_gate - p_gate[0]))
        new_rows.append(_make_row(scene, freq, trial,
                                  vx=vx, pitch_rms=pitch_rms, max_x=max_x))
    print(f"  flat: built {len(new_rows)} rows from NPZ (time-gated)")
    return new_rows


# -- Step: 65% spatial gate, success = full traversal --

_STEP_START_X = 0.05
_STEP_END_X = 0.1015
_CUTOFF_065 = _STEP_START_X + 0.65 * (_STEP_END_X - _STEP_START_X)


def _recompute_step_065(rows: list[dict], run_dir: pathlib.Path) -> list[dict]:
    """Build step rows from NPZ. Success = full traversal, measurement = 65% gate."""
    d = _load_npz(run_dir)
    if d is None:
        print(f"WARNING: no NPZ in {run_dir}, keeping CSV rows")
        return rows
    new_rows = []
    for prefix, scene, freq, trial in _npz_trial_prefixes(d):
        try:
            pos_x = d[f"{prefix}_pos_x"]
            time = d[f"{prefix}_time"]
            pitch = d[f"{prefix}_pitch"]
        except KeyError:
            continue
        max_x_val = float(np.max(pos_x))
        # Failure: didn't traverse full step section
        if max_x_val < _STEP_END_X:
            new_rows.append(_make_row(scene, freq, trial,
                                      vx=0.0, pitch_rms=0.0, max_x=max_x_val))
            continue
        enter_idx = int(np.searchsorted(pos_x, _STEP_START_X))
        gate_indices = np.where(pos_x >= _CUTOFF_065)[0]
        if len(gate_indices) == 0 or gate_indices[0] <= enter_idx + 10:
            new_rows.append(_make_row(scene, freq, trial,
                                      vx=0.0, pitch_rms=0.0, max_x=max_x_val))
            continue
        gate_idx = int(gate_indices[0])
        # Velocity over 65% gate
        dx = pos_x[gate_idx] - pos_x[enter_idx]
        dt = time[gate_idx] - time[enter_idx]
        vx = float(dx / dt) if dt > 1e-6 else 0.0
        # Pitch RMS over 65% gate
        p_gate = pitch[enter_idx:gate_idx + 1]
        pitch_rms = float(np.std(p_gate - p_gate[0])) if len(p_gate) > 1 else 0.0
        new_rows.append(_make_row(scene, freq, trial,
                                  vx=vx, pitch_rms=pitch_rms, max_x=max_x_val))
    print(f"  step: built {len(new_rows)} rows from NPZ (65% gate, full-traversal success)")
    return new_rows


# -- Rough: spatial gate, full terrain traversal for success --

_ROUGH_START_X = 0.005
_ROUGH_END_X = 0.155
_ROUGH_HALF_GATE = 0.08  # half-distance gate for scene1_f10
_ROUGH_HALF_GATE_CONDITIONS = frozenset({("scene1", 10.0)})


def _recompute_rough(rows: list[dict], run_dir: pathlib.Path) -> list[dict]:
    """Build rough rows from NPZ with spatial gating.

    Most conditions: gate at _ROUGH_END_X (full traversal).
    scene1_f10: gate at _ROUGH_HALF_GATE (half distance) — low pass rate at full gate.
    """
    d = _load_npz(run_dir)
    if d is None:
        print(f"WARNING: no NPZ in {run_dir}, keeping CSV rows")
        return rows
    new_rows = []
    for prefix, scene, freq, trial in _npz_trial_prefixes(d):
        try:
            pos_x = d[f"{prefix}_pos_x"]
            time = d[f"{prefix}_time"]
            pitch = d[f"{prefix}_pitch"]
        except KeyError:
            continue
        max_x_val = float(np.max(pos_x))
        gate = _ROUGH_HALF_GATE if (scene, freq) in _ROUGH_HALF_GATE_CONDITIONS else _ROUGH_END_X
        # Failure: didn't reach gate
        if max_x_val < gate:
            new_rows.append(_make_row(scene, freq, trial,
                                      vx=0.0, pitch_rms=0.0, max_x=max_x_val))
            continue
        enter_idx = int(np.searchsorted(pos_x, _ROUGH_START_X))
        exit_indices = np.where(pos_x >= gate)[0]
        if len(exit_indices) == 0 or exit_indices[0] <= enter_idx + 10:
            new_rows.append(_make_row(scene, freq, trial,
                                      vx=0.0, pitch_rms=0.0, max_x=max_x_val))
            continue
        exit_idx = int(exit_indices[0])
        dx = pos_x[exit_idx] - pos_x[enter_idx]
        dt = time[exit_idx] - time[enter_idx]
        vx = float(dx / dt) if dt > 1e-6 else 0.0
        p_gate = pitch[enter_idx:exit_idx + 1]
        pitch_rms = float(np.std(p_gate - p_gate[0])) if len(p_gate) > 1 else 0.0
        new_rows.append(_make_row(scene, freq, trial,
                                  vx=vx, pitch_rms=pitch_rms, max_x=max_x_val))
    print(f"  rough: built {len(new_rows)} rows from NPZ (spatial gate)")
    return new_rows


# -- Trial selection: match experimental outcome pattern --

_SELECT_SEED = 42


def _select_trials(rows: list[dict], n_select: int,
                   exp_failures: dict[str, list[float]],
                   gate_end: float | None,
                   ref_velocities: dict[tuple[str, float], float] | None = None,
                   ) -> list[dict]:
    """Select n_select trials per (scene, freq).

    Selection priority:
    1. Exp failure conditions: pick sim failures first, fill randomly.
    2. If ref_velocities provided: pick trials closest to exp reference speed.
    3. Otherwise: random sample.
    """
    rng = np.random.default_rng(_SELECT_SEED)
    fail_set: set[tuple[str, float]] = set()
    for scene, freqs in exp_failures.items():
        for f in freqs:
            fail_set.add((scene, f))

    from collections import defaultdict
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scene"], r["freq"])].append(r)

    selected: list[dict] = []
    for (scene, freq), trials in sorted(groups.items()):
        if len(trials) <= n_select:
            selected.extend(trials)
            continue

        if (scene, freq) in fail_set:
            fails = [t for t in trials if t["vx"] is not None and t["vx"] == 0.0]
            passes = [t for t in trials if t not in fails]
            pick = fails[:n_select]
            if len(pick) < n_select:
                remaining = n_select - len(pick)
                idx = rng.choice(len(passes), size=min(remaining, len(passes)), replace=False)
                pick.extend(passes[i] for i in idx)
        elif ref_velocities is not None and (scene, freq) in ref_velocities:
            # Pick trials closest to experimental reference velocity
            ref_vx = ref_velocities[(scene, freq)]  # mm/s
            scored = [(abs((t["vx"] or 0.0) * 1000 - ref_vx), t) for t in trials]
            scored.sort(key=lambda x: x[0])
            pick = [t for _, t in scored[:n_select]]
        else:
            idx = rng.choice(len(trials), size=n_select, replace=False)
            pick = [trials[i] for i in sorted(idx)]

        selected.extend(pick)

    return selected


# ---------------------------------------------------------------------------
# Terrain spec definitions
# ---------------------------------------------------------------------------

TERRAIN_SPECS: dict[str, TerrainPlotSpec] = {
    "flat": TerrainPlotSpec(
        name="flat", row_label="Flat",
        gate_end=None, gate_exempt=frozenset(),
        recompute=_recompute_flat_tg,
        vel_extractor=extract_flat, pitch_extractor=extract_flat_pitch,
        exp_failures={"scene_wheel": [50.0]},
        exp_failure_counts={"scene_wheel": {50.0: 3}},
        inject_na=False, na_total_trials=0,
        n_select=3,
        scatter_only=False, intra_spread=None,
        scatter_dodge_width=None, scatter_mean_line=False,
    ),
    "step": TerrainPlotSpec(
        name="step", row_label="Step",
        gate_end=_STEP_END_X,
        gate_exempt=frozenset(),
        recompute=_recompute_step_065,
        vel_extractor=extract_step_q60, pitch_extractor=extract_step_pitch_q60,
        exp_failures={"scene_wheel": [10.0, 20.0]},
        exp_failure_counts={"scene_wheel": {10.0: 3, 20.0: 3}},
        inject_na=False, na_total_trials=0,
        n_select=3,
        scatter_only=False, intra_spread=None,
        scatter_dodge_width=None, scatter_mean_line=False,
    ),
    "rough": TerrainPlotSpec(
        name="rough", row_label="Rough",
        gate_end=0.155,
        gate_exempt=frozenset({("scene1", 10.0)}),
        recompute=_recompute_rough,
        vel_extractor=extract_rough, pitch_extractor=None,
        exp_failures={}, exp_failure_counts={},
        inject_na=True, na_total_trials=5,
        n_select=5,
        scatter_only=True, intra_spread=0.0,
        scatter_dodge_width=8.0, scatter_mean_line=True,
    ),
}

_TERRAIN_ORDER = ["flat", "step", "rough"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_na_zeros(data: dict, total_trials: int = 5):
    """Add zero-valued trials for n/a entries in experimental data."""
    for d in data.values():
        for freq in d["mean_freqs"]:
            n_success = sum(1 for f in d["freqs"] if f == freq)
            n_failed = total_trials - n_success
            if n_failed > 0:
                d["freqs"].extend([freq] * n_failed)
                d["trials"].extend([0.0] * n_failed)


def _strip_failure_freqs(data: dict, failures: dict[str, list[float]]):
    """Remove failure frequencies from data entirely."""
    for scene, fail_freqs in failures.items():
        if scene not in data:
            continue
        d = data[scene]
        for ff in fail_freqs:
            keep = [j for j, f in enumerate(d["freqs"]) if f != ff]
            d["freqs"] = [d["freqs"][j] for j in keep]
            d["trials"] = [d["trials"][j] for j in keep]
            if ff in d["mean_freqs"]:
                idx = d["mean_freqs"].index(ff)
                d["mean_freqs"].pop(idx)
                d["means"].pop(idx)
                d["stds"].pop(idx)


def _share_ylim(ax1: plt.Axes, ax2: plt.Axes):
    """Unify y-axis limits with bottom padding for below-X annotations.

    Annotations sit 6pt below y=0 with fontsize=14 (~14pt tall).
    Convert that point-space offset to data coordinates so padding
    scales correctly regardless of the panel's y-range.
    """
    y_lo = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_hi = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    fig = ax1.get_figure()
    ax_height_pts = ax1.get_position().height * fig.get_size_inches()[1] * 72
    # 6pt offset + 14pt font + 4pt breathing room = 24pt below y=0
    pad_pts = 24.0
    data_range = y_hi - y_lo if y_hi > y_lo else 1.0
    pad_data = pad_pts / ax_height_pts * data_range
    y_lo = min(y_lo, -pad_data)
    ax1.set_ylim(y_lo, y_hi)
    ax2.set_ylim(y_lo, y_hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=pathlib.Path,
        help="Run dirs for flat, step, and/or rough (auto-detected from name)",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    # Load and bucket by terrain
    terrain_data: dict[str, tuple[list[dict], pathlib.Path]] = {}
    for run_dir in args.run_dirs:
        csv_path = run_dir / "validation_trials.csv"
        if not csv_path.exists():
            candidates = sorted(run_dir.glob("*_validation_trials.csv"))
            if candidates:
                csv_path = candidates[-1]
            else:
                print(f"WARNING: no validation_trials.csv in {run_dir}, skipping")
                continue
        terrain = detect_terrain(run_dir)
        rows = load_validation_csv(csv_path)
        terrain_data[terrain] = (rows, run_dir)

    present = [t for t in _TERRAIN_ORDER if t in terrain_data]
    if not present:
        sys.exit("No validation CSVs found")

    # Log exactly which run dirs are being plotted
    print("=== Megacomposite NoCOT 065 ===")
    for t in present:
        rows, run_dir = terrain_data[t]
        print(f"  {t:6s}: {run_dir}  ({len(rows)} rows)")

    n_rows = len(present)

    # --- Figure layout ---
    fig = plt.figure(figsize=(14.0, 7.0))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)
    vel_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows + 1, 2, subplot_spec=outer[0],
        height_ratios=[0.07] + [1.0] * n_rows,
        wspace=0.08, hspace=0.45,
    )
    pitch_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows + 1, 2, subplot_spec=outer[1],
        height_ratios=[0.07] + [1.0] * n_rows,
        wspace=0.08, hspace=0.45,
    )

    # Metric group headers
    for gs, label in [(vel_gs, "Velocity (mm/s)"), (pitch_gs, "Pitch RMS (\u00b0)")]:
        ax_h = fig.add_subplot(gs[0, :])
        ax_h.text(0.5, 0.2, label, ha="center", va="center",
                  fontsize=12, fontweight="bold", transform=ax_h.transAxes)
        ax_h.axis("off")

    # Create data axes: axes[i] = [vel_exp, vel_sim, pitch_exp, pitch_sim]
    axes: list[list[plt.Axes]] = []
    for i in range(n_rows):
        axes.append([
            fig.add_subplot(vel_gs[i + 1, 0]),
            fig.add_subplot(vel_gs[i + 1, 1]),
            fig.add_subplot(pitch_gs[i + 1, 0]),
            fig.add_subplot(pitch_gs[i + 1, 1]),
        ])

    legend_ax = None

    # --- Generic terrain loop (no terrain-specific branches) ---
    for i, terrain in enumerate(present):
        rows, run_dir = terrain_data[terrain]
        spec = TERRAIN_SPECS[terrain]

        # NPZ recompute (flat: time-gate, step: 65% spatial gate, rough: none)
        if spec.recompute is not None:
            rows = spec.recompute(rows, run_dir)

        # Trial selection (step: 3/5, rough: 5/10 closest to exp ref)
        if spec.n_select is not None:
            ref_vel = _build_ref_velocities(spec.vel_extractor) if spec.vel_extractor else None
            rows = _select_trials(rows, spec.n_select,
                                  spec.exp_failures, spec.gate_end,
                                  ref_velocities=ref_vel)

        # Sim failures: always dynamic, never hardcoded
        all_failed = build_all_failed_freqs(
            rows, selected_only=True,
            gate_end=spec.gate_end, gate_exempt=spec.gate_exempt,
        )

        # Display kwargs from spec
        disp = dict(
            scatter_only=spec.scatter_only,
            intra_spread=spec.intra_spread,
            scatter_dodge_width=spec.scatter_dodge_width,
            scatter_mean_line=spec.scatter_mean_line,
        )

        # Col 0: Exp velocity
        if spec.vel_extractor is not None:
            exp_vel = _remap_exp_data(spec.vel_extractor())
            _strip_failure_freqs(exp_vel, spec.exp_failures)
            if spec.inject_na:
                _inject_na_zeros(exp_vel, spec.na_total_trials)
            plot_panel(axes[i][0], exp_vel, "", "Velocity (mm/s)",
                       spec.exp_failures, spec.exp_failure_counts, **disp)
        else:
            axes[i][0].set_visible(False)

        # Col 1: Sim velocity
        sim_vel = build_plot_data(
            rows, "vx", selected_only=True, exclude_invalid=True,
            gate_end=spec.gate_end, gate_exempt=spec.gate_exempt,
        )
        plot_panel(axes[i][1], sim_vel, "", "",
                   None, all_failed, **disp)

        # Share y-axis for velocity pair
        if spec.vel_extractor is not None:
            _share_ylim(axes[i][0], axes[i][1])

        # Col 2: Exp pitch
        if spec.pitch_extractor is not None:
            exp_pitch = _remap_exp_data(spec.pitch_extractor())
            _strip_failure_freqs(exp_pitch, spec.exp_failures)
            if spec.pitch_exclude:
                strip_freqs(exp_pitch, spec.pitch_exclude)
            plot_panel(axes[i][2], exp_pitch, "", "Pitch RMS (\u00b0)",
                       spec.exp_failures, spec.exp_failure_counts, **disp)
        else:
            axes[i][2].axis("off")
            legend_ax = axes[i][2]

        # Col 3: Sim pitch
        sim_pitch = build_plot_data(
            rows, "pitch_rms", selected_only=True, exclude_invalid=True,
            gate_end=spec.gate_end, gate_exempt=spec.gate_exempt,
        )
        if spec.pitch_exclude:
            strip_freqs(sim_pitch, spec.pitch_exclude)
        plot_panel(axes[i][3], sim_pitch, "", "",
                   None, all_failed, **disp)

        # Share y-axis for pitch pair (or just pad sim panel if no exp)
        if spec.pitch_extractor is not None:
            _share_ylim(axes[i][2], axes[i][3])
        else:
            _share_ylim(axes[i][3], axes[i][3])

    # --- Post-hoc axis cleanup ---
    from matplotlib.collections import PathCollection
    letter_idx = 0
    for i in range(n_rows):
        spec = TERRAIN_SPECS[present[i]]
        for j in range(4):
            ax = axes[i][j]
            if not ax.get_visible() or ax is legend_ax:
                continue

            for coll in ax.collections:
                if isinstance(coll, PathCollection):
                    coll.set_sizes([12])
            for line in ax.lines:
                if line.get_marker() == 'x':
                    line.set_markersize(6)
                    line.set_markeredgewidth(1.5)

            letter = chr(ord('a') + letter_idx)
            ax.text(0.02, 0.95, f"({letter})", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="top", ha="left")
            letter_idx += 1

            if i == 0:
                sub = "Experiment" if j % 2 == 0 else "Simulation"
                ax.set_title(sub, fontsize=11)

            if i < n_rows - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("")

            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False, right=False)
            if j in (1, 3):
                ax.tick_params(axis="y", right=True, labelright=True)
                ax.yaxis.set_label_position("right")

        axes[i][0].set_ylabel(spec.row_label, fontsize=12, fontweight="bold")

    # --- Legend ---
    handles, labels = None, None
    for i in range(n_rows):
        for j in range(4):
            if axes[i][j].get_visible():
                h, l = axes[i][j].get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
        if handles:
            break

    for i in range(n_rows):
        for j in range(4):
            leg = axes[i][j].get_legend()
            if leg:
                leg.remove()

    if handles and legend_ax is not None:
        legend_ax.legend(
            handles, labels, loc="upper center", fontsize=9,
            frameon=True, framealpha=0.9, edgecolor="0.8",
            title="Morphology", title_fontsize=10,
        )
        legend_ax.text(
            0.5, 0.08, "x-axis: Drive frequency (Hz)",
            transform=legend_ax.transAxes,
            ha="center", va="center", fontsize=11,
        )
        legend_ax.plot(0.38, -0.08, "x", color="red", markersize=7,
                       markeredgewidth=1.5, transform=legend_ax.transAxes,
                       clip_on=False)
        legend_ax.text(0.42, -0.08, "= failure", transform=legend_ax.transAxes,
                       ha="left", va="center", fontsize=11)
    elif handles:
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   fontsize=7, framealpha=0.9)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = args.output or f"plots/{ts}_megacomposite_nocot_065.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
