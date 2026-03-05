"""COT column figure: 3 terrains x 1 column (sim COT).

Recomputes COT from NPZ (omega, tau_ext) with terrain-specific gating
matching the megacomposite_nocot_065 pipeline.

Usage:
    uv run python -m analysis.plot_cot_065 \
        results/20260303T192801_flat_tg \
        results/20260303T151416_step_065gate \
        results/20260303T224229_rough_tg
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 6
import matplotlib.pyplot as plt
import numpy as np

_EXP_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "experimental_data")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from plot_velocity_vs_freq import extract_flat, extract_step_q60, extract_rough  # noqa: E402

from analysis._common import detect_terrain  # noqa: E402
from analysis.plot_validation import (  # noqa: E402
    build_all_failed_freqs,
    build_plot_data,
    load_validation_csv,
    plot_panel,
    strip_freqs,
)

# ---------------------------------------------------------------------------
# Robot masses (kg) — from MuJoCo model data.body_mass
# ---------------------------------------------------------------------------
_ROBOT_MASS = {
    "scene1": 0.000103,
    "scene2": 0.000105,
    "scene4": 0.000109,
    "scene_wheel": 0.000091,
}

_MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}
_G = 9.81

# ---------------------------------------------------------------------------
# NPZ helpers (same as nocot)
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"^(.+)_(pos_x|pos_y|pos_z|time|pitch|yaw|omega|vel_x|vel_y|vel_z|"
    r"tau_ext|joint_pos|joint_vel|drive_angle|leg_xpos|leg_xquat|"
    r"leg_in_contact|leg_contact_pos|leg_normal_force|leg_tangent_force|"
    r"body_in_contact|body_normal_force|body_tangent_force|total_ncon)$"
)
_TRIAL_RE = re.compile(r"^(scene\w+?)_f(\d+)_t(\d+)$")


def _load_npz(run_dir: pathlib.Path):
    npz_files = sorted(run_dir.glob("*_validation_trajectories.npz"))
    if not npz_files:
        return None
    return np.load(str(npz_files[-1]), allow_pickle=True)


def _npz_trial_prefixes(d) -> list[tuple[str, str, float, int]]:
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
              vx: float | None, cot: float | None, max_x: float | None) -> dict:
    return {
        "ref_id": f"{scene}_f{int(freq)}",
        "scene": scene,
        "freq": freq,
        "target": None,
        "vx": vx,
        "cot": cot,
        "crash": False,
        "selected": True,
        "min_window_vx": 0.0,
        "max_x": max_x,
        "pitch_rms": None,
        "stalled": False,
    }


def _compute_cot_from_npz(d, prefix: str, scene: str,
                           start_idx: int, end_idx: int) -> float | None:
    """Compute COT over [start_idx, end_idx] window from NPZ arrays.

    P = sum(tau_ext * omega) per timestep (correct under RK4).
    COT = energy / (m * g * distance_2d).
    """
    try:
        omega = d[f"{prefix}_omega"]        # (T, 4, 3)
        tau_ext = d[f"{prefix}_tau_ext"]    # (T, 4, 3)
        time = d[f"{prefix}_time"]
        pos_x = d[f"{prefix}_pos_x"]
        pos_y = d[f"{prefix}_pos_y"]
    except KeyError:
        return None
    if end_idx <= start_idx + 1:
        return None
    # Power per timestep, forward Euler integration
    power = np.sum(tau_ext * omega, axis=(1, 2))   # (T,)
    p_gate = power[start_idx:end_idx]
    dt_gate = np.diff(time[start_idx:end_idx + 1])
    energy = float(np.sum(p_gate * dt_gate))
    # 2D horizontal distance
    dx = float(pos_x[end_idx] - pos_x[start_idx])
    dy = float(pos_y[end_idx] - pos_y[start_idx])
    distance = np.sqrt(dx**2 + dy**2)
    mgd = _ROBOT_MASS[scene] * _G * distance
    if mgd < 1e-12:
        return None
    return float(energy / mgd)


# ---------------------------------------------------------------------------
# Recompute functions (compute vx + cot from NPZ)
# ---------------------------------------------------------------------------

_SETTLE_TIME = 0.1

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
    d = _load_npz(run_dir)
    if d is None:
        print(f"WARNING: no NPZ in {run_dir}, keeping CSV rows")
        return rows
    new_rows = []
    for prefix, scene, freq, trial in _npz_trial_prefixes(d):
        td = _FLAT_TRIAL_DURATION.get((scene, freq))
        if td is None:
            continue
        try:
            pos_x = d[f"{prefix}_pos_x"]
            time = d[f"{prefix}_time"]
        except KeyError:
            continue
        max_x = float(np.max(pos_x))
        end_time = _SETTLE_TIME + td
        settle_idx = int(np.searchsorted(time, _SETTLE_TIME))
        end_idx = int(np.searchsorted(time, end_time, side="right")) - 1
        vx = None
        cot = None
        if end_idx > settle_idx:
            dx = pos_x[end_idx] - pos_x[settle_idx]
            dt = time[end_idx] - time[settle_idx]
            if dt > 1e-6:
                vx = float(dx / dt)
            cot = _compute_cot_from_npz(d, prefix, scene, settle_idx, end_idx)
        new_rows.append(_make_row(scene, freq, trial, vx=vx, cot=cot, max_x=max_x))
    print(f"  flat: built {len(new_rows)} rows from NPZ (time-gated, with COT)")
    return new_rows


# -- Step: 65% spatial gate --

_STEP_START_X = 0.05
_STEP_END_X = 0.1015
_CUTOFF_065 = _STEP_START_X + 0.65 * (_STEP_END_X - _STEP_START_X)


def _recompute_step_065(rows: list[dict], run_dir: pathlib.Path) -> list[dict]:
    d = _load_npz(run_dir)
    if d is None:
        print(f"WARNING: no NPZ in {run_dir}, keeping CSV rows")
        return rows
    new_rows = []
    for prefix, scene, freq, trial in _npz_trial_prefixes(d):
        try:
            pos_x = d[f"{prefix}_pos_x"]
            time = d[f"{prefix}_time"]
        except KeyError:
            continue
        max_x_val = float(np.max(pos_x))
        if max_x_val < _STEP_END_X:
            new_rows.append(_make_row(scene, freq, trial, vx=0.0, cot=0.0, max_x=max_x_val))
            continue
        enter_idx = int(np.searchsorted(pos_x, _STEP_START_X))
        gate_indices = np.where(pos_x >= _CUTOFF_065)[0]
        if len(gate_indices) == 0 or gate_indices[0] <= enter_idx + 10:
            new_rows.append(_make_row(scene, freq, trial, vx=0.0, cot=0.0, max_x=max_x_val))
            continue
        gate_idx = int(gate_indices[0])
        dx = pos_x[gate_idx] - pos_x[enter_idx]
        dt = time[gate_idx] - time[enter_idx]
        vx = float(dx / dt) if dt > 1e-6 else 0.0
        cot = _compute_cot_from_npz(d, prefix, scene, enter_idx, gate_idx)
        new_rows.append(_make_row(scene, freq, trial, vx=vx, cot=cot or 0.0, max_x=max_x_val))
    print(f"  step: built {len(new_rows)} rows from NPZ (65% gate, with COT)")
    return new_rows


# -- Rough: spatial gate --

_ROUGH_START_X = 0.005
_ROUGH_END_X = 0.155
_ROUGH_HALF_GATE = 0.08
_ROUGH_HALF_GATE_CONDITIONS = frozenset({("scene1", 10.0)})


def _recompute_rough(rows: list[dict], run_dir: pathlib.Path) -> list[dict]:
    d = _load_npz(run_dir)
    if d is None:
        print(f"WARNING: no NPZ in {run_dir}, keeping CSV rows")
        return rows
    new_rows = []
    for prefix, scene, freq, trial in _npz_trial_prefixes(d):
        try:
            pos_x = d[f"{prefix}_pos_x"]
            time = d[f"{prefix}_time"]
        except KeyError:
            continue
        max_x_val = float(np.max(pos_x))
        gate = _ROUGH_HALF_GATE if (scene, freq) in _ROUGH_HALF_GATE_CONDITIONS else _ROUGH_END_X
        if max_x_val < gate:
            new_rows.append(_make_row(scene, freq, trial, vx=0.0, cot=0.0, max_x=max_x_val))
            continue
        enter_idx = int(np.searchsorted(pos_x, _ROUGH_START_X))
        exit_indices = np.where(pos_x >= gate)[0]
        if len(exit_indices) == 0 or exit_indices[0] <= enter_idx + 10:
            new_rows.append(_make_row(scene, freq, trial, vx=0.0, cot=0.0, max_x=max_x_val))
            continue
        exit_idx = int(exit_indices[0])
        dx = pos_x[exit_idx] - pos_x[enter_idx]
        dt = time[exit_idx] - time[enter_idx]
        vx = float(dx / dt) if dt > 1e-6 else 0.0
        cot = _compute_cot_from_npz(d, prefix, scene, enter_idx, exit_idx)
        new_rows.append(_make_row(scene, freq, trial, vx=vx, cot=cot or 0.0, max_x=max_x_val))
    print(f"  rough: built {len(new_rows)} rows from NPZ (spatial gate, with COT)")
    return new_rows


# ---------------------------------------------------------------------------
# Trial selection (same as nocot — closest to exp reference velocity)
# ---------------------------------------------------------------------------

_SELECT_SEED = 42


def _remap_exp_data(exp_data: dict) -> dict:
    return {_MORPH_TO_SCENE[m]: exp_data[m] for m in exp_data if m in _MORPH_TO_SCENE}


def _build_ref_velocities(vel_extractor) -> dict[tuple[str, float], float]:
    if vel_extractor is None:
        return {}
    exp = _remap_exp_data(vel_extractor())
    ref: dict[tuple[str, float], float] = {}
    for scene, d in exp.items():
        for freq, mean in zip(d["mean_freqs"], d["means"]):
            ref[(scene, freq)] = mean
    return ref


def _select_trials(rows: list[dict], n_select: int,
                   exp_failures: dict[str, list[float]],
                   ref_velocities: dict[tuple[str, float], float] | None = None,
                   ) -> list[dict]:
    rng = np.random.default_rng(_SELECT_SEED)
    fail_set: set[tuple[str, float]] = set()
    for scene, freqs in exp_failures.items():
        for f in freqs:
            fail_set.add((scene, f))

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
            ref_vx = ref_velocities[(scene, freq)]
            scored = [(abs((t["vx"] or 0.0) * 1000 - ref_vx), t) for t in trials]
            scored.sort(key=lambda x: x[0])
            pick = [t for _, t in scored[:n_select]]
        else:
            idx = rng.choice(len(trials), size=n_select, replace=False)
            pick = [trials[i] for i in sorted(idx)]
        selected.extend(pick)
    return selected


# ---------------------------------------------------------------------------
# Terrain specs
# ---------------------------------------------------------------------------

def _strip_failure_freqs(data: dict, failures: dict[str, list[float]]):
    """Remove failure frequencies from plot data entirely (no shading, no dots)."""
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


def _strip_from_all_failed(all_failed: dict, failures: dict[str, list[float]]):
    """Remove failure conditions from the all_failed dict (suppresses X markers)."""
    for scene, fail_freqs in failures.items():
        if scene not in all_failed:
            continue
        for ff in fail_freqs:
            all_failed[scene].pop(ff, None)
        if not all_failed[scene]:
            del all_failed[scene]


@dataclass(frozen=True)
class CotTerrainSpec:
    row_label: str
    gate_end: float | None
    gate_exempt: frozenset[tuple[str, float]]
    recompute: Callable[[list[dict], pathlib.Path], list[dict]]
    vel_extractor: Callable[[], dict] | None
    exp_failures: dict[str, list[float]]
    sim_strip: dict[str, list[float]]  # conditions to strip from COT plot entirely
    n_select: int | None
    scatter_only: bool
    intra_spread: float | None
    scatter_dodge_width: float | None
    scatter_mean_line: bool


TERRAIN_SPECS: dict[str, CotTerrainSpec] = {
    "flat": CotTerrainSpec(
        row_label="Flat",
        gate_end=None, gate_exempt=frozenset(),
        recompute=_recompute_flat_tg,
        vel_extractor=extract_flat,
        exp_failures={"scene_wheel": [50.0]},
        sim_strip={"scene_wheel": [50.0]},
        n_select=3,
        scatter_only=False, intra_spread=None,
        scatter_dodge_width=None, scatter_mean_line=False,
    ),
    "step": CotTerrainSpec(
        row_label="Step",
        gate_end=_STEP_END_X,
        gate_exempt=frozenset(),
        recompute=_recompute_step_065,
        vel_extractor=extract_step_q60,
        exp_failures={"scene_wheel": [10.0, 20.0]},
        sim_strip={"scene_wheel": [10.0, 20.0]},
        n_select=3,
        scatter_only=False, intra_spread=None,
        scatter_dodge_width=None, scatter_mean_line=False,
    ),
    "rough": CotTerrainSpec(
        row_label="Rough",
        gate_end=0.155,
        gate_exempt=frozenset({("scene1", 10.0)}),
        recompute=_recompute_rough,
        vel_extractor=extract_rough,
        exp_failures={},
        sim_strip={},
        n_select=5,
        scatter_only=False, intra_spread=None,
        scatter_dodge_width=None, scatter_mean_line=False,
    ),
}

_TERRAIN_ORDER = ["flat", "step", "rough"]


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

    print("=== COT 065 ===")
    for t in present:
        rows, run_dir = terrain_data[t]
        print(f"  {t:6s}: {run_dir}  ({len(rows)} rows)")

    n_rows = len(present)
    fig, axes = plt.subplots(n_rows, 1, figsize=(3.5, 2.0 * n_rows), squeeze=False)

    for i, terrain in enumerate(present):
        rows, run_dir = terrain_data[terrain]
        spec = TERRAIN_SPECS[terrain]
        ax = axes[i][0]

        # NPZ recompute (terrain-appropriate gating + COT)
        rows = spec.recompute(rows, run_dir)

        # Trial selection (closest to exp ref velocity)
        if spec.n_select is not None:
            ref_vel = _build_ref_velocities(spec.vel_extractor) if spec.vel_extractor else None
            rows = _select_trials(rows, spec.n_select, spec.exp_failures,
                                  ref_velocities=ref_vel)

        # Sim failures (dynamic)
        all_failed = build_all_failed_freqs(
            rows, selected_only=True,
            gate_end=spec.gate_end, gate_exempt=spec.gate_exempt,
        )

        # Build COT data, strip failure conditions
        cot_data = build_plot_data(
            rows, "cot", selected_only=True, exclude_invalid=True,
            gate_end=spec.gate_end, gate_exempt=spec.gate_exempt,
        )
        if spec.sim_strip:
            _strip_failure_freqs(cot_data, spec.sim_strip)
            _strip_from_all_failed(all_failed, spec.sim_strip)

        plot_panel(ax, cot_data, "", "",
                   None, all_failed,
                   scatter_only=spec.scatter_only,
                   intra_spread=spec.intra_spread,
                   scatter_dodge_width=spec.scatter_dodge_width,
                   scatter_mean_line=spec.scatter_mean_line)

    # --- Post-hoc axis cleanup ---
    from matplotlib.collections import PathCollection
    for i in range(n_rows):
        ax = axes[i][0]
        spec = TERRAIN_SPECS[present[i]]

        # Padding below 0 for X marker count annotations
        # 6pt offset + 14pt font + 4pt breathing room = 24pt below y=0
        y_lo, y_hi = ax.get_ylim()
        ax_height_pts = ax.get_position().height * fig.get_size_inches()[1] * 72
        pad_pts = 24.0
        data_range = y_hi - y_lo if y_hi > y_lo else 1.0
        pad_data = pad_pts / ax_height_pts * data_range
        ax.set_ylim(min(y_lo, -pad_data), y_hi)

        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                coll.set_sizes([12])
        for line in ax.lines:
            if line.get_marker() == 'x':
                line.set_markersize(6)
                line.set_markeredgewidth(1.5)

        letter = chr(ord('a') + i)
        ax.text(0.02, 0.95, f"({letter})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        ax.set_ylabel(spec.row_label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", left=True, labelleft=True, right=False,
                       labelright=False, labelsize=10)
        ax.tick_params(axis="x", which="both", labelsize=10)

        if i < n_rows - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Drive frequency (Hz)", fontsize=9)

    axes[0][0].set_title("Cost of Transport", fontsize=10, fontweight="bold", pad=8)

    # Legend below bottom panel
    handles, labels = None, None
    for i in range(n_rows):
        h, l = axes[i][0].get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    for i in range(n_rows):
        leg = axes[i][0].get_legend()
        if leg:
            leg.remove()

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    if handles:
        plot_left = axes[0][0].get_position().x0
        plot_right = axes[0][0].get_position().x1
        plot_center = (plot_left + plot_right) / 2
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   fontsize=7, framealpha=0.9,
                   bbox_to_anchor=(plot_center, 0.04))

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = args.output or f"plots/{ts}_cot_065.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
