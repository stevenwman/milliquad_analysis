"""Dump all nocot + COT numbers from canonical runs as markdown tables.

Runs the same NPZ recompute + trial selection pipeline as
plot_megacomposite_nocot_065.py and plot_cot_065.py.

Usage:
    cd milliquad_opt
    uv run python -m analysis.dump_numbers \
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

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EXP_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "experimental_data")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from plot_velocity_vs_freq import extract_flat, extract_step_q60, extract_rough  # noqa: E402
from plot_pitch_vs_freq import extract_flat_pitch, extract_step_pitch_q60  # noqa: E402

from analysis._common import detect_terrain  # noqa: E402
from analysis.plot_validation import load_validation_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Robot masses (kg) — from MuJoCo model data.body_mass
# ---------------------------------------------------------------------------
_ROBOT_MASS = {
    "scene1": 0.000103,
    "scene2": 0.000105,
    "scene4": 0.000109,
    "scene_wheel": 0.000091,
}
_G = 9.81

# ---------------------------------------------------------------------------
# NPZ helpers (shared with nocot_065 / cot_065)
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


def _make_row(scene, freq, trial, *, vx, pitch_rms, cot, max_x) -> dict:
    return {
        "ref_id": f"{scene}_f{int(freq)}",
        "scene": scene, "freq": freq,
        "target": None, "vx": vx, "cot": cot,
        "crash": False, "selected": True,
        "min_window_vx": 0.0, "max_x": max_x,
        "pitch_rms": pitch_rms, "stalled": False,
    }


def _compute_cot_from_npz(d, prefix, scene, start_idx, end_idx):
    try:
        omega = d[f"{prefix}_omega"]
        tau_ext = d[f"{prefix}_tau_ext"]
        time = d[f"{prefix}_time"]
        pos_x = d[f"{prefix}_pos_x"]
        pos_y = d[f"{prefix}_pos_y"]
    except KeyError:
        return None
    if end_idx <= start_idx + 1:
        return None
    power = np.sum(tau_ext * omega, axis=(1, 2))
    p_gate = power[start_idx:end_idx]
    dt_gate = np.diff(time[start_idx:end_idx + 1])
    energy = float(np.sum(p_gate * dt_gate))
    dx = float(pos_x[end_idx] - pos_x[start_idx])
    dy = float(pos_y[end_idx] - pos_y[start_idx])
    distance = np.sqrt(dx**2 + dy**2)
    mgd = _ROBOT_MASS[scene] * _G * distance
    if mgd < 1e-12:
        return None
    return float(energy / mgd)


# ---------------------------------------------------------------------------
# Recompute functions (unified: vx + pitch + cot)
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


def _recompute_flat(rows, run_dir):
    d = _load_npz(run_dir)
    if d is None:
        return rows
    new_rows = []
    for prefix, scene, freq, trial in _npz_trial_prefixes(d):
        td = _FLAT_TRIAL_DURATION.get((scene, freq))
        if td is None:
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
        vx = pitch_rms = cot = None
        if end_idx > settle_idx:
            dx = pos_x[end_idx] - pos_x[settle_idx]
            dt = time[end_idx] - time[settle_idx]
            if dt > 1e-6:
                vx = float(dx / dt)
            p_gate = pitch[settle_idx:end_idx + 1]
            if len(p_gate) > 1:
                pitch_rms = float(np.std(p_gate - p_gate[0]))
            cot = _compute_cot_from_npz(d, prefix, scene, settle_idx, end_idx)
        new_rows.append(_make_row(scene, freq, trial,
                                  vx=vx, pitch_rms=pitch_rms, cot=cot, max_x=max_x))
    return new_rows


_STEP_START_X = 0.05
_STEP_END_X = 0.1015
_CUTOFF_065 = _STEP_START_X + 0.65 * (_STEP_END_X - _STEP_START_X)


def _recompute_step(rows, run_dir):
    d = _load_npz(run_dir)
    if d is None:
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
        if max_x_val < _STEP_END_X:
            new_rows.append(_make_row(scene, freq, trial,
                                      vx=0.0, pitch_rms=0.0, cot=0.0, max_x=max_x_val))
            continue
        enter_idx = int(np.searchsorted(pos_x, _STEP_START_X))
        gate_indices = np.where(pos_x >= _CUTOFF_065)[0]
        if len(gate_indices) == 0 or gate_indices[0] <= enter_idx + 10:
            new_rows.append(_make_row(scene, freq, trial,
                                      vx=0.0, pitch_rms=0.0, cot=0.0, max_x=max_x_val))
            continue
        gate_idx = int(gate_indices[0])
        dx = pos_x[gate_idx] - pos_x[enter_idx]
        dt = time[gate_idx] - time[enter_idx]
        vx = float(dx / dt) if dt > 1e-6 else 0.0
        p_gate = pitch[enter_idx:gate_idx + 1]
        pitch_rms = float(np.std(p_gate - p_gate[0])) if len(p_gate) > 1 else 0.0
        cot = _compute_cot_from_npz(d, prefix, scene, enter_idx, gate_idx)
        new_rows.append(_make_row(scene, freq, trial,
                                  vx=vx, pitch_rms=pitch_rms, cot=cot or 0.0, max_x=max_x_val))
    return new_rows


_ROUGH_START_X = 0.005
_ROUGH_END_X = 0.155
_ROUGH_HALF_GATE = 0.08
_ROUGH_HALF_GATE_CONDITIONS = frozenset({("scene1", 10.0)})


def _recompute_rough(rows, run_dir):
    d = _load_npz(run_dir)
    if d is None:
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
        if max_x_val < gate:
            new_rows.append(_make_row(scene, freq, trial,
                                      vx=0.0, pitch_rms=0.0, cot=0.0, max_x=max_x_val))
            continue
        enter_idx = int(np.searchsorted(pos_x, _ROUGH_START_X))
        exit_indices = np.where(pos_x >= gate)[0]
        if len(exit_indices) == 0 or exit_indices[0] <= enter_idx + 10:
            new_rows.append(_make_row(scene, freq, trial,
                                      vx=0.0, pitch_rms=0.0, cot=0.0, max_x=max_x_val))
            continue
        exit_idx = int(exit_indices[0])
        dx = pos_x[exit_idx] - pos_x[enter_idx]
        dt = time[exit_idx] - time[enter_idx]
        vx = float(dx / dt) if dt > 1e-6 else 0.0
        p_gate = pitch[enter_idx:exit_idx + 1]
        pitch_rms = float(np.std(p_gate - p_gate[0])) if len(p_gate) > 1 else 0.0
        cot = _compute_cot_from_npz(d, prefix, scene, enter_idx, exit_idx)
        new_rows.append(_make_row(scene, freq, trial,
                                  vx=vx, pitch_rms=pitch_rms, cot=cot or 0.0, max_x=max_x_val))
    return new_rows


# ---------------------------------------------------------------------------
# Trial selection (same as nocot_065 / cot_065)
# ---------------------------------------------------------------------------

_SELECT_SEED = 42
_MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}
_SCENE_TO_LABEL = {"scene1": "L1", "scene2": "L2", "scene4": "L4", "scene_wheel": "WR"}


def _remap_exp_data(exp_data):
    return {_MORPH_TO_SCENE[m]: exp_data[m] for m in exp_data if m in _MORPH_TO_SCENE}


def _build_ref_velocities(vel_extractor):
    if vel_extractor is None:
        return {}
    exp = _remap_exp_data(vel_extractor())
    ref = {}
    for scene, d in exp.items():
        for freq, mean in zip(d["mean_freqs"], d["means"]):
            ref[(scene, freq)] = mean
    return ref


def _select_trials(rows, n_select, exp_failures, ref_velocities=None):
    rng = np.random.default_rng(_SELECT_SEED)
    fail_set = set()
    for scene, freqs in exp_failures.items():
        for f in freqs:
            fail_set.add((scene, f))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scene"], r["freq"])].append(r)
    selected = []
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
# Terrain config
# ---------------------------------------------------------------------------

_TERRAIN_CONFIG = {
    "flat": {
        "recompute": _recompute_flat,
        "vel_extractor": extract_flat,
        "pitch_extractor": extract_flat_pitch,
        "exp_failures": {"scene_wheel": [50.0]},
        "n_select": 3,
        "gate_end": None,
        "gate_exempt": frozenset(),
        "sim_strip": {"scene_wheel": [50.0]},
    },
    "step": {
        "recompute": _recompute_step,
        "vel_extractor": extract_step_q60,
        "pitch_extractor": extract_step_pitch_q60,
        "exp_failures": {"scene_wheel": [10.0, 20.0]},
        "n_select": 3,
        "gate_end": _STEP_END_X,
        "gate_exempt": frozenset(),
        "sim_strip": {"scene_wheel": [10.0, 20.0]},
    },
    "rough": {
        "recompute": _recompute_rough,
        "vel_extractor": extract_rough,
        "pitch_extractor": None,
        "exp_failures": {},
        "n_select": 5,
        "gate_end": 0.155,
        "gate_exempt": frozenset({("scene1", 10.0)}),
        "sim_strip": {},
    },
}

_TERRAIN_ORDER = ["flat", "step", "rough"]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _group_stats(rows, metric):
    """Group by (scene, freq), compute mean/std of metric (skip None/0.0 gate failures)."""
    groups = defaultdict(list)
    for r in rows:
        val = r.get(metric)
        if val is not None and val != 0.0:
            groups[(r["scene"], r["freq"])].append(val)
    result = {}
    for (scene, freq), vals in sorted(groups.items()):
        arr = np.array(vals)
        result[(scene, freq)] = {
            "n": len(vals),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    return result


def _exp_table(extractor, label_suffix=""):
    """Extract experimental data and return {(scene, freq): {n, mean, std}}."""
    raw = _remap_exp_data(extractor())
    result = {}
    for scene, d in raw.items():
        for f, m, s in zip(d["mean_freqs"], d["means"], d["stds"]):
            # Count trials from the raw data
            n = sum(1 for ff in d["freqs"] if ff == f)
            result[(scene, f)] = {"n": n, "mean": m, "std": s}
    return result


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def _print_sim_table(terrain, vx_stats, pitch_stats, cot_stats):
    print(f"\n## Sim {terrain.title()}\n")
    print("| Scene | Freq | n | vx mean | vx std | pitch mean | pitch std | COT mean | COT std |")
    print("|-------|------|---|---------|--------|------------|-----------|----------|---------|")
    all_keys = sorted(set(list(vx_stats.keys()) + list(pitch_stats.keys()) + list(cot_stats.keys())))
    for (scene, freq) in all_keys:
        v = vx_stats.get((scene, freq), {})
        p = pitch_stats.get((scene, freq), {})
        c = cot_stats.get((scene, freq), {})
        n = v.get("n", p.get("n", c.get("n", 0)))
        vx_m = f"{v['mean']*1000:.1f}" if v else "-"
        vx_s = f"{v['std']*1000:.1f}" if v else "-"
        p_m = f"{p['mean']:.2f}" if p else "-"
        p_s = f"{p['std']:.2f}" if p else "-"
        c_m = f"{c['mean']:.2f}" if c else "-"
        c_s = f"{c['std']:.2f}" if c else "-"
        print(f"| {scene} | {int(freq)} | {n} | {vx_m} | {vx_s} | {p_m} | {p_s} | {c_m} | {c_s} |")


def _print_exp_vel_table(terrain, exp_data):
    print(f"\n## Exp {terrain.title()} (velocity mm/s)\n")
    print("| Scene | Freq | n | mean | std |")
    print("|-------|------|---|------|-----|")
    for (scene, freq) in sorted(exp_data.keys()):
        d = exp_data[(scene, freq)]
        print(f"| {scene} | {int(freq)} | {d['n']} | {d['mean']:.1f} | {d['std']:.1f} |")


def _print_exp_pitch_table(terrain, exp_data):
    print(f"\n## Exp {terrain.title()} (pitch RMS degrees)\n")
    print("| Scene | Freq | n | mean | std |")
    print("|-------|------|---|------|-----|")
    for (scene, freq) in sorted(exp_data.keys()):
        d = exp_data[(scene, freq)]
        print(f"| {scene} | {int(freq)} | {d['n']} | {d['mean']:.2f} | {d['std']:.2f} |")


def _print_error_table(terrain, sim_stats, exp_data, *,
                       extra_exclude: dict[str, list[float]] | None = None,
                       extra_exclude_label: str = "",
                       note: str = ""):
    """Print sim vs exp velocity error table.

    extra_exclude: additional conditions to exclude for a secondary summary row.
    """
    print(f"\n## {terrain.title()} Velocity Error (3 selected trials)\n")
    print("| Condition | Exp (mm/s) | Sim mean | Sim std | % Error |")
    print("|-----------|-----------|----------|---------|---------|")

    extra_set = set()
    if extra_exclude:
        for scene, freqs in extra_exclude.items():
            for f in freqs:
                extra_set.add((scene, f))

    errors_all = []
    errors_clean = []
    for (scene, freq) in sorted(exp_data.keys()):
        if (scene, freq) not in sim_stats:
            continue
        exp_mean = exp_data[(scene, freq)]["mean"]
        sim = sim_stats[(scene, freq)]
        sim_mean = sim["mean"] * 1000
        sim_std = sim["std"] * 1000
        pct = abs(sim_mean - exp_mean) / exp_mean * 100 if exp_mean > 0 else 0.0
        label = f"{_SCENE_TO_LABEL[scene]} f{int(freq)}"
        print(f"| {label} | {exp_mean:.1f} | {sim_mean:.1f} | {sim_std:.1f} | {pct:.1f}% |")
        errors_all.append(pct)
        if (scene, freq) not in extra_set:
            errors_clean.append(pct)

    if errors_all:
        print(f"\n| Subset | Mean % Error | Median % Error | N |")
        print(f"|--------|-------------|----------------|---|")
        print(f"| All {len(errors_all)} conditions | {np.mean(errors_all):.1f}% | {np.median(errors_all):.1f}% | {len(errors_all)} |")
        if extra_exclude and errors_clean:
            print(f"| {extra_exclude_label} | {np.mean(errors_clean):.1f}% | {np.median(errors_clean):.1f}% | {len(errors_clean)} |")
    if note:
        print(f"\n{note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    terrain_data = {}
    for run_dir in args.run_dirs:
        csv_path = run_dir / "validation_trials.csv"
        if not csv_path.exists():
            candidates = sorted(run_dir.glob("*_validation_trials.csv"))
            if candidates:
                csv_path = candidates[-1]
            else:
                continue
        terrain = detect_terrain(run_dir)
        rows = load_validation_csv(csv_path)
        terrain_data[terrain] = (rows, run_dir)

    present = [t for t in _TERRAIN_ORDER if t in terrain_data]

    # Header
    print("# NoCOT + COT Numbers Reference")
    print()
    print("Auto-generated from NPZ recompute pipeline (same as nocot_065 + cot_065 scripts).")
    print()

    # Run dirs
    print("## Canonical Run Dirs\n")
    print("| Terrain | Run dir |")
    print("|---------|---------|")
    for t in present:
        _, run_dir = terrain_data[t]
        print(f"| {t} | `{run_dir}` |")

    # Pipeline summary
    print("\n## Pipeline\n")
    print("- **Flat**: time-gated to match experimental recording duration per condition")
    print("- **Step**: 65% spatial gate (0.05m to 0.0835m), success = full traversal to 0.1015m")
    print("- **Rough**: spatial gate to 0.155m (scene1_f10: half-gate 0.08m)")
    print("- **Trial selection**: flat/step 3 per condition, rough 5 per condition, closest to exp ref velocity")
    print("- **COT**: P = tau_ext . omega (correct under RK4), COT = energy / (m*g*distance_2d)")

    # Process each terrain
    sim_results = {}  # terrain -> (vx_stats, pitch_stats, cot_stats, selected_rows)
    for t in present:
        rows, run_dir = terrain_data[t]
        cfg = _TERRAIN_CONFIG[t]

        # Recompute from NPZ
        rows = cfg["recompute"](rows, run_dir)

        # Trial selection
        ref_vel = _build_ref_velocities(cfg["vel_extractor"]) if cfg["vel_extractor"] else None
        rows = _select_trials(rows, cfg["n_select"], cfg["exp_failures"],
                              ref_velocities=ref_vel)

        vx_stats = _group_stats(rows, "vx")
        pitch_stats = _group_stats(rows, "pitch_rms")
        cot_stats = _group_stats(rows, "cot")
        sim_results[t] = (vx_stats, pitch_stats, cot_stats, rows)

    print("\n---\n")
    print("# Sim Data (selected trials, mean +/- std)\n")
    print("Units: velocity mm/s, pitch degrees, COT dimensionless.")

    for t in present:
        vx_stats, pitch_stats, cot_stats, _ = sim_results[t]
        _print_sim_table(t, vx_stats, pitch_stats, cot_stats)

    print("\n---\n")
    print("# Experimental Data\n")
    print("From `experimental_data/plot_velocity_vs_freq.py` and `plot_pitch_vs_freq.py`.")

    for t in present:
        cfg = _TERRAIN_CONFIG[t]
        exp_vel = _exp_table(cfg["vel_extractor"])
        _print_exp_vel_table(t, exp_vel)
        if cfg["pitch_extractor"] is not None:
            exp_pitch = _exp_table(cfg["pitch_extractor"])
            _print_exp_pitch_table(t, exp_pitch)

    print("\n---\n")
    print("# Sim vs Exp Velocity Error\n")
    print("Percent error = |sim_mean - exp_mean| / exp_mean * 100.")

    # Flat error
    if "flat" in sim_results:
        vx_stats = sim_results["flat"][0]
        exp_vel = _exp_table(extract_flat)
        _print_error_table("flat", vx_stats, exp_vel,
                           extra_exclude={"scene_wheel": [50.0]},
                           extra_exclude_label="Excl. WR f50",
                           note="WR f50 = experimental failure (robot self-destructs at 50Hz).")

    # Step error
    if "step" in sim_results:
        vx_stats = sim_results["step"][0]
        exp_vel = _exp_table(extract_step_q60)
        _print_error_table("step", vx_stats, exp_vel,
                           note="WR f10/f20 excluded (experimental failure). "
                                "L2 f10 (49%) and WR f30 (31%) are the main outliers.")


if __name__ == "__main__":
    main()
