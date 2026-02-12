"""
Verify simulation_fast.py produces identical trajectories to simulation.py.

Runs both implementations with the same params and seeds, compares the
floating-base trajectory (pos, vel, quat) at every timestep.

Usage:
    cd mujoco_refactor
    uv run python verify_sim.py
"""

import time

import numpy as np

from config import (
    DEFAULT_CTRL_FREQ,
    MAGNETIC_MOMENT,
    MJCF_PATHS,
    SIM_DURATION,
    sim_params_from_point,
    space,
)
import simulation
import simulation_fast


def _random_point(rng: np.random.Generator) -> list[float]:
    """Generate a random point within the search space bounds."""
    point = []
    for dim in space:
        if dim.prior == "log-uniform":
            val = np.exp(rng.uniform(np.log(dim.low), np.log(dim.high)))
        else:
            val = rng.uniform(dim.low, dim.high)
        point.append(val)
    return point


def _compare_trajectories(traj_a, traj_b, label: str) -> dict:
    """Compare two trajectories. Returns dict of max differences."""
    if traj_a is None and traj_b is None:
        print(f"  {label}: Both None (sim failure) — match")
        return {"match": True, "both_none": True}

    if (traj_a is None) != (traj_b is None):
        print(f"  {label}: One None, other not — MISMATCH")
        return {"match": False, "one_none": True}

    if len(traj_a) != len(traj_b):
        print(f"  {label}: Length mismatch ({len(traj_a)} vs {len(traj_b)})")
        return {"match": False, "len_a": len(traj_a), "len_b": len(traj_b)}

    max_pos = 0.0
    max_vel = 0.0
    max_quat = 0.0
    first_diverge_step = None

    for i, (a, b) in enumerate(zip(traj_a, traj_b)):
        dp = np.max(np.abs(a["pos"] - b["pos"]))
        dv = np.max(np.abs(a["vel"] - b["vel"]))
        dq = np.max(np.abs(a["quat"] - b["quat"]))
        max_pos = max(max_pos, dp)
        max_vel = max(max_vel, dv)
        max_quat = max(max_quat, dq)
        if first_diverge_step is None and (dp > 0 or dv > 0 or dq > 0):
            first_diverge_step = i

    exact = max_pos == 0.0 and max_vel == 0.0 and max_quat == 0.0
    status = "EXACT" if exact else f"max Δ: pos={max_pos:.2e} vel={max_vel:.2e} quat={max_quat:.2e}"
    if first_diverge_step is not None and not exact:
        status += f" (first diff at step {first_diverge_step}/{len(traj_a)})"
    print(f"  {label} ({len(traj_a)} steps): {status}")

    return {
        "match": exact,
        "max_pos": max_pos,
        "max_vel": max_vel,
        "max_quat": max_quat,
        "steps": len(traj_a),
        "first_diverge": first_diverge_step,
    }


def _run_one_trial(sim_params: dict, scene_name: str, label: str, seed: int = 42):
    """Run both implementations and compare."""
    mjcf_path = MJCF_PATHS[scene_name]

    t0 = time.perf_counter()
    traj_orig = simulation.run_simulation(
        sim_params,
        mjcf_path=mjcf_path,
        sim_duration=SIM_DURATION,
        visualize=False,
        rng_seed=seed,
        init_yaw_jitter_deg=2.0,
    )
    t_orig = time.perf_counter() - t0

    t0 = time.perf_counter()
    traj_fast = simulation_fast.run_simulation(
        sim_params,
        mjcf_path=mjcf_path,
        sim_duration=SIM_DURATION,
        visualize=False,
        rng_seed=seed,
        init_yaw_jitter_deg=2.0,
    )
    t_fast = time.perf_counter() - t0

    print(f"\n  [{label}] scene={scene_name} seed={seed}")
    print(f"    orig: {t_orig:.2f}s | fast: {t_fast:.2f}s | speedup: {t_orig/t_fast:.2f}x")
    result = _compare_trajectories(traj_orig, traj_fast, f"    {label}")
    result["t_orig"] = t_orig
    result["t_fast"] = t_fast
    return result


def main():
    rng = np.random.default_rng(99)

    # Trial 0: default params (known good)
    default_params = {
        "ground_friction": [1e-5, 1e-5, 1e-5],
        "dof_damping": 7e-10,
        "solref": [0.004, 1],
        "solimp": [0.95, 0.99, 1e-3, 0.5, 1.0],
        "kp_mag": 2.5e-6,
        "drive_freq": DEFAULT_CTRL_FREQ,
        "mag_params": {"m_mag": MAGNETIC_MOMENT},
    }

    all_results = []

    print("=" * 70)
    print("Trial 0: Default params")
    print("=" * 70)
    for scene in MJCF_PATHS:
        r = _run_one_trial(default_params, scene, f"default/{scene}", seed=42)
        all_results.append(r)

    # Trials 1-4: random params
    for trial in range(1, 5):
        point = _random_point(rng)
        sim_params = sim_params_from_point(point)
        sim_params["drive_freq"] = DEFAULT_CTRL_FREQ
        seed = rng.integers(0, 100000)

        print(f"\n{'=' * 70}")
        print(f"Trial {trial}: Random params (seed={seed})")
        print(f"{'=' * 70}")
        for scene in MJCF_PATHS:
            r = _run_one_trial(sim_params, scene, f"trial{trial}/{scene}", seed=int(seed))
            all_results.append(r)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    n_exact = sum(1 for r in all_results if r.get("match"))
    n_total = len(all_results)
    print(f"  Exact matches: {n_exact}/{n_total}")

    t_orig_total = sum(r.get("t_orig", 0) for r in all_results)
    t_fast_total = sum(r.get("t_fast", 0) for r in all_results)
    if t_fast_total > 0:
        print(f"  Total time: orig={t_orig_total:.1f}s fast={t_fast_total:.1f}s speedup={t_orig_total/t_fast_total:.2f}x")

    worst_pos = max((r.get("max_pos", 0) for r in all_results), default=0)
    worst_vel = max((r.get("max_vel", 0) for r in all_results), default=0)
    worst_quat = max((r.get("max_quat", 0) for r in all_results), default=0)
    if worst_pos > 0 or worst_vel > 0 or worst_quat > 0:
        print(f"  Worst diffs: pos={worst_pos:.2e} vel={worst_vel:.2e} quat={worst_quat:.2e}")


if __name__ == "__main__":
    main()
