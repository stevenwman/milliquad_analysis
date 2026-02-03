"""
Benchmark headless sim: run one or both scenes with benchmark=True to see
step-level timing (apply_forces, mj_step, record_state, etc.) without
editing sim_optimizer_couple.py.

Conditions match tune_multi_params.py _evaluate_point: same sim_params shape,
same SIM_DURATION, same MJCF paths and order. Params below from low-cost row
(multi_optimization_results.csv id=cea31a43, cost≈0.067) for representative benchmark.

Run from mujoco/:  uv run benchmark_sim.py
"""
import time
import sim_optimizer_couple as sim
from sim_optimizer_couple import MAGNETIC_MOMENT, MAGNETIC_FIELD_MAGNITUDE

# Params from multi_optimization_results.csv row id=cea31a43 (cost≈0.067)
# Build sim_params the same way as tune_multi_params._evaluate_point
_sliding = 0.0010251245028333018
_torsional = 0.0081452228834028
_rolling = 0.05388550972627239
_solref_tc = 0.032264159755217826
_solref_dr = 1.3732811017109758
_solimp_dmin = 0.8786825401722523
_solimp_dmax = 0.967195635410058
_solimp_width = 0.003013064775868027
_moment_fudge = 0.8044633110365063
_field_fudge = 1.1477915883468777
_dof_damping = 8.81101465594395e-10

m_mag = MAGNETIC_MOMENT * _moment_fudge
kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE * _field_fudge

BENCHMARK_SIM_PARAMS = {
    "ground_friction": [_sliding, _torsional, _rolling],
    "solref": [_solref_tc, _solref_dr],
    "solimp": [_solimp_dmin, _solimp_dmax, _solimp_width, 0.5, 1.0],
    "dof_damping": _dof_damping,
    "kp_mag": kp_mag,
    "mag_params": {"m_mag": m_mag},
}

# Same paths and order as tune_multi_params.MJCF_PATHS
MJCF_PATHS = {
    "scene4": "mulit_milli_quad/scene_4.xml",
    "scene2": "mulit_milli_quad/scene_2.xml",
}

# Same as tune_multi_params.SIM_DURATION
SIM_DURATION = 5.0

if __name__ == "__main__":
    print("Benchmarking headless sim (step timing)...")
    print("  Params: multi_optimization_results.csv id=cea31a43 (cost≈0.067)")
    print(f"  sim_duration={SIM_DURATION}s, 2000 Hz => ~{int(SIM_DURATION * 2000)} steps per scene\n")

    for name, mjcf_path in MJCF_PATHS.items():
        print(f"--- {name} ({mjcf_path}) ---")
        t0 = time.perf_counter()
        traj = sim.run_simulation(
            BENCHMARK_SIM_PARAMS,
            mjcf_path=mjcf_path,
            sim_duration=SIM_DURATION,
            visualize=False,
            benchmark=True,
        )
        wall = time.perf_counter() - t0
        print(f"  Total wall (incl. benchmark overhead): {wall:.2f}s")
        print()

    print("Done. Use the step timing above to see where time goes (mj_step vs record_state vs apply_forces).")
    print("Note: During optimization, each worker pays process spawn + import (numpy/mujoco/etc.) once;")
    print("      step loop for both scenes is ~14s, so ~25s per point is spawn/import + sim.")
