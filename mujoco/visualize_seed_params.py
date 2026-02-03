"""
Visualize the supposedly optimized params from the old single-scene opt
(optimized_params/optimization_results.csv row abe1b74c). Uses fudges=1, damping=7e-10.
Run from mujoco/:  uv run visualize_seed_params.py [--scene scene4|scene2] [--record out.mp4] [--duration 10]
"""
import argparse
import sim_optimizer_couple as sim_optimizer
from sim_optimizer_couple import MAGNETIC_MOMENT, MAGNETIC_FIELD_MAGNITUDE

# Old opt row abe1b74c (cost ~0.00053, avg_velocity ~0.233). No fudges/damping in that CSV; assume fudge=1, damping=7e-10.
SEED_FRICTION = [0.00014225746640521907, 0.0021388784110800154, 5.292387847485097e-05]
SEED_SOLREF = [0.001, 0.7414912155887285]
SEED_SOLIMP = [0.9084351427617432, 0.9734506827063522, 0.0037927813470769885, 0.5, 1.0]
SEED_DOF_DAMPING = 7e-10
MOMENT_FUDGE = 1.0
FIELD_FUDGE = 1.0

MJCF_PATHS = {
    "scene4": "mulit_milli_quad/scene_4.xml",
    "scene2": "mulit_milli_quad/scene_2.xml",
}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize rollout using old-opt seed params (abe1b74c, fudges=1, damping=7e-10)."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene4",
        choices=list(MJCF_PATHS.keys()),
        help="Scene to run (scene4 or scene2).",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="If set, record video to this path instead of interactive viewer.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds.",
    )
    args = parser.parse_args()

    m_mag = MAGNETIC_MOMENT * MOMENT_FUDGE
    kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE * FIELD_FUDGE

    sim_params = {
        "ground_friction": SEED_FRICTION,
        "solref": SEED_SOLREF,
        "solimp": SEED_SOLIMP,
        "dof_damping": SEED_DOF_DAMPING,
        "kp_mag": kp_mag,
        "mag_params": {"m_mag": m_mag},
    }

    mjcf_path = MJCF_PATHS[args.scene]
    print(f"--- Seed params (old opt abe1b74c, fudges=1, damping=7e-10) ---")
    print(f"  scene: {args.scene} ({mjcf_path})")
    print(f"  duration: {args.duration}s")
    print(f"  friction: {sim_params['ground_friction']}")
    print(f"  solref: {sim_params['solref']}, solimp (dmin,dmax,width): {sim_params['solimp'][:3]}")
    print(f"  kp_mag: {kp_mag:.4e}, m_mag: {m_mag:.4e}\n")

    if args.record:
        print(f"Recording to {args.record}...")
        sim_optimizer.run_simulation(
            sim_params,
            mjcf_path=mjcf_path,
            sim_duration=args.duration,
            record_path=args.record,
            ignore_stuck_detection=True,
        )
    else:
        print("Launching viewer (SPACE = play/pause)...")
        sim_optimizer.run_simulation(
            sim_params,
            mjcf_path=mjcf_path,
            sim_duration=args.duration,
            visualize=True,
            ignore_stuck_detection=True,
        )


if __name__ == "__main__":
    main()
