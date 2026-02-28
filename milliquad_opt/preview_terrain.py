"""Quick GUI preview of any terrain + morphology combo.

Usage:
    uv run python preview_terrain.py --terrain rough --scene scene4 --freq 30
    uv run python preview_terrain.py --terrain step --scene scene2 --freq 10
    uv run python preview_terrain.py --terrain flat --scene scene_wheel --freq 30
"""
import argparse
from simulation import run_simulation
from config import sim_params_from_point


def main():
    parser = argparse.ArgumentParser(description="Preview terrain in MuJoCo GUI")
    parser.add_argument("--terrain", choices=["flat", "step", "rough"], default="flat")
    parser.add_argument("--scene", default="scene4")
    parser.add_argument("--freq", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    # Load terrain config
    import importlib
    cfg = importlib.import_module(f"config_{args.terrain}")

    if args.scene not in cfg.MJCF_PATHS:
        print(f"Scene '{args.scene}' not in {args.terrain} MJCF_PATHS: {list(cfg.MJCF_PATHS)}")
        return

    params = sim_params_from_point(cfg.CMAES_X0)
    params["drive_freq"] = args.freq

    extra = {}
    if args.terrain == "rough":
        from config_rough import SPAWN_X, SPAWN_Z_RAISE
        extra["spawn_offset"] = (SPAWN_X, 0.0, SPAWN_Z_RAISE)

    run_simulation(
        params,
        mjcf_path=cfg.MJCF_PATHS[args.scene],
        sim_duration=args.duration,
        visualize=True,
        **extra,
    )


if __name__ == "__main__":
    main()
