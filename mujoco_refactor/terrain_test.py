#!/usr/bin/env python3
"""Run best-fit params across flat / step / rough terrain for validation.

Loads params from a completed optimization run, generates terrain-modified
MJCF files, and reports velocity / tumble / lateral metrics per morphology.

Usage:
    uv run python terrain_test.py results/20260222T... --preset step_default
    uv run python terrain_test.py results/20260222T... --preset rough_mild --scenes scene4
    uv run python terrain_test.py results/20260222T... --preset flat --freqs 10 30 50
    uv run python terrain_test.py --list-presets
"""

from __future__ import annotations

import argparse
import csv
import importlib
import pathlib
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

from config_new import (
    DEFAULT_CTRL_FREQ,
    MJCF_PATHS,
    SETTLE_TIME,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    sim_params_from_point,
    space,
)
from optimizer import calculate_cost
from terrain_config import (
    DEFAULT_JITTER_DEG,
    DEFAULT_JITTER_TRIALS,
    JITTER_BASE_SEED,
    TERRAIN_PRESETS,
)

# Add utils/ to path for terrain_mesh import
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "utils"))

SIM_MODULE = "simulation_fast"
PARAM_NAMES = [dim.name for dim in space]


# ---------------------------------------------------------------------------
# Param loading (same pattern as replay_best.py)
# ---------------------------------------------------------------------------

def load_best_point(run_dir: pathlib.Path) -> list[float]:
    """Load full-precision params for the final best from multi CSV."""
    bests_csv = run_dir / "optimization_bests.csv"
    bests_rows = list(csv.DictReader(open(bests_csv)))
    if not bests_rows:
        print(f"ERROR: no rows in {bests_csv}")
        sys.exit(1)
    best_id = bests_rows[-1]["id"]

    multi_csv = run_dir / "multi_optimization_results.csv"
    with open(multi_csv) as f:
        for row in csv.DictReader(f):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    raise ValueError(f"id {best_id!r} not found in {multi_csv}")


# ---------------------------------------------------------------------------
# MJCF terrain editing
# ---------------------------------------------------------------------------

def _inject_steps(xml_path: str, preset: dict, out_xml: str) -> str:
    """Add step box geoms to the MJCF and write to out_xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    flat_lead = preset["flat_lead"]
    step_height = preset["step_height"]
    step_length = preset["step_length"]
    step_count = preset["step_count"]
    final_step_length = preset["final_step_length"]
    step_width = preset["step_width"]

    for i in range(step_count):
        is_final = (i == step_count - 1)
        length = final_step_length if is_final else step_length

        if is_final:
            pos_x = flat_lead + (step_count - 1) * step_length + length / 2.0
        else:
            pos_x = flat_lead + i * step_length + length / 2.0
        pos_z = (i + 1) * step_height - step_height / 2.0

        geom = ET.SubElement(worldbody, "geom")
        geom.set("name", f"step_{i}")
        geom.set("type", "box")
        geom.set("size", f"{length/2.0} {step_width/2.0} {step_height/2.0}")
        geom.set("pos", f"{pos_x} 0.0 {pos_z}")
        geom.set("rgba", "0.5 0.5 0.5 1")

    tree.write(out_xml)
    return out_xml


def _inject_rough(xml_path: str, preset: dict, out_xml: str, tmp_dir: str) -> str:
    """Generate hfield PNG and inject into MJCF. Write to out_xml."""
    from terrain_mesh import generate_terrain_hfield

    # Put the heightmap PNG in the same directory as the output XML so MuJoCo
    # can resolve it, but use tmp_dir for the generation step.
    hfield_base = str(pathlib.Path(tmp_dir) / "terrain")
    heights, (x_half, y_half, z_top, z_bottom) = generate_terrain_hfield(
        nX=preset["grid_nx"],
        nY=preset["grid_ny"],
        sL=preset["tile_size"],
        height_mean=preset["height_mean"],
        height_std=preset["height_std"],
        z_safe=preset["z_safe"],
        seed=preset.get("seed"),
        output_path=hfield_base,
    )
    # MuJoCo requires all hfield size params > 0; generator returns z_bottom=0
    if z_bottom <= 0:
        z_bottom = 0.001

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # Add hfield asset (use absolute path to the PNG so MuJoCo always finds it)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    hfield_png = str(pathlib.Path(hfield_base).with_suffix(".png").resolve())
    hfield_el = ET.SubElement(asset, "hfield")
    hfield_el.set("name", "rough_terrain")
    hfield_el.set("file", hfield_png)
    hfield_el.set("size", f"{x_half} {y_half} {z_top} {z_bottom}")

    # Place terrain geom after flat_lead distance
    flat_lead = preset["flat_lead"]
    pos_x = flat_lead + x_half
    # Surface ranges from z_safe to z_top. Shift down so the average surface
    # height is near z=0 (flush with the flat floor).
    avg_z = (z_top + float(heights.min())) / 2.0
    pos_z = -avg_z

    geom = ET.SubElement(worldbody, "geom")
    geom.set("name", "rough_terrain_geom")
    geom.set("type", "hfield")
    geom.set("hfield", "rough_terrain")
    geom.set("pos", f"{pos_x} 0.0 {pos_z}")
    geom.set("rgba", "0.5 0.5 0.5 1")

    tree.write(out_xml)
    return out_xml


# Track temp XML files written into source dirs for cleanup
_temp_xml_files: list[str] = []


def prepare_mjcf(scene: str, preset: dict, tmp_dir: str) -> str:
    """Return MJCF path — original for flat, edited copy for step/rough.

    For step/rough, writes a temp XML **in the same directory** as the original
    so that MuJoCo can resolve <include>, mesh, and texture relative paths.
    These temp files are tracked in _temp_xml_files for cleanup.
    """
    original = MJCF_PATHS[scene]
    terrain_type = preset["type"]

    if terrain_type == "flat":
        return original

    # Write edited XML alongside the original (same dir = correct relative paths)
    src_dir = pathlib.Path(original).parent
    stem = pathlib.Path(original).stem
    out_xml = str(src_dir / f"{stem}_terrain_tmp.xml")
    _temp_xml_files.append(out_xml)

    if terrain_type == "step":
        return _inject_steps(original, preset, out_xml)
    elif terrain_type == "rough":
        return _inject_rough(original, preset, out_xml, tmp_dir)
    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")


def cleanup_temp_xmls():
    """Remove any temp XML files written into source directories."""
    for path in _temp_xml_files:
        try:
            pathlib.Path(path).unlink(missing_ok=True)
        except OSError:
            pass
    _temp_xml_files.clear()


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_config(
    sim_module,
    sim_params: dict,
    mjcf_path: str,
    freq: float,
    n_trials: int,
    jitter_deg: float,
    config_idx: int,
    sim_duration: float,
) -> list[dict | None]:
    """Run N jitter trials for one (scene, freq) config. Return list of cost_data dicts."""
    sp = dict(sim_params)
    sp["drive_freq"] = freq
    results = []

    for trial_idx in range(n_trials):
        seed = JITTER_BASE_SEED + config_idx * n_trials + trial_idx
        yaw = jitter_deg if n_trials > 1 else 0.0

        traj = sim_module.run_simulation(
            sp,
            mjcf_path=mjcf_path,
            sim_duration=sim_duration,
            wall_timeout=SIMULATION_TIMEOUT,
            init_yaw_jitter_deg=yaw,
            rng_seed=seed,
        )
        if traj is None:
            results.append(None)
        else:
            cd = calculate_cost(traj, target_velocity=1.0, verbose=False)
            results.append(cd)

    return results


def aggregate_trials(trial_results: list[dict | None]) -> dict:
    """Aggregate jitter trials into median/mean/std summary."""
    valid = [r for r in trial_results if r is not None]
    n_total = len(trial_results)
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "vel": None, "vel_std": None, "vel_mean": None,
            "tumble": None, "lateral": None, "yaw": None,
            "n_valid": 0, "n_total": n_total,
        }

    vels = [r["avg_forward_velocity"] for r in valid]
    tumbles = [r["tumble_penalty"] for r in valid]
    lats = [r["lateral_displacement"] for r in valid]
    yaws = [r["yaw_deviation_deg"] for r in valid]

    # Median (same as optimizer convention)
    median_idx = int(np.argsort(vels)[len(vels) // 2])
    return {
        "vel": vels[median_idx],
        "vel_std": float(np.std(vels, ddof=1)) if len(vels) > 1 else 0.0,
        "vel_mean": float(np.mean(vels)),
        "tumble": any(t > 0 for t in tumbles),
        "lateral": lats[median_idx],
        "yaw": yaws[median_idx],
        "n_valid": n_valid,
        "n_total": n_total,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def preset_summary(preset: dict) -> str:
    """One-line description of the preset for the table header."""
    t = preset["type"]
    if t == "flat":
        return "flat"
    elif t == "step":
        h = preset["step_height"] * 1000
        l = preset["step_length"] * 1000
        n = preset["step_count"]
        lead = preset["flat_lead"] * 1000
        return f"step (h={h:.1f}mm, l={l:.1f}mm, n={n}, lead={lead:.1f}mm)"
    elif t == "rough":
        hm = preset["height_mean"] * 1000
        hs = preset["height_std"] * 1000
        s = preset.get("seed", "?")
        lead = preset["flat_lead"] * 1000
        return f"rough (mean={hm:.1f}mm, std={hs:.1f}mm, seed={s}, lead={lead:.1f}mm)"
    return t


def print_results_table(
    configs: list[dict],
    terrain_results: dict[str, dict],
    flat_results: dict[str, dict] | None,
    n_trials: int,
    jitter_deg: float,
    preset: dict,
):
    """Print the summary results table."""
    print()
    print(f"  preset: {preset_summary(preset)}")
    freq_str = f"f={preset.get('ctrl_freq', '?')}Hz"
    if n_trials > 1:
        print(f"  jitter: +/-{jitter_deg}deg yaw, {n_trials} trials, median agg")
    else:
        print(f"  jitter: none (single trial)")
    print()

    has_flat = flat_results is not None and preset["type"] != "flat"

    # Header
    hdr = f"  {'config':<20} {'vel(mm/s)':>9}"
    if n_trials > 1:
        hdr += f" {'std':>6}"
    if has_flat:
        hdr += f" {'flat_vel':>8} {'ratio':>6}"
    hdr += f" {'tumble':>7} {'lat(cm)':>7} {'yaw':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cfg in configs:
        ref_id = cfg["id"]
        r = terrain_results.get(ref_id)
        if r is None or r["vel"] is None:
            row = f"  {ref_id:<20} {'FAIL':>9}"
            if n_trials > 1:
                row += f" {'--':>6}"
            if has_flat:
                row += f" {'--':>8} {'--':>6}"
            row += f" {'--':>7} {'--':>7} {'--':>5}"
            print(row)
            continue

        vel_mm = r["vel"] * 1000
        row = f"  {ref_id:<20} {vel_mm:>8.1f}"

        if n_trials > 1:
            std_mm = r["vel_std"] * 1000
            row += f" {std_mm:>6.1f}"

        if has_flat:
            fr = flat_results.get(ref_id)
            if fr and fr["vel"] is not None and fr["vel"] > 0:
                flat_mm = fr["vel"] * 1000
                ratio = r["vel"] / fr["vel"]
                row += f" {flat_mm:>8.1f} {ratio:>6.2f}"
            else:
                row += f" {'--':>8} {'--':>6}"

        tumble_str = "Y" if r["tumble"] else "N"
        lat_cm = r["lateral"] * 100
        yaw_deg = r["yaw"]
        row += f" {'':>4}{tumble_str:>3} {lat_cm:>6.2f} {yaw_deg:>5.1f}"
        print(row)

    print()


def write_csv(
    out_path: pathlib.Path,
    configs: list[dict],
    terrain_results: dict[str, dict],
    flat_results: dict[str, dict] | None,
    preset: dict,
):
    """Write results to CSV."""
    has_flat = flat_results is not None and preset["type"] != "flat"
    fieldnames = ["config", "scene", "freq_hz", "vel_m_s", "vel_std_m_s", "vel_mean_m_s"]
    if has_flat:
        fieldnames += ["flat_vel_m_s", "ratio"]
    fieldnames += ["tumble", "lateral_m", "yaw_deg", "n_valid", "n_total"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg in configs:
            ref_id = cfg["id"]
            r = terrain_results.get(ref_id, {})
            row = {
                "config": ref_id,
                "scene": cfg["scene"],
                "freq_hz": cfg["freq"],
            }
            if r.get("vel") is not None:
                row["vel_m_s"] = f"{r['vel']:.6f}"
                row["vel_std_m_s"] = f"{r['vel_std']:.6f}"
                row["vel_mean_m_s"] = f"{r['vel_mean']:.6f}"
                row["tumble"] = "Y" if r["tumble"] else "N"
                row["lateral_m"] = f"{r['lateral']:.6f}"
                row["yaw_deg"] = f"{r['yaw']:.2f}"
                row["n_valid"] = r["n_valid"]
                row["n_total"] = r["n_total"]
                if has_flat:
                    fr = flat_results.get(ref_id, {})
                    if fr.get("vel") is not None and fr["vel"] > 0:
                        row["flat_vel_m_s"] = f"{fr['vel']:.6f}"
                        row["ratio"] = f"{r['vel'] / fr['vel']:.4f}"
            else:
                row["vel_m_s"] = "FAIL"
                row["n_valid"] = r.get("n_valid", 0)
                row["n_total"] = r.get("n_total", 0)
            writer.writerow(row)
    print(f"  CSV saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Terrain validation: run best-fit params on flat/step/rough terrain",
    )
    parser.add_argument("run_dir", nargs="?", type=str,
                        help="Results directory with optimization CSVs")
    parser.add_argument("--preset", type=str, default="flat",
                        help="Terrain preset name from terrain_config.py")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available presets and exit")

    # Config filters
    parser.add_argument("--scenes", nargs="+", type=str, default=None,
                        help="Filter morphologies (e.g. scene1 scene4)")
    parser.add_argument("--freqs", nargs="+", type=float, default=None,
                        help="Override preset ctrl_freq with specific frequencies")

    # Jitter
    parser.add_argument("--jitter-trials", type=int, default=None,
                        help=f"Trials per config (default: {DEFAULT_JITTER_TRIALS})")
    parser.add_argument("--jitter-deg", type=float, default=None,
                        help=f"Yaw jitter degrees (default: {DEFAULT_JITTER_DEG})")

    # Terrain param overrides
    parser.add_argument("--step-height", type=float, default=None)
    parser.add_argument("--step-length", type=float, default=None)
    parser.add_argument("--step-count", type=int, default=None)
    parser.add_argument("--final-step-length", type=float, default=None)
    parser.add_argument("--flat-lead", type=float, default=None)
    parser.add_argument("--height-mean", type=float, default=None)
    parser.add_argument("--height-std", type=float, default=None)
    parser.add_argument("--tile-size", type=float, default=None)
    parser.add_argument("--grid-nx", type=int, default=None)
    parser.add_argument("--grid-ny", type=int, default=None)
    parser.add_argument("--z-safe", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Output
    parser.add_argument("--record", action="store_true",
                        help="Save mp4 videos for median trial per config")
    parser.add_argument("--csv", action="store_true",
                        help="Write results CSV to run_dir")
    parser.add_argument("--duration", type=float, default=None,
                        help=f"Sim duration override (default: {SIM_DURATION}s)")

    args = parser.parse_args()

    # -- List presets --
    if args.list_presets:
        print("Available terrain presets:")
        for name, p in TERRAIN_PRESETS.items():
            print(f"  {name:<20} {preset_summary(p)}")
        return

    if args.run_dir is None:
        parser.error("run_dir is required (or use --list-presets)")

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    # -- Build preset (base + CLI overrides) --
    if args.preset not in TERRAIN_PRESETS:
        print(f"ERROR: unknown preset '{args.preset}'")
        print("Available:", ", ".join(TERRAIN_PRESETS.keys()))
        sys.exit(1)

    preset = dict(TERRAIN_PRESETS[args.preset])  # shallow copy

    # Apply CLI overrides
    cli_overrides = {
        "step_height": args.step_height,
        "step_length": args.step_length,
        "step_count": args.step_count,
        "final_step_length": args.final_step_length,
        "flat_lead": args.flat_lead,
        "height_mean": args.height_mean,
        "height_std": args.height_std,
        "tile_size": args.tile_size,
        "grid_nx": args.grid_nx,
        "grid_ny": args.grid_ny,
        "z_safe": args.z_safe,
        "seed": args.seed,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            preset[key] = val

    n_trials = args.jitter_trials if args.jitter_trials is not None else DEFAULT_JITTER_TRIALS
    jitter_deg = args.jitter_deg if args.jitter_deg is not None else DEFAULT_JITTER_DEG
    sim_duration = args.duration if args.duration is not None else SIM_DURATION

    # -- Build config list (scene, freq pairs) --
    scenes = list(MJCF_PATHS.keys())
    if args.scenes:
        scenes = [s for s in args.scenes if s in MJCF_PATHS]
        if not scenes:
            print(f"ERROR: no valid scenes in {args.scenes}")
            print("Available:", ", ".join(MJCF_PATHS.keys()))
            sys.exit(1)

    if args.freqs:
        freqs = args.freqs
    else:
        freqs = [preset.get("ctrl_freq", DEFAULT_CTRL_FREQ)]

    configs = []
    for scene in scenes:
        for freq in freqs:
            ref_id = f"{scene}_f{int(freq)}"
            configs.append({"id": ref_id, "scene": scene, "freq": freq})

    # -- Load params --
    print(f"Loading best params from {run_dir} ...")
    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)
    sim_module = importlib.import_module(SIM_MODULE)

    total_sims = len(configs) * n_trials
    terrain_type = preset["type"]
    if terrain_type != "flat":
        total_sims *= 2  # flat baseline + terrain
    print(f"  {len(configs)} configs x {n_trials} trials = {total_sims} sims")
    print(f"  preset: {preset_summary(preset)}")
    print()

    # -- Run sims --
    # tmp_dir is only for hfield PNGs; edited XMLs go alongside originals
    # and are cleaned up via cleanup_temp_xmls().
    try:
        with tempfile.TemporaryDirectory(prefix="terrain_test_") as tmp_dir:
            # Run flat baseline (for ratio column when terrain != flat)
            flat_results: dict[str, dict] | None = None
            if terrain_type != "flat":
                flat_preset = {"type": "flat"}
                print("Running flat baseline ...")
                flat_results = {}
                for ci, cfg in enumerate(configs):
                    mjcf = prepare_mjcf(cfg["scene"], flat_preset, tmp_dir)
                    trials = run_config(
                        sim_module, sim_params, mjcf, cfg["freq"],
                        n_trials, jitter_deg, ci, sim_duration,
                    )
                    flat_results[cfg["id"]] = aggregate_trials(trials)
                    vel = flat_results[cfg["id"]]["vel"]
                    vel_str = f"{vel*1000:.1f}mm/s" if vel is not None else "FAIL"
                    print(f"  {cfg['id']:<20} {vel_str}")
                print()

            # Run terrain
            print(f"Running {terrain_type} terrain ...")
            terrain_results: dict[str, dict] = {}
            t0_all = time.perf_counter()

            for ci, cfg in enumerate(configs):
                mjcf = prepare_mjcf(cfg["scene"], preset, tmp_dir)
                t0 = time.perf_counter()
                trials = run_config(
                    sim_module, sim_params, mjcf, cfg["freq"],
                    n_trials, jitter_deg, ci, sim_duration,
                )
                elapsed = time.perf_counter() - t0
                agg = aggregate_trials(trials)
                terrain_results[cfg["id"]] = agg
                vel = agg["vel"]
                vel_str = f"{vel*1000:.1f}mm/s" if vel is not None else "FAIL"
                print(f"  {cfg['id']:<20} {vel_str:>12} ({elapsed:.1f}s)")

            elapsed_all = time.perf_counter() - t0_all
            print(f"\n  Total terrain time: {elapsed_all:.1f}s")

            # -- Record videos for median trial (optional) --
            if args.record:
                vid_dir = run_dir / "terrain_videos"
                vid_dir.mkdir(exist_ok=True)
                print(f"\nRecording videos to {vid_dir} ...")
                for ci, cfg in enumerate(configs):
                    agg = terrain_results[cfg["id"]]
                    if agg["vel"] is None:
                        continue
                    sp = dict(sim_params)
                    sp["drive_freq"] = cfg["freq"]
                    yaw = jitter_deg if n_trials > 1 else 0.0
                    mjcf = prepare_mjcf(cfg["scene"], preset, tmp_dir)

                    # For single trial, just re-record it directly
                    if n_trials == 1:
                        median_seed = JITTER_BASE_SEED + ci * n_trials
                    else:
                        # Find median trial index by re-running headless
                        trial_vels = []
                        for ti in range(n_trials):
                            seed = JITTER_BASE_SEED + ci * n_trials + ti
                            traj = sim_module.run_simulation(
                                sp, mjcf_path=mjcf, sim_duration=sim_duration,
                                wall_timeout=SIMULATION_TIMEOUT,
                                init_yaw_jitter_deg=yaw, rng_seed=seed,
                            )
                            v = calculate_cost(traj, 1.0, verbose=False)["avg_forward_velocity"] if traj else -1e6
                            trial_vels.append((ti, v))
                        trial_vels.sort(key=lambda x: x[1])
                        median_ti = trial_vels[len(trial_vels) // 2][0]
                        median_seed = JITTER_BASE_SEED + ci * n_trials + median_ti

                    vid_path = str(vid_dir / f"{cfg['id']}_{args.preset}.mp4")
                    sim_module.run_simulation(
                        sp, mjcf_path=mjcf, sim_duration=sim_duration,
                        wall_timeout=SIMULATION_TIMEOUT,
                        record_path=vid_path,
                        init_yaw_jitter_deg=yaw, rng_seed=median_seed,
                    )
                    print(f"  Saved: {vid_path}")
    finally:
        cleanup_temp_xmls()

    # -- Print results table --
    print_results_table(
        configs, terrain_results, flat_results,
        n_trials, jitter_deg, preset,
    )

    # -- Write CSV (optional) --
    if args.csv:
        csv_path = run_dir / f"terrain_results_{args.preset}.csv"
        write_csv(csv_path, configs, terrain_results, flat_results, preset)


if __name__ == "__main__":
    main()
