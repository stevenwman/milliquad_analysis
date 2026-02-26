#!/usr/bin/env python3
"""Verify that REFERENCE_DATA velocities match experimental data processing."""

import sys
from pathlib import Path
import numpy as np

# Add experimental_data/scripts to path
scripts_dir = Path(__file__).parent.parent / "experimental_data" / "scripts"
sys.path.insert(0, str(scripts_dir))

from pipeline_common import MM_SCALE, load_trial_csv

# Scene name mapping
SCENE_MAP = {
    "scene1": "leg",
    "scene2": "2leg",
    "scene4": "4leg",
    "scene_wheel": "w",
}

def compute_flat_velocity(freq: int, scene_name: str, steady_t: float) -> tuple[float, float]:
    """Compute steady-state mean velocity for flat terrain."""
    scene_code = SCENE_MAP[scene_name]

    # Find CSV files
    csv_dir = Path(__file__).parent.parent / "experimental_data" / "csv" / "flat"
    pattern = f"f{freq}{scene_code}*.csv"
    files = sorted(csv_dir.glob(pattern))

    if not files:
        return None, None

    vx_steady_values = []
    for file_path in files:
        dat = load_trial_csv(str(file_path))
        t = dat[:, 0]
        vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))

        idx_steady = t > steady_t
        if np.any(idx_steady):
            vx_steady_mean = np.nanmean(vx[idx_steady])
            vx_steady_values.append(vx_steady_mean)

    if vx_steady_values:
        mean_vel = np.nanmean(vx_steady_values)
        std_vel = np.nanstd(vx_steady_values, ddof=0)
        return mean_vel, std_vel
    return None, None

def compute_step_velocity_q75_300(freq: int, scene_name: str) -> tuple[float, float]:
    """Compute q75-300 forward velocity for step terrain."""
    scene_code = SCENE_MAP[scene_name]

    # Find CSV files
    csv_dir = Path(__file__).parent.parent / "experimental_data" / "csv" / "steps"
    pattern = f"s{freq}{scene_code}*.csv"
    files = sorted(csv_dir.glob(pattern))

    if not files:
        return None, None

    vx_q75_values = []
    for file_path in files:
        dat = load_trial_csv(str(file_path))
        t = dat[:, 0]
        vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))

        # q75-300: 75% index +/- 150 timesteps, clamped
        n = len(vx)
        q75_idx = int(0.75 * n)
        start_idx = max(0, q75_idx - 150)
        end_idx = min(n, q75_idx + 150)

        if start_idx < end_idx:
            vx_window = vx[start_idx:end_idx]
            vx_mean = np.nanmean(vx_window)
            vx_q75_values.append(vx_mean)

    if vx_q75_values:
        mean_vel = np.nanmean(vx_q75_values)
        std_vel = np.nanstd(vx_q75_values, ddof=0)
        return mean_vel, std_vel
    return None, None

# Steady time mapping from flat_pipeline.py build_conditions()
STEADY_TIME_MAP = {
    ("scene1", 10): 0.3,
    ("scene1", 30): 0.15,
    ("scene1", 50): 0.35,
    ("scene2", 10): 0.3,
    ("scene2", 30): 0.3,
    ("scene2", 50): 0.35,
    ("scene4", 10): 0.3,
    ("scene4", 30): 0.3,
    ("scene4", 50): 0.35,
    ("scene_wheel", 10): 0.3,
    ("scene_wheel", 20): 0.3,
    ("scene_wheel", 30): 0.3,
    ("scene_wheel", 50): 0.25,
}

def verify_flat_references():
    """Verify flat terrain reference velocities."""
    from config_multi_terrain import REFERENCE_DATA

    flat_refs = [r for r in REFERENCE_DATA if r["terrain"] == "flat"]

    print("=" * 100)
    print("FLAT TERRAIN VERIFICATION")
    print("=" * 100)
    print(f"{'Scene':<12} {'Freq':<6} {'Config':<12} {'Computed':<12} {'Diff':<10} {'Status'}")
    print("-" * 100)

    errors = []
    for ref in flat_refs:
        scene = ref["scene"]
        freq = int(ref["ctrl_freq"])
        config_speed = ref["speed"] * 1000  # Convert to mm/s for display

        steady_t = STEADY_TIME_MAP.get((scene, freq), 0.3)
        computed_speed, computed_std = compute_flat_velocity(freq, scene, steady_t)

        if computed_speed is None:
            print(f"{scene:<12} {freq:<6} {config_speed:>10.1f} {'NO DATA':<12} {'N/A':<10} ERROR")
            errors.append(f"{scene} f{freq}: NO DATA")
        else:
            diff = abs(computed_speed - config_speed)
            diff_pct = 100 * diff / computed_speed if computed_speed > 0 else 0
            status = "✓ OK" if diff < 1.0 else "✗ MISMATCH"

            print(f"{scene:<12} {freq:<6} {config_speed:>10.1f} {computed_speed:>10.1f} {diff:>8.1f} {status}")

            if diff >= 1.0:
                errors.append(f"{scene} f{freq}: config={config_speed:.1f}, computed={computed_speed:.1f}, diff={diff:.1f}mm/s ({diff_pct:.1f}%)")

    return errors

def verify_step_references():
    """Verify step terrain reference velocities."""
    from config_multi_terrain import REFERENCE_DATA

    step_refs = [r for r in REFERENCE_DATA if r["terrain"] == "step"]

    print()
    print("=" * 100)
    print("STEP TERRAIN VERIFICATION (q75-300 window)")
    print("=" * 100)
    print(f"{'Scene':<12} {'Freq':<6} {'Config':<12} {'Computed':<12} {'Diff':<10} {'Status'}")
    print("-" * 100)

    errors = []
    for ref in step_refs:
        scene = ref["scene"]
        freq = int(ref["ctrl_freq"])
        config_speed = ref["speed"] * 1000  # Convert to mm/s for display

        computed_speed, computed_std = compute_step_velocity_q75_300(freq, scene)

        if computed_speed is None:
            print(f"{scene:<12} {freq:<6} {config_speed:>10.1f} {'NO DATA':<12} {'N/A':<10} ERROR")
            errors.append(f"{scene} f{freq}: NO DATA")
        else:
            diff = abs(computed_speed - config_speed)
            diff_pct = 100 * diff / computed_speed if computed_speed > 0 else 0
            status = "✓ OK" if diff < 1.0 else "✗ MISMATCH"

            print(f"{scene:<12} {freq:<6} {config_speed:>10.1f} {computed_speed:>10.1f} {diff:>8.1f} {status}")

            if diff >= 1.0:
                errors.append(f"{scene} f{freq}: config={config_speed:.1f}, computed={computed_speed:.1f}, diff={diff:.1f}mm/s ({diff_pct:.1f}%)")

    return errors

def main():
    flat_errors = verify_flat_references()
    step_errors = verify_step_references()

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    if not flat_errors and not step_errors:
        print("✓ All reference velocities match experimental data!")
    else:
        print(f"✗ Found {len(flat_errors) + len(step_errors)} mismatches:")
        print()
        if flat_errors:
            print("FLAT TERRAIN ERRORS:")
            for err in flat_errors:
                print(f"  - {err}")
        if step_errors:
            print()
            print("STEP TERRAIN ERRORS:")
            for err in step_errors:
                print(f"  - {err}")
        print()
        print("ACTION REQUIRED: Fix REFERENCE_DATA in config_multi_terrain.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
