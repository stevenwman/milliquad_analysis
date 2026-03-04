"""Estimate lower-bound COT from experimental kinematics.

Lower bound: robot starts from rest, so total kinetic energy gained
is the minimum the actuator must have supplied:

    E_lower = KE_trans + KE_rot_legs + KE_rot_body
    COT_lower = E_lower / (m * g * d_forward)

Evaluated at the end of the measurement window. PE is excluded —
on flat terrain y is lateral (not height), and on steps the height
gain just reflects climbing, not locomotion efficiency.

CSV columns:
    col 0:  t        (s)
    col 1:  mass_A x (m)      col 2: mass_A y (m)
    col 3:  mass_A vx (m/s)   col 4: mass_A vy (m/s)
    col 5:  mass_C x (m)      col 6: mass_C y (m)
    col 7:  mass_C vx (m/s)   col 8: mass_C vy (m/s)
    col 9:  mass_B theta (deg, cumulative leg rotation)
    col 10: mass_B omega (deg/s, leg angular velocity)
    col 11: mass_C theta (deg, body pitch)

For step terrain: y = height (side-view camera), delta ~8mm over 8 steps.
For flat terrain: y = lateral (top-down camera), delta ~0.

Usage:
    python cot_lower_bound.py [--terrain flat|step]
"""

import sys
from pathlib import Path

import numpy as np

from cot_upper_bound import (
    FLAT_CONDITIONS,
    STEP_CONDITIONS,
    CSV_DIR_FLAT,
    CSV_DIR_STEP,
    ROBOT_MASS,
    _sanity_check_csv,
)

# ---------------------------------------------------------------------------
# Physical constants from MJCF
# ---------------------------------------------------------------------------
G = 9.81
N_LEGS = 4

# Leg rotational inertia about joint axis (Izz, hinge axis = [0,0,1])
# From model.body_inertia[leg_body][2] for each morphology
I_LEG = {
    "leg":   4.1935e-12,  # scene1
    "2leg":  3.4764e-12,  # scene2
    "4leg":  6.9923e-12,  # scene4
    "wheel": 2.5894e-12,  # scene_wheel
}

# Body inertia about pitch axis (Iyy — rotation in the xz/side-view plane)
I_BODY_PITCH = 3.1387e-10  # same body across all morphologies


def compute_trial_cot_lower(csv_path: Path, morph: str, terrain: str) -> dict | None:
    """Compute COT lower bound for a single trial."""
    dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    t = dat[:, 0]
    n = len(t)

    # Windowing: last 50% for flat, q60 for step
    if terrain == "flat":
        lo = n // 2
        hi = n - 1
    else:
        lo = int(0.45 * n)
        hi = int(0.75 * n) - 1

    if hi <= lo + 10:
        return None

    mass = ROBOT_MASS[morph]
    i_leg = I_LEG[morph]

    # --- Find last valid row for velocity (tracker may lose marker at end) ---
    # Need at least one of (vx_A, vy_A) or (vx_C, vy_C) to be valid
    for hi_v in range(hi, lo, -1):
        vA = dat[hi_v, 3]**2 + dat[hi_v, 4]**2
        vC = dat[hi_v, 7]**2 + dat[hi_v, 8]**2
        if not (np.isnan(vA) and np.isnan(vC)):
            break
    else:
        return None  # no valid velocity in window

    # --- Velocities at adjusted window end ---
    v_sq = np.nanmean([
        dat[hi_v, 3]**2 + dat[hi_v, 4]**2,
        dat[hi_v, 7]**2 + dat[hi_v, 8]**2,
    ])
    ke_trans = 0.5 * mass * v_sq

    # Leg rotation (use hi_v as endpoint)
    omega_deg = dat[hi_v, 10]
    if np.isnan(omega_deg):
        valid = dat[lo:hi_v+1, 10]
        valid = valid[~np.isnan(valid)]
        if len(valid) == 0:
            return None
        omega_deg = valid[-1]
    omega_rad = np.deg2rad(omega_deg)
    ke_rot_legs = N_LEGS * 0.5 * i_leg * omega_rad**2

    # Body pitch rotation
    pitch = dat[:, 11]  # degrees
    dt = np.median(np.diff(t))
    # d(pitch)/dt at hi_v using central difference
    if hi_v + 1 < n and not np.isnan(pitch[hi_v + 1]) and not np.isnan(pitch[hi_v - 1]):
        dpitch_dt = (pitch[hi_v + 1] - pitch[hi_v - 1]) / (2 * dt)
    elif hi_v > 0 and not np.isnan(pitch[hi_v]) and not np.isnan(pitch[hi_v - 1]):
        dpitch_dt = (pitch[hi_v] - pitch[hi_v - 1]) / dt
    else:
        dpitch_dt = 0.0
    dpitch_dt_rad = np.deg2rad(dpitch_dt)
    ke_rot_body = 0.5 * I_BODY_PITCH * dpitch_dt_rad**2

    ke_total = ke_trans + ke_rot_legs + ke_rot_body

    e_lower = ke_total

    # --- Forward displacement ---
    x_start = -np.nanmean([dat[lo, 1], dat[lo, 5]])  # negated (camera frame)
    x_end = -np.nanmean([dat[hi_v, 1], dat[hi_v, 5]])
    d_forward = x_end - x_start  # m
    if d_forward < 1e-6:
        return None

    cot_lower = e_lower / (mass * G * d_forward)

    return {
        "cot_lower": cot_lower,
        "ke_trans_uJ": ke_trans * 1e6,
        "ke_rot_legs_uJ": ke_rot_legs * 1e6,
        "ke_rot_body_uJ": ke_rot_body * 1e6,
        "ke_total_uJ": ke_total * 1e6,
        "d_forward_mm": d_forward * 1000,
        "v_end_mm_s": np.sqrt(v_sq) * 1000,
        "omega_end_deg_s": omega_deg,
    }


def main():
    terrain = "flat"
    if "--terrain" in sys.argv:
        idx = sys.argv.index("--terrain")
        terrain = sys.argv[idx + 1]

    conditions = FLAT_CONDITIONS if terrain == "flat" else STEP_CONDITIONS
    csv_dir = CSV_DIR_FLAT if terrain == "flat" else CSV_DIR_STEP

    print(f"=== COT Lower Bound ({terrain} terrain) ===")
    print()

    # Sanity check
    print("Sanity checking CSV column layouts...")
    for freq, morph, files in conditions:
        for fname in files:
            p = csv_dir / fname
            if p.exists():
                _sanity_check_csv(p)
    print("All CSVs pass.\n")

    header = (
        f"{'Morph':<8} {'Freq':>4} {'Trial':>5}  "
        f"{'COT_lb':>8} {'KE_t(uJ)':>9} {'KE_rl(uJ)':>10} {'KE_rb(uJ)':>10} "
        f"{'KE_tot(uJ)':>11} {'d(mm)':>7}"
    )
    print(header)
    print("-" * len(header))

    for freq, morph, files in conditions:
        for ti, fname in enumerate(files):
            p = csv_dir / fname
            if not p.exists():
                print(f"{morph:<8} {freq:>4} {ti+1:>5}  {'MISSING':>8}")
                continue
            r = compute_trial_cot_lower(p, morph, terrain)
            if r is None:
                print(f"{morph:<8} {freq:>4} {ti+1:>5}  {'SKIP':>8}")
                continue
            print(
                f"{morph:<8} {freq:>4} {ti+1:>5}  "
                f"{r['cot_lower']:8.4f} "
                f"{r['ke_trans_uJ']:9.3f} "
                f"{r['ke_rot_legs_uJ']:10.6f} "
                f"{r['ke_rot_body_uJ']:10.6f} "
                f"{r['ke_total_uJ']:11.3f} "
                f"{r['d_forward_mm']:7.1f}"
            )

    # Summary
    print(f"\n{'=== Summary: mean COT lower bound per condition ===':}")
    print(f"{'Morph':<8} {'Freq':>4} {'COT_lb':>10} {'n':>3}")
    print("-" * 30)
    for freq, morph, files in conditions:
        cots = []
        for fname in files:
            p = csv_dir / fname
            if not p.exists():
                continue
            r = compute_trial_cot_lower(p, morph, terrain)
            if r is not None:
                cots.append(r["cot_lower"])
        if cots:
            print(f"{morph:<8} {freq:>4} {np.mean(cots):10.4f} {len(cots):>3}")
        else:
            print(f"{morph:<8} {freq:>4} {'n/a':>10} {0:>3}")


if __name__ == "__main__":
    main()
