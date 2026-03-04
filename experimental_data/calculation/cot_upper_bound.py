"""Estimate upper-bound COT from experimental kinematics.

Upper bound assumes max magnetic torque at all times:
    tau_max = MAGNETIC_MOMENT * MAGNETIC_FIELD_MAGNITUDE  (per leg)
    P_max   = 4 * tau_max * |omega|                       (4 legs, same drive)
    COT_max = mean(P_max) / (m * g * v_avg)

Experimental CSV columns (verified consistent across all files):
    col 0:  t       (s, dt=1ms)
    col 1:  mass_A x    (m)
    col 2:  mass_A y    (m)
    col 3:  mass_A vx   (m/s, camera frame — negate for forward)
    col 4:  mass_A vy   (m/s)
    col 5:  mass_C x    (m)
    col 6:  mass_C y    (m)
    col 7:  mass_C vx   (m/s, camera frame — negate for forward)
    col 8:  mass_C vy   (m/s)
    col 9:  mass_B theta (deg, cumulative leg rotation)
    col 10: mass_B omega (deg/s, leg angular velocity)
    col 11: mass_C theta (deg, body pitch)

Usage:
    python cot_upper_bound.py [--terrain flat|step]
"""

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MAGNETIC_MOMENT = 1.13e-3          # A·m² per magnet (1 magnet per leg)
MAGNETIC_FIELD_MAGNITUDE = 2e-3    # T
TAU_MAX = MAGNETIC_MOMENT * MAGNETIC_FIELD_MAGNITUDE  # N·m per leg
N_LEGS = 4
G = 9.81  # m/s²

# Robot masses from MJCF (kg)
ROBOT_MASS = {
    "leg":   0.000103,
    "2leg":  0.000105,
    "4leg":  0.000109,
    "wheel": 0.000091,
}

# ---------------------------------------------------------------------------
# CSV file registry
# ---------------------------------------------------------------------------
CSV_DIR_FLAT = Path(__file__).resolve().parent.parent / "csv" / "flat"
CSV_DIR_STEP = Path(__file__).resolve().parent.parent / "csv" / "steps"

FLAT_CONDITIONS = [
    (10, "leg",   ["f10leg1-1.csv", "f10leg2-2.csv", "f10leg3-3.csv"]),
    (10, "2leg",  ["f102leg1-1.csv", "f102leg2-2.csv", "f102leg3-3.csv"]),
    (10, "4leg",  ["f104leg1-1.csv", "f104leg2-2.csv", "f104leg3-3.csv"]),
    (10, "wheel", ["f10w1-1.csv", "f10w2-2.csv", "f10w3-3.csv"]),
    (20, "leg",   ["f20leg1-1.csv", "f20leg2-2.csv", "f20leg3-3.csv"]),
    (20, "2leg",  ["f202leg1-1.csv", "f202leg2-2.csv", "f202leg3-3.csv"]),
    (20, "4leg",  ["f204leg1-1.csv", "f204leg2-2.csv", "f204leg3-3.csv"]),
    (20, "wheel", ["f20w1-1.csv", "f20w2-2.csv", "f20w3-3.csv"]),
    (30, "leg",   ["f30leg1-1.csv", "f30leg2-2.csv", "f30leg3-3.csv"]),
    (30, "2leg",  ["f302leg1-1.csv", "f302leg2-2.csv", "f302leg3-3.csv"]),
    (30, "4leg",  ["f304leg1-1.csv", "f304leg2-2.csv", "f304leg3-3.csv"]),
    (30, "wheel", ["f30w1-1.csv", "f30w2-2.csv", "f30w3-3.csv"]),
    (50, "leg",   ["f50leg1-1.csv", "f50leg2-2.csv", "f50leg3-3.csv"]),
    (50, "2leg",  ["f502leg1-1.csv", "f502leg2-2.csv", "f502leg3-3.csv"]),
    (50, "4leg",  ["f504leg1-1.csv", "f504leg2-2.csv", "f504leg3-3.csv"]),
    (50, "wheel", ["f50w1-1.csv", "f50w2-2.csv", "f50w3-3.csv"]),
]

STEP_CONDITIONS = [
    (10, "leg",   ["s10leg1-1.csv", "s10leg2-2.csv", "s10leg3-3.csv"]),
    (10, "2leg",  ["s102leg1-1.csv", "s102leg2-2.csv", "s102leg3-3.csv"]),
    (10, "4leg",  ["s104leg1-1.csv", "s104leg2-2.csv", "s104leg3-3.csv"]),
    (20, "leg",   ["s20leg1-1.csv", "s20leg2-2.csv", "s20leg3-3.csv"]),
    (20, "2leg",  ["s202leg1-1.csv", "s202leg2-2.csv", "s202leg3-3.csv"]),
    (20, "4leg",  ["s204leg1-1.csv", "s204leg2-2.csv", "s204leg3-3.csv"]),
    (30, "leg",   ["s30leg1-1.csv", "s30leg2-2.csv", "s30leg3-3.csv"]),
    (30, "2leg",  ["s302leg1-1.csv", "s302leg2-2.csv", "s302leg3-3.csv"]),
    (30, "4leg",  ["s304leg1-1.csv", "s304leg2-2.csv", "s304leg3-3.csv"]),
    (30, "wheel", ["s30w1-1.csv", "s30w2-2.csv", "s30w3-3.csv"]),
]


def _sanity_check_csv(csv_path: Path) -> None:
    """Verify CSV column layout matches expectations."""
    with open(csv_path, "rb") as f:
        row1 = f.readline().decode("utf-8", errors="replace").strip().split(",")
        row2 = f.readline().decode("utf-8", errors="replace").strip().split(",")

    # Propagate group labels (row1 has blanks for continuation columns)
    groups = []
    current = ""
    for label in row1:
        if label:
            current = label
        groups.append(current)

    expected = [
        (0, "", "t"),
        (1, "mass_A", "x"), (2, "mass_A", "y"),
        (3, "mass_A", "vx"), (4, "mass_A", "vy"),
        (5, "mass_C", "x"), (6, "mass_C", "y"),
        (7, "mass_C", "vx"), (8, "mass_C", "vy"),
        (9, "mass_B", "\u03b8"), (10, "mass_B", "\u03c9"),
        (11, "mass_C", "\u03b8"),
    ]
    for col, exp_group, exp_label in expected:
        if col >= len(groups):
            raise ValueError(f"{csv_path.name}: only {len(groups)} cols, expected {col+1}")
        if groups[col] != exp_group:
            raise ValueError(
                f"{csv_path.name}: col {col} group={groups[col]!r}, expected {exp_group!r}"
            )
        if row2[col] != exp_label:
            raise ValueError(
                f"{csv_path.name}: col {col} label={row2[col]!r}, expected {exp_label!r}"
            )


def compute_trial_cot(csv_path: Path, morph: str, terrain: str) -> dict | None:
    """Compute COT upper bound for a single trial.

    Returns dict with intermediate values for inspection, or None if invalid.
    """
    dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    t = dat[:, 0]
    omega_deg = dat[:, 10]  # deg/s
    n = len(t)

    # Forward velocity: avg of mass_A and mass_C, negated (camera frame)
    vx = -0.5 * (dat[:, 3] + dat[:, 7])  # m/s forward

    # Windowing: last 50% for flat, q60 for step
    if terrain == "flat":
        lo = n // 2
        hi = n
    else:
        lo = int(0.45 * n)
        hi = int(0.75 * n)

    omega_window = omega_deg[lo:hi]
    vx_window = vx[lo:hi]
    t_window = t[lo:hi]

    # Skip if too many NaNs in omega
    nan_frac = np.isnan(omega_window).sum() / len(omega_window)
    if nan_frac > 0.5:
        return None

    omega_rad = np.abs(np.deg2rad(np.nan_to_num(omega_window)))  # rad/s
    p_max = N_LEGS * TAU_MAX * omega_rad  # W (instantaneous upper bound)

    v_avg = float(np.nanmean(vx_window))  # m/s
    if abs(v_avg) < 1e-6:
        return None

    p_avg = float(np.nanmean(p_max))  # W
    mass = ROBOT_MASS[morph]
    cot = p_avg / (mass * G * abs(v_avg))

    # Energy: integrate P_max over window
    dt = float(np.median(np.diff(t_window)))
    e_max = float(np.nansum(p_max) * dt)  # J
    distance = abs(v_avg) * (t_window[-1] - t_window[0])  # m

    return {
        "cot_upper": cot,
        "p_avg_uW": p_avg * 1e6,
        "e_max_uJ": e_max * 1e6,
        "v_avg_mm_s": v_avg * 1000,
        "omega_mean_deg_s": float(np.nanmean(np.abs(omega_deg[lo:hi]))),
        "distance_mm": distance * 1000,
        "nan_frac": nan_frac,
    }


def main():
    terrain = "flat"
    if "--terrain" in sys.argv:
        idx = sys.argv.index("--terrain")
        terrain = sys.argv[idx + 1]

    conditions = FLAT_CONDITIONS if terrain == "flat" else STEP_CONDITIONS
    csv_dir = CSV_DIR_FLAT if terrain == "flat" else CSV_DIR_STEP

    print(f"=== COT Upper Bound ({terrain} terrain) ===")
    print(f"tau_max = {TAU_MAX*1e6:.2f} uN·m per leg, {N_LEGS} legs")
    print()

    # Sanity check all CSVs first
    print("Sanity checking CSV column layouts...")
    for freq, morph, files in conditions:
        for fname in files:
            p = csv_dir / fname
            if p.exists():
                _sanity_check_csv(p)
    print("All CSVs pass sanity check.\n")

    header = f"{'Morph':<8} {'Freq':>4} {'Trial':>5}  {'COT_ub':>8} {'P_avg(uW)':>10} {'v(mm/s)':>9} {'w(d/s)':>8} {'NaN%':>5}"
    print(header)
    print("-" * len(header))

    for freq, morph, files in conditions:
        for ti, fname in enumerate(files):
            p = csv_dir / fname
            if not p.exists():
                print(f"{morph:<8} {freq:>4} {ti+1:>5}  {'MISSING':>8}")
                continue
            result = compute_trial_cot(p, morph, terrain)
            if result is None:
                print(f"{morph:<8} {freq:>4} {ti+1:>5}  {'SKIP':>8}  (too many NaNs or zero velocity)")
                continue
            print(
                f"{morph:<8} {freq:>4} {ti+1:>5}  "
                f"{result['cot_upper']:8.2f} "
                f"{result['p_avg_uW']:10.1f} "
                f"{result['v_avg_mm_s']:9.1f} "
                f"{result['omega_mean_deg_s']:8.0f} "
                f"{result['nan_frac']*100:5.1f}"
            )

    # Summary: mean COT per (morph, freq)
    print(f"\n{'=== Summary: mean COT upper bound per condition ===':}")
    print(f"{'Morph':<8} {'Freq':>4} {'COT_ub':>8} {'n':>3}")
    print("-" * 28)
    for freq, morph, files in conditions:
        cots = []
        for fname in files:
            p = csv_dir / fname
            if not p.exists():
                continue
            r = compute_trial_cot(p, morph, terrain)
            if r is not None:
                cots.append(r["cot_upper"])
        if cots:
            print(f"{morph:<8} {freq:>4} {np.mean(cots):8.2f} {len(cots):>3}")
        else:
            print(f"{morph:<8} {freq:>4} {'n/a':>8} {0:>3}")


if __name__ == "__main__":
    main()
