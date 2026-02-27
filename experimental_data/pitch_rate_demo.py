#!/usr/bin/env python3
"""Demo: pitch rate processing on a single flat trial.

Shows raw θ, dθ/dt, and computed RMS values so we can verify the pipeline.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 12
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CSV_ROOT = Path(__file__).resolve().parent / "csv"

# --- Load one trial: f30 4-leg trial 1 (clear locomotion, moderate pitch) ---
csv_path = CSV_ROOT / "flat" / "f304leg1-1.csv"
dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)

# Parse headers to find mass_C θ column (last θ column)
with open(csv_path) as f:
    row1 = f.readline().strip().split(",")
    row2 = f.readline().strip().split(",")

groups = []
current = ""
for label in row1:
    if label:
        current = label
    groups.append(current)

theta_cols = [i for i in range(len(row2)) if row2[i] == "θ" and groups[i] == "mass_C"]
theta_col = theta_cols[-1]
print(f"Using column {theta_col} for body pitch (mass_C θ)")

t = dat[:, 0]
theta = dat[:, theta_col]  # degrees

# Trim to recording length used in velocity analysis (550 points for f30 4leg)
N_POINTS = 550
t = t[:N_POINTS]
theta = theta[:N_POINTS]

# Remove NaNs
valid = ~np.isnan(theta) & ~np.isnan(t)
t = t[valid]
theta = theta[valid]

print(f"Time range: {t[0]:.4f} – {t[-1]:.4f} s  ({len(t)} points)")
print(f"Sampling dt: {np.median(np.diff(t))*1000:.3f} ms")

# --- Steady state: t > 0.3s (same as pitch analysis) ---
STEADY_T = 0.3
mask_ss = t > STEADY_T
t_ss = t[mask_ss]
theta_ss = theta[mask_ss]

# --- Pitch RMS (existing method): std of θ in steady state ---
pitch_rms = np.std(theta_ss)
print(f"\nPitch RMS (existing): {pitch_rms:.2f}°")

# --- Pitch rate: dθ/dt via finite differences ---
dt = np.diff(t_ss)
dtheta = np.diff(theta_ss)
dtheta_dt = dtheta / dt  # deg/s
t_rate = t_ss[:-1] + dt / 2  # midpoints

# Pitch rate RMS
pitch_rate_rms = np.sqrt(np.mean(dtheta_dt**2))
print(f"Pitch rate RMS (dθ/dt): {pitch_rate_rms:.1f} °/s")
print(f"  mean(dθ/dt) = {np.mean(dtheta_dt):.1f} °/s")
print(f"  std(dθ/dt)  = {np.std(dtheta_dt):.1f} °/s")
print(f"  max|dθ/dt|  = {np.max(np.abs(dtheta_dt)):.1f} °/s")

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 1. Raw θ
ax = axes[0]
ax.plot(t * 1000, theta, color="#007561", linewidth=0.5)
ax.axvline(STEADY_T * 1000, color="red", linestyle="--", alpha=0.5, label=f"steady state ({STEADY_T}s)")
ax.set_ylabel("θ (°)")
ax.set_title(f"f30 4-leg trial 1 — Pitch RMS = {pitch_rms:.2f}°")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# 2. Detrended θ (θ - mean)
ax = axes[1]
theta_detrend = theta_ss - np.mean(theta_ss)
ax.plot(t_ss * 1000, theta_detrend, color="#007561", linewidth=0.5)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axhline(pitch_rms, color="red", linestyle="--", alpha=0.5, label=f"±RMS = {pitch_rms:.2f}°")
ax.axhline(-pitch_rms, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("θ − mean(θ) (°)")
ax.set_title("Detrended pitch (steady state)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# 3. dθ/dt
ax = axes[2]
ax.plot(t_rate * 1000, dtheta_dt, color="#1E88E5", linewidth=0.5)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axhline(pitch_rate_rms, color="red", linestyle="--", alpha=0.5, label=f"RMS = {pitch_rate_rms:.0f} °/s")
ax.axhline(-pitch_rate_rms, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("dθ/dt (°/s)")
ax.set_xlabel("Time (ms)")
ax.set_title("Pitch rate")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

fig.tight_layout()
out = "plots/pitch_rate_demo.png"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
