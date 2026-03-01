"""Plot experimental pitch amplitude vs frequency for flat and step terrains.

Pitch amplitude = RMS of detrended body angle (θ - mean(θ)) in steady state.
Same trial selection and trimming as velocity analysis.
  - Flat: per-condition time window (points, steady_t) or last-50% (20Hz)
  - Step: q75 ± 150 sample window (matches velocity analysis)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 14
import matplotlib.pyplot as plt
import numpy as np

# ── CSV loading ──
CSV_ROOT = Path(__file__).resolve().parent / "csv"


def _find_csv(csv_name: str) -> Path:
    for sub in ("flat", "steps"):
        p = CSV_ROOT / sub / csv_name
        if p.exists():
            return p
    raise FileNotFoundError(csv_name)


def _load(csv_name: str) -> np.ndarray:
    return np.genfromtxt(_find_csv(csv_name), delimiter=",", skip_header=2)


def _body_theta_col(csv_name: str) -> int:
    """Find the mass_C θ column by parsing both header rows.

    Row 1 labels groups (mass_A, mass_C, mass_B); labels propagate right
    until the next label. We want the θ under the LAST mass_C group.
    """
    with open(_find_csv(csv_name)) as f:
        row1 = f.readline().strip().split(",")
        row2 = f.readline().strip().split(",")
    # Propagate row1 labels rightward to fill blanks
    groups = []
    current = ""
    for label in row1:
        if label:
            current = label
        groups.append(current)
    # Find θ columns that belong to mass_C
    theta_cols = [i for i in range(len(row2)) if row2[i] == "θ" and i < len(groups) and groups[i] == "mass_C"]
    if theta_cols:
        return theta_cols[-1]
    raise ValueError(f"No mass_C θ column found in {csv_name}")


def _flat_pitch_rms(csv_name: str, points: int, steady_t: float) -> float:
    """RMS pitch amplitude (degrees) in steady state for flat terrain."""
    dat = _load(csv_name)[:points, :]
    t = dat[:, 0]
    theta_col = _body_theta_col(csv_name)
    theta = dat[:, theta_col]
    mask = t > steady_t
    if mask.sum() < 10:
        return 0.0
    theta_ss = theta[mask]
    theta_ss = theta_ss[~np.isnan(theta_ss)]
    if len(theta_ss) < 10:
        return 0.0
    return float(np.std(theta_ss))  # std = RMS of (θ - mean(θ))


def _flat_pitch_rms_last50(csv_name: str) -> float:
    """RMS pitch amplitude using last 50% of recording (for 20Hz)."""
    dat = _load(csv_name)
    theta_col = _body_theta_col(csv_name)
    theta = dat[:, theta_col]
    mid = len(theta) // 2
    theta_ss = theta[mid:]
    theta_ss = theta_ss[~np.isnan(theta_ss)]
    if len(theta_ss) < 10:
        return 0.0
    return float(np.std(theta_ss))


def _step_pitch_rms(csv_name: str) -> float:
    """RMS pitch amplitude using 50%–90% of recording (skip transient + cliff-fall)."""
    dat = _load(csv_name)
    theta_col = _body_theta_col(csv_name)
    theta = dat[:, theta_col]
    n = len(theta)
    lo = n // 2
    hi = int(0.9 * n)
    theta_ss = theta[lo:hi]
    theta_ss = theta_ss[~np.isnan(theta_ss)]
    if len(theta_ss) < 10:
        return 0.0
    return float(np.std(theta_ss))


def _step_pitch_rms_q60(csv_name: str) -> float:
    """RMS pitch amplitude using 30% window centered at 60% (indices 45%–75%)."""
    dat = _load(csv_name)
    theta_col = _body_theta_col(csv_name)
    theta = dat[:, theta_col]
    n = len(theta)
    lo = int(0.45 * n)
    hi = int(0.75 * n)
    theta_ss = theta[lo:hi]
    theta_ss = theta_ss[~np.isnan(theta_ss)]
    if len(theta_ss) < 10:
        return 0.0
    return float(np.std(theta_ss))


# ── Condition definitions (same as velocity) ──
# fmt: off
FLAT_CONDITIONS = [
    (10, "leg",   ["f10leg1-1.csv","f10leg2-2.csv","f10leg3-3.csv","f10leg4-4.csv"], [1,2,3], 2500, 0.3),
    (10, "2leg",  ["f102leg1-1.csv","f102leg2-2.csv","f102leg3-3.csv","f102leg4-4.csv"], [1,2,4], 1480, 0.3),
    (10, "4leg",  ["f104leg1-1.csv","f104leg2-2.csv","f104leg3-3.csv","f104leg4-4.csv"], [1,2,4], 1199, 0.3),
    (10, "wheel", ["f10w1-1.csv","f10w2-2.csv","f10w3-3.csv","f10w4-4.csv"], [1,2,3], 910, 0.3),
    (30, "leg",   ["f30leg1-1.csv","f30leg2-2.csv","f30leg3-3.csv","f30leg4-4.csv"], [1,2,3,4], 1100, 0.15),
    (30, "2leg",  ["f302leg1-1.csv","f302leg2-2.csv","f302leg3-3.csv","f302leg4-4.csv"], [1,2,3], 760, 0.3),
    (30, "4leg",  ["f304leg1-1.csv","f304leg2-2.csv","f304leg3-3.csv","f304leg4-4.csv"], [1,2,3,4], 550, 0.3),
    (30, "wheel", ["f30w1-1.csv","f30w2-2.csv","f30w3-3.csv","f30w4-4.csv"], [1,2,3,4], 350, 0.3),
    (50, "leg",   ["f50leg1-1.csv","f50leg2-2.csv","f50leg3-3.csv"], [1,2,3], 1960, 0.35),
    (50, "2leg",  ["f502leg1-1.csv","f502leg2-2.csv","f502leg3-3.csv"], [1,2,3], 1280, 0.35),
    (50, "4leg",  ["f504leg1-1.csv","f504leg2-2.csv","f504leg3-3.csv"], [1,2,3], 1060, 0.35),
    (50, "wheel", ["f50w1-1.csv","f50w2-2.csv","f50w3-3.csv"], [1,2,3], 620, 0.25),
]

FLAT_20HZ = [
    (20, "leg",   ["f20leg1-1.csv","f20leg2-2.csv","f20leg3-3.csv"], [1,2,3]),
    (20, "2leg",  ["f202leg1-1.csv","f202leg2-2.csv","f202leg3-3.csv"], [1,2,3]),
    (20, "4leg",  ["f204leg1-1.csv","f204leg2-2.csv","f204leg3-3.csv"], [1,2,3]),
    (20, "wheel", ["f20w1-1.csv","f20w2-2.csv","f20w3-3.csv"], [1,2,3]),
]

STEP_CONDITIONS = [
    (10, "leg",   ["s10leg1-1.csv","s10leg2-2.csv","s10leg3-3.csv"], [1,2,3]),
    (10, "2leg",  ["s102leg1-1.csv","s102leg2-2.csv","s102leg3-3.csv"], [1,2,3]),
    (10, "4leg",  ["s104leg1-1.csv","s104leg2-2.csv","s104leg3-3.csv"], [1,2,3]),
    (20, "leg",   ["s20leg1-1.csv","s20leg2-2.csv","s20leg3-3.csv"], [1,2,3]),
    (20, "2leg",  ["s202leg1-1.csv","s202leg2-2.csv","s202leg3-3.csv"], [1,2,3]),
    (20, "4leg",  ["s204leg1-1.csv","s204leg2-2.csv","s204leg3-3.csv"], [1,2,3]),
    (30, "leg",   ["s30leg1-1.csv","s30leg2-2.csv","s30leg3-3.csv"], [1,2,3]),
    (30, "2leg",  ["s302leg1-1.csv","s302leg2-2.csv","s302leg3-3.csv"], [1,2,3]),
    (30, "4leg",  ["s304leg1-1.csv","s304leg2-2.csv","s304leg3-3.csv"], [1,2,3]),
    (30, "wheel", ["s30w1-1.csv","s30w2-2.csv","s30w3-3.csv"], [1,2,3]),
]
# fmt: on

COLORS = {"leg": "#1E88E5", "2leg": "#FFC107", "4leg": "#007561", "wheel": "#D81B60"}
LABELS = {"leg": "L1", "2leg": "L2", "4leg": "L4", "wheel": "WR"}


# ── Extract per-trial pitch amplitudes ──

def extract_flat_pitch():
    data = {m: {"freqs": [], "trials": [], "mean_freqs": [], "means": [], "stds": []} for m in COLORS}

    for freq, morph, files, idx, points, steady_t in FLAT_CONDITIONS:
        trial_files = [files[i - 1] for i in idx]
        vals = [_flat_pitch_rms(f, points, steady_t) for f in trial_files]
        data[morph]["freqs"].extend([freq] * len(vals))
        data[morph]["trials"].extend(vals)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(vals))
        data[morph]["stds"].append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    for freq, morph, files, idx in FLAT_20HZ:
        trial_files = [files[i - 1] for i in idx]
        vals = [_flat_pitch_rms_last50(f) for f in trial_files]
        data[morph]["freqs"].extend([freq] * len(vals))
        data[morph]["trials"].extend(vals)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(vals))
        data[morph]["stds"].append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    # Sort by frequency
    for morph in data:
        order = np.argsort(data[morph]["mean_freqs"])
        data[morph]["mean_freqs"] = [data[morph]["mean_freqs"][i] for i in order]
        data[morph]["means"] = [data[morph]["means"][i] for i in order]
        data[morph]["stds"] = [data[morph]["stds"][i] for i in order]

    return data


def extract_step_pitch():
    data = {m: {"freqs": [], "trials": [], "mean_freqs": [], "means": [], "stds": []} for m in COLORS}

    for freq, morph, files, idx in STEP_CONDITIONS:
        trial_files = [files[i - 1] for i in idx]
        vals = [_step_pitch_rms(f) for f in trial_files]
        data[morph]["freqs"].extend([freq] * len(vals))
        data[morph]["trials"].extend(vals)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(vals))
        data[morph]["stds"].append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    return data


def extract_step_pitch_q60():
    """Step pitch using 30% window centered at 60% (indices 45%–75%)."""
    data = {m: {"freqs": [], "trials": [], "mean_freqs": [], "means": [], "stds": []} for m in COLORS}

    for freq, morph, files, idx in STEP_CONDITIONS:
        trial_files = [files[i - 1] for i in idx]
        vals = [_step_pitch_rms_q60(f) for f in trial_files]
        data[morph]["freqs"].extend([freq] * len(vals))
        data[morph]["trials"].extend(vals)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(vals))
        data[morph]["stds"].append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    return data


# ── Plotting ──

def plot_pitch(ax, data, title):
    morphs = ("leg", "2leg", "4leg", "wheel")
    n = len(morphs)
    dodge_width = 1.2
    for idx, morph in enumerate(morphs):
        d = data[morph]
        if not d["mean_freqs"]:
            continue
        dx = (idx - (n - 1) / 2) * (dodge_width / (n - 1))
        freqs_scatter = np.array(d["freqs"], dtype=float) + dx
        mean = np.array(d["means"])
        std = np.array(d["stds"])
        freq_arr = np.array(d["mean_freqs"], dtype=float) + dx
        ax.fill_between(freq_arr, mean - std, mean + std, color=COLORS[morph], alpha=0.2, label=LABELS[morph])
        ax.scatter(freqs_scatter, d["trials"], color=COLORS[morph], alpha=0.6, s=30, zorder=3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Pitch Amplitude RMS (\u00b0)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


# ── Main ──

def main():
    flat_pitch = extract_flat_pitch()
    step_pitch = extract_step_pitch()

    # Print summary tables
    for terrain, pdata in [("FLAT", flat_pitch), ("STEP", step_pitch)]:
        print(f"\n{terrain}:")
        deg = "\u00b0"
        print(f"{'Morph':<8} {'Freq':<6} {f'Mean ({deg})':<10} {f'Std ({deg})':<10} {'N':>3}")
        print("-" * 40)
        for morph in ("leg", "2leg", "4leg", "wheel"):
            d = pdata[morph]
            for i, freq in enumerate(d["mean_freqs"]):
                n = sum(1 for f in d["freqs"] if f == freq)
                print(f"{morph:<8} {freq:<6.0f} {d['means'][i]:<10.2f} {d['stds'][i]:<10.2f} {n:>3}")

    # ── Individual flat plot (keep existing output) ──
    fig_flat, ax_flat = plt.subplots(figsize=(7, 5))
    plot_pitch(ax_flat, flat_pitch, "Flat Terrain: Pitch Amplitude vs Frequency")
    ax_flat.set_xticks([10, 20, 30, 50])
    ax_flat.set_xlim(5, 55)
    fig_flat.tight_layout()
    fig_flat.savefig("experimental_data/plots/pitch_vs_freq_flat.png", dpi=150)

    # ── Combined vertical stack: flat + step ──
    fig_both, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 7.2))

    plot_pitch(ax_top, flat_pitch, "Flat Terrain")
    ax_top.set_xticks([10, 20, 30, 50])
    ax_top.set_xlim(7, 53)

    plot_pitch(ax_bot, step_pitch, "Step Terrain")
    ax_bot.set_xticks([10, 20, 30])
    ax_bot.set_xlim(7, 33)

    # Single legend on top panel
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.get_legend().remove()
    ax_bot.get_legend().remove()
    ax_top.legend(handles, labels, loc="upper left", fontsize=12, framealpha=0.9)

    fig_both.tight_layout()
    fig_both.savefig("experimental_data/plots/pitch_flat_vs_step.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved: experimental_data/plots/pitch_vs_freq_flat.png")
    print(f"Saved: experimental_data/plots/pitch_flat_vs_step.png")
    plt.show()


if __name__ == "__main__":
    main()
