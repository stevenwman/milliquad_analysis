"""Plot experimental velocity vs frequency for flat and step terrains.

Individual trial points shown as scatter; lines through per-condition means.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 14
import matplotlib.pyplot as plt
import numpy as np

# ── CSV loading (same logic as pipeline_common.py) ──
CSV_ROOT = Path(__file__).resolve().parent / "csv"
MM_SCALE = 1000.0


def _load(csv_name: str) -> np.ndarray:
    for sub in ("flat", "steps"):
        p = CSV_ROOT / sub / csv_name
        if p.exists():
            return np.genfromtxt(p, delimiter=",", skip_header=2)
    raise FileNotFoundError(csv_name)


def _flat_vx(csv_name: str, points: int, steady_t: float) -> float:
    """Per-trial steady-state forward velocity (mm/s) for flat terrain."""
    dat = _load(csv_name)[:points, :]
    t = dat[:, 0]
    vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))
    return float(np.nanmean(vx[t > steady_t]))


def _flat_vx_last50(csv_name: str) -> float:
    """Per-trial velocity using last 50% of recording (for 20Hz)."""
    dat = _load(csv_name)
    t = dat[:, 0]
    vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))
    mid = len(t) // 2
    return float(np.nanmean(vx[mid:]))


def _step_vx_q75(csv_name: str) -> float:
    """Per-trial velocity using q75 ± 150 window (matches config_step.py targets)."""
    dat = _load(csv_name)
    vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))
    q75 = int(0.75 * len(vx))
    lo = max(0, q75 - 150)
    hi = min(len(vx), q75 + 150)
    return float(np.nanmean(vx[lo:hi]))


def _step_vx_q60(csv_name: str) -> float:
    """Per-trial velocity using 30% window centered at 60% (indices 45%–75%)."""
    dat = _load(csv_name)
    vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))
    n = len(vx)
    lo = int(0.45 * n)
    hi = int(0.75 * n)
    return float(np.nanmean(vx[lo:hi]))


# ── Flat terrain condition definitions (from flat_pipeline.py) ──
# fmt: off
FLAT_CONDITIONS = [
    # (freq, morphology, files, trial_indices_1based, points, steady_t)
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

# 20Hz flat: use last-50% method (no pipeline condition defined; corrected per VELOCITY_SUMMARY_FLAT.md)
FLAT_20HZ = [
    (20, "leg",   ["f20leg1-1.csv","f20leg2-2.csv","f20leg3-3.csv"], [1,2,3]),
    (20, "2leg",  ["f202leg1-1.csv","f202leg2-2.csv","f202leg3-3.csv"], [1,2,3]),
    (20, "4leg",  ["f204leg1-1.csv","f204leg2-2.csv","f204leg3-3.csv"], [1,2,3]),
    (20, "wheel", ["f20w1-1.csv","f20w2-2.csv","f20w3-3.csv"], [1,2,3]),
]

# Step terrain conditions (from steps_pipeline.py)
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
JITTER = {"leg": 0, "2leg": 0, "4leg": 0, "wheel": 0}


# ── Extract per-trial velocities ──

def extract_flat():
    """Returns {morph: {"freqs": [...], "trials": [...], "means": [...]}}.
    freqs/trials are per-trial; means are per-condition."""
    data = {m: {"freqs": [], "trials": [], "mean_freqs": [], "means": [], "stds": []} for m in COLORS}

    for freq, morph, files, idx, points, steady_t in FLAT_CONDITIONS:
        trial_files = [files[i - 1] for i in idx]
        vels = [_flat_vx(f, points, steady_t) for f in trial_files]
        data[morph]["freqs"].extend([freq] * len(vels))
        data[morph]["trials"].extend(vels)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(vels))
        data[morph]["stds"].append(np.std(vels, ddof=1) if len(vels) > 1 else 0.0)

    for freq, morph, files, idx in FLAT_20HZ:
        trial_files = [files[i - 1] for i in idx]
        vels = [_flat_vx_last50(f) for f in trial_files]
        data[morph]["freqs"].extend([freq] * len(vels))
        data[morph]["trials"].extend(vels)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(vels))
        data[morph]["stds"].append(np.std(vels, ddof=1) if len(vels) > 1 else 0.0)

    # Sort mean arrays by frequency
    for morph in data:
        order = np.argsort(data[morph]["mean_freqs"])
        data[morph]["mean_freqs"] = [data[morph]["mean_freqs"][i] for i in order]
        data[morph]["means"] = [data[morph]["means"][i] for i in order]
        data[morph]["stds"] = [data[morph]["stds"][i] for i in order]
    return data


def extract_step():
    data = {m: {"freqs": [], "trials": [], "mean_freqs": [], "means": [], "stds": []} for m in COLORS}

    for freq, morph, files, idx in STEP_CONDITIONS:
        trial_files = [files[i - 1] for i in idx]
        vels = [_step_vx_q75(f) for f in trial_files]
        data[morph]["freqs"].extend([freq] * len(vels))
        data[morph]["trials"].extend(vels)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(vels))
        data[morph]["stds"].append(np.std(vels, ddof=1) if len(vels) > 1 else 0.0)

    return data


def extract_step_q60():
    """Step velocity using 30% window centered at 60% (indices 45%–75%)."""
    data = {m: {"freqs": [], "trials": [], "mean_freqs": [], "means": [], "stds": []} for m in COLORS}

    for freq, morph, files, idx in STEP_CONDITIONS:
        trial_files = [files[i - 1] for i in idx]
        vels = [_step_vx_q60(f) for f in trial_files]
        data[morph]["freqs"].extend([freq] * len(vels))
        data[morph]["trials"].extend(vels)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(vels))
        data[morph]["stds"].append(np.std(vels, ddof=1) if len(vels) > 1 else 0.0)

    return data


def extract_rough():
    """Rough (random) terrain velocity from random_terrain_raw.csv.

    Returns same format as extract_flat: {morph: {freqs, trials, mean_freqs, means, stds}}.
    Only successful trials (non-'n/a') are included.
    """
    import csv

    csv_path = Path(__file__).resolve().parent / "csv" / "random_terrain_raw.csv"
    _CSV_MORPH = {"leg": "leg", "2-leg": "2leg", "4-leg": "4leg", "wheel": "wheel"}
    data = {m: {"freqs": [], "trials": [], "mean_freqs": [], "means": [], "stds": []} for m in COLORS}

    for row in csv.DictReader(open(csv_path)):
        freq = float(row["freq_hz"])
        morph = _CSV_MORPH[row["morphology"]]
        trials = []
        for k in ("trial_1", "trial_2", "trial_3", "trial_4", "trial_5"):
            v = row[k].strip()
            if v and v != "n/a":
                trials.append(float(v))
        if not trials:
            continue
        data[morph]["freqs"].extend([freq] * len(trials))
        data[morph]["trials"].extend(trials)
        data[morph]["mean_freqs"].append(freq)
        data[morph]["means"].append(np.mean(trials))
        data[morph]["stds"].append(np.std(trials, ddof=1) if len(trials) > 1 else 0.0)

    for morph in data:
        order = np.argsort(data[morph]["mean_freqs"])
        data[morph]["mean_freqs"] = [data[morph]["mean_freqs"][i] for i in order]
        data[morph]["means"] = [data[morph]["means"][i] for i in order]
        data[morph]["stds"] = [data[morph]["stds"][i] for i in order]
    return data


# ── Plotting ──

def plot_terrain(ax, data, title):
    morphs = ("leg", "2leg", "4leg", "wheel")
    n = len(morphs)
    dodge_width = 1.2  # total spread in Hz
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
    ax.set_ylabel("Forward Velocity (mm/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


# ── Main ──

def main():
    flat_data = extract_flat()
    step_data = extract_step()

    # Pop wheel 50Hz gentle mean, replace with failure (0) so shading tapers to 0
    wheel_50_mean = flat_data["wheel"]["means"].pop()  # last entry = 50Hz
    flat_data["wheel"]["mean_freqs"].pop()
    flat_data["wheel"]["stds"].pop()
    # Re-add 50Hz as failure mode → shading collapses to 0
    flat_data["wheel"]["mean_freqs"].append(50)
    flat_data["wheel"]["means"].append(0.0)
    flat_data["wheel"]["stds"].append(0.0)

    # Flat plot
    fig_flat, ax_flat = plt.subplots(figsize=(7, 5))
    plot_terrain(ax_flat, flat_data, "Flat Terrain: Velocity vs Frequency")

    # Wheel 50 Hz: dashed branch up to gentle actuation mean
    wc = COLORS["wheel"]
    ax_flat.plot([30, 50], [flat_data["wheel"]["means"][-2], wheel_50_mean], "--", color=wc, linewidth=1.5)
    ax_flat.plot(50, wheel_50_mean, "^", color=wc, markersize=8, zorder=5)
    # X marker at failure point
    ax_flat.plot(50, 0, "x", color=wc, markersize=10, markeredgewidth=2.5, zorder=5)
    ax_flat.set_xticks([10, 20, 30, 50])
    ax_flat.set_xlim(5, 55)
    fig_flat.tight_layout()
    fig_flat.savefig("experimental_data/plots/velocity_vs_freq_flat.png", dpi=150)

    # Clean version: no dashed branch, no 50Hz wheel scatter/markers
    flat_data_clean = extract_flat()
    # Pop 50Hz wheel entirely (shading + scatter)
    flat_data_clean["wheel"]["means"].pop()
    flat_data_clean["wheel"]["mean_freqs"].pop()
    flat_data_clean["wheel"]["stds"].pop()
    # Remove 50Hz wheel scatter points
    wf = flat_data_clean["wheel"]
    keep = [i for i, f in enumerate(wf["freqs"]) if f != 50]
    wf["freqs"] = [wf["freqs"][i] for i in keep]
    wf["trials"] = [wf["trials"][i] for i in keep]
    # Re-add 50Hz as failure (shading tapers to 0, no scatter)
    wf["mean_freqs"].append(50)
    wf["means"].append(0.0)
    wf["stds"].append(0.0)

    fig_clean, ax_clean = plt.subplots(figsize=(7, 5))
    plot_terrain(ax_clean, flat_data_clean, "Flat Terrain: Velocity vs Frequency")
    ax_clean.plot(50, 0, "x", color=COLORS["wheel"], markersize=10, markeredgewidth=2.5, zorder=5)
    ax_clean.set_xticks([10, 20, 30, 50])
    ax_clean.set_xlim(5, 55)
    fig_clean.tight_layout()
    fig_clean.savefig("experimental_data/plots/velocity_vs_freq_flat_clean.png", dpi=150)

    # Inject wheel failure modes at 10/20 Hz into step data so shading covers 10→20→30
    step_data["wheel"]["mean_freqs"] = [10, 20] + step_data["wheel"]["mean_freqs"]
    step_data["wheel"]["means"] = [0.0, 0.0] + step_data["wheel"]["means"]
    step_data["wheel"]["stds"] = [0.0, 0.0] + step_data["wheel"]["stds"]

    # Step plot
    fig_step, ax_step = plt.subplots(figsize=(7, 5))
    plot_terrain(ax_step, step_data, "Step Terrain: Velocity vs Frequency")
    # X markers for wheel failure modes
    ax_step.plot(10, 0, "x", color=wc, markersize=10, markeredgewidth=2.5, zorder=5)
    ax_step.plot(20, 0, "x", color=wc, markersize=10, markeredgewidth=2.5, zorder=5)
    ax_step.set_xticks([10, 20, 30])
    ax_step.set_xlim(5, 35)
    fig_step.tight_layout()
    fig_step.savefig("experimental_data/plots/velocity_vs_freq_step.png", dpi=150)

    # ── Combined side-by-side: flat (clean) + step ──
    flat_side = extract_flat()
    # Remove 50Hz wheel scatter, add failure mode
    wf2 = flat_side["wheel"]
    wf2["means"].pop(); wf2["mean_freqs"].pop(); wf2["stds"].pop()
    keep2 = [i for i, f in enumerate(wf2["freqs"]) if f != 50]
    wf2["freqs"] = [wf2["freqs"][i] for i in keep2]
    wf2["trials"] = [wf2["trials"][i] for i in keep2]
    wf2["mean_freqs"].append(50); wf2["means"].append(0.0); wf2["stds"].append(0.0)

    step_side = extract_step()
    step_side["wheel"]["mean_freqs"] = [10, 20] + step_side["wheel"]["mean_freqs"]
    step_side["wheel"]["means"] = [0.0, 0.0] + step_side["wheel"]["means"]
    step_side["wheel"]["stds"] = [0.0, 0.0] + step_side["wheel"]["stds"]

    fig_both, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 7.2))

    plot_terrain(ax_top, flat_side, "Flat Terrain")
    ax_top.plot(50, 0, "x", color=COLORS["wheel"], markersize=10, markeredgewidth=2.5, zorder=5)
    ax_top.set_xticks([10, 20, 30, 50])
    ax_top.set_xlim(7, 53)

    plot_terrain(ax_bot, step_side, "Step Terrain")
    ax_bot.plot(10, 0, "x", color=COLORS["wheel"], markersize=10, markeredgewidth=2.5, zorder=5)
    ax_bot.plot(20, 0, "x", color=COLORS["wheel"], markersize=10, markeredgewidth=2.5, zorder=5)
    ax_bot.set_xticks([10, 20, 30])
    ax_bot.set_xlim(7, 33)

    # Single legend on top panel
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.get_legend().remove()
    ax_bot.get_legend().remove()
    ax_top.legend(handles, labels, loc="upper left", fontsize=12, framealpha=0.9)

    fig_both.tight_layout()
    fig_both.savefig("experimental_data/plots/velocity_flat_vs_step.png", dpi=150, bbox_inches="tight")

    print("Saved: experimental_data/plots/velocity_vs_freq_flat.png")
    print("Saved: experimental_data/plots/velocity_vs_freq_flat_clean.png")
    print("Saved: experimental_data/plots/velocity_vs_freq_step.png")
    print("Saved: experimental_data/plots/velocity_flat_vs_step.png")
    plt.show()


if __name__ == "__main__":
    main()
