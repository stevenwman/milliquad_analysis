#!/usr/bin/env python3
"""H3 Plot: FFT harmonic spectrum of forward velocity.

Grid of panels — rows = terrain, cols = frequency.
Each panel shows FFT amplitude (mm/s) at harmonic multiples of drive freq.
L1 should peak at 1x, L4 energy shifted to 4x, WR minimal.

Usage (from milliquad_opt/):
    uv run python -m analysis.l2_l4.H3.plot_fft_spectrum \
        results/20260228T013353_rk4_flat \
        results/20260228T230022_step_q60_rk-warm \
        results/20260228T202903_rough_spatial_rk4
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from analysis.l2_l4._plot_style import (
    MORPH_COLORS,
    MORPH_LABELS,
    MORPH_ORDER,
    TERRAIN_TITLES,
)
from analysis.l2_l4._trial_filter import (
    active_mask,
    detect_terrain,
    find_npz,
    is_valid_trial,
    parse_key,
)

FIGURE_DIR = pathlib.Path(__file__).parent / "figures"
HARMONICS = [0.5, 1, 2, 3, 4, 5, 8]  # multiples of drive freq


def _collect_fft(run_dir: pathlib.Path):
    """Extract FFT amplitudes at harmonic multiples for each (scene, freq).

    Returns:
        data: {freq: {scene: {"amps": mean_array(len(HARMONICS)), "std": ...}}}
        freqs: sorted list
        terrain: str
    """
    terrain = detect_terrain(run_dir)
    npz = np.load(find_npz(run_dir))

    trials: dict[tuple[str, float], list[int]] = {}
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            scene, freq, tidx, _ = parsed
            trials.setdefault((scene, freq), []).append(tidx)

    per_trial: dict[float, dict[str, list[np.ndarray]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for (scene, freq), tidxs in trials.items():
        if scene not in MORPH_ORDER:
            continue
        for t in tidxs:
            prefix = f"{scene}_f{freq:g}_t{t}"
            time = npz[f"{prefix}_time"]
            pos_x = npz[f"{prefix}_pos_x"]
            pitch = npz[f"{prefix}_pitch"]

            if not is_valid_trial(pos_x, pitch, terrain, scene, freq):
                continue
            mask = active_mask(time, pos_x, terrain)
            if mask.sum() < 20:
                continue

            time_a = time[mask]
            dt = float(time_a[1] - time_a[0])
            if dt < 1e-10:
                continue

            vx = npz[f"{prefix}_vel_x"][mask] * 1000  # mm/s
            N = len(vx)

            fft_vals = np.fft.rfft(vx - vx.mean())
            fft_freqs = np.fft.rfftfreq(N, d=dt)
            amps_at_harmonics = np.zeros(len(HARMONICS))

            for i, h in enumerate(HARMONICS):
                target_f = h * freq
                idx = np.argmin(np.abs(fft_freqs - target_f))
                amps_at_harmonics[i] = 2.0 * np.abs(fft_vals[idx]) / N

            per_trial[freq][scene].append(amps_at_harmonics)

    npz.close()

    all_freqs = sorted(per_trial.keys())
    data: dict[float, dict[str, dict]] = {}
    for freq in all_freqs:
        data[freq] = {}
        for scene in MORPH_ORDER:
            arr_list = per_trial[freq].get(scene, [])
            if not arr_list:
                continue
            arr = np.array(arr_list)
            data[freq][scene] = {
                "amps": arr.mean(axis=0),
                "std": arr.std(axis=0),
                "n": len(arr_list),
            }

    return data, all_freqs, terrain


def plot(run_dirs: list[pathlib.Path], save: bool = True):
    terrain_data = {}
    for rd in run_dirs:
        t = detect_terrain(rd)
        terrain_data[t] = _collect_fft(rd)

    terrain_order = [t for t in ["flat", "step", "rough"] if t in terrain_data]
    if not terrain_order:
        print("No data to plot.")
        return None

    # Union of all freqs across terrains
    all_freqs = sorted(set(
        f for t in terrain_order for f in terrain_data[t][1]
    ))

    n_rows = len(terrain_order)
    n_cols = len(all_freqs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows),
                             sharey=False, squeeze=False)

    x = np.arange(len(HARMONICS))
    bar_w = 0.18
    harmonic_labels = [f"{h:g}x" for h in HARMONICS]

    for row, terrain in enumerate(terrain_order):
        data, freqs, _ = terrain_data[terrain]
        for col, freq in enumerate(all_freqs):
            ax = axes[row, col]

            if freq not in data:
                ax.set_visible(False)
                continue

            morphs = [s for s in MORPH_ORDER if s in data[freq]]
            n_m = len(morphs)
            if n_m == 0:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="0.5")
                continue

            offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bar_w
            for i, scene in enumerate(morphs):
                d = data[freq][scene]
                ax.bar(x + offsets[i], d["amps"], width=bar_w * 0.9,
                       color=MORPH_COLORS[scene],
                       label=MORPH_LABELS[scene] if row == 0 and col == 0 else None,
                       edgecolor="white", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(harmonic_labels, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(f"{freq:g} Hz")
            if row == n_rows - 1:
                ax.set_xlabel("Harmonic")

        axes[row, 0].set_ylabel(f"{TERRAIN_TITLES.get(terrain, terrain)}\nAmplitude (mm/s)")

    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=MORPH_COLORS[s]) for s in MORPH_ORDER],
        [MORPH_LABELS[s] for s in MORPH_ORDER],
        loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.99),
    )

    fig.suptitle("FFT harmonic spectrum of forward velocity", fontsize=12, y=1.02)
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        tag = "_".join(terrain_order)
        out = FIGURE_DIR / f"{ts}_fft_spectrum_{tag}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    plot(args.run_dirs, save=not args.no_save)


if __name__ == "__main__":
    main()
