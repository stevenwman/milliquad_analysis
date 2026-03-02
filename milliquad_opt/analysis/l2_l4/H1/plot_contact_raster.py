#!/usr/bin/env python3
"""H1 Plot 1a: Contact raster (sanity check).

Per-leg on/off heatmap for one representative trial per morphology.
Picks the median-duty-factor trial at the highest common frequency.

Usage (from milliquad_opt/):
    uv run python -m analysis.l2_l4.H1.plot_contact_raster \
        results/20260228T202903_rough_spatial_rk4

    # Specify frequency:
    uv run python -m analysis.l2_l4.H1.plot_contact_raster \
        results/20260228T013353_rk4_flat --freq 30
"""

from __future__ import annotations

import argparse
import pathlib
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
LEG_NAMES = ["FR", "FL", "BR", "BL"]


def _pick_representative_trial(
    npz, scene: str, freq: float, terrain: str,
) -> tuple[int, str] | None:
    """Pick the valid trial closest to median duty factor."""
    tidxs = []
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[0] == scene and parsed[1] == freq and parsed[3] == "time":
            tidxs.append(parsed[2])

    if not tidxs:
        return None

    valid = []
    for t in tidxs:
        prefix = f"{scene}_f{freq:g}_t{t}"
        time = npz[f"{prefix}_time"]
        pos_x = npz[f"{prefix}_pos_x"]
        pitch = npz[f"{prefix}_pitch"]
        contact = npz[f"{prefix}_leg_in_contact"]

        if not is_valid_trial(pos_x, pitch, terrain, scene, freq):
            continue

        mask = active_mask(time, pos_x, terrain)
        if mask.sum() < 10:
            continue

        df = float((contact[mask].sum(axis=1) >= 1).mean())
        valid.append((t, df, prefix))

    if not valid:
        return None

    valid.sort(key=lambda x: x[1])
    median_idx = len(valid) // 2
    return valid[median_idx][0], valid[median_idx][2]


def plot(run_dir: pathlib.Path, freq: float | None = None, save: bool = True):
    terrain = detect_terrain(run_dir)
    npz = np.load(find_npz(run_dir))

    # Find available frequencies
    freqs = set()
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            freqs.add(parsed[1])
    freqs = sorted(freqs)

    if freq is None:
        freq = freqs[-1]
    elif freq not in freqs:
        print(f"Freq {freq} not found. Available: {freqs}")
        return

    morphs_present = []
    trial_data = {}

    for scene in MORPH_ORDER:
        result = _pick_representative_trial(npz, scene, freq, terrain)
        if result is None:
            continue
        tidx, prefix = result
        morphs_present.append(scene)
        trial_data[scene] = {
            "time": npz[f"{prefix}_time"],
            "contact": npz[f"{prefix}_leg_in_contact"],
            "pos_x": npz[f"{prefix}_pos_x"],
            "tidx": tidx,
        }

    npz.close()

    if not morphs_present:
        print("No valid trials found.")
        return

    n_morphs = len(morphs_present)
    fig, axes = plt.subplots(n_morphs, 1, figsize=(10, 1.5 * n_morphs + 0.5),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax_idx, scene in enumerate(morphs_present):
        ax = axes[ax_idx]
        d = trial_data[scene]
        time_s = d["time"]
        contact = d["contact"].astype(float)  # (T, 4)

        mask = active_mask(time_s, d["pos_x"], terrain)
        time_ms = time_s[mask] * 1000
        contact_masked = contact[mask]  # (N, 4)

        # Raster: imshow with legs on y-axis
        extent = [time_ms[0], time_ms[-1], -0.5, 3.5]
        ax.imshow(
            contact_masked.T,
            aspect="auto",
            extent=extent,
            cmap=plt.cm.colors.ListedColormap(["white", MORPH_COLORS[scene]]),
            vmin=0, vmax=1,
            interpolation="none",
            origin="lower",
        )

        label = MORPH_LABELS[scene]
        ax.set_ylabel(f"{label}\n(t{d['tidx']})", fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(LEG_NAMES, fontsize=8)
        ax.tick_params(axis="y", length=0)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(
        f"H1 sanity check: leg contact raster — {TERRAIN_TITLES.get(terrain, terrain)}, "
        f"{freq:g} Hz",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out = FIGURE_DIR / f"{ts}_contact_raster_{terrain}_{freq:g}Hz.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--freq", type=float, default=None,
                        help="Drive frequency (Hz). Default: highest available.")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    plot(args.run_dir, freq=args.freq, save=not args.no_save)


if __name__ == "__main__":
    main()
