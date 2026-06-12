#!/usr/bin/env python3
"""H4: Phase-propulsion correlation analysis.

Bins instantaneous forward acceleration by magnetic drive phase.
Measures how much of acceleration variance is explained by drive phase.

Metrics:
- Phase R²: var(bin_means) / var(all_ax). 0 = no phase structure, 1 = fully phase-locked.
- Amplitude: (max_bin - min_bin) in mm/s², raw modulation amplitude.
- Peak phase: drive angle where propulsion is maximal.

Usage:
    cd milliquad_opt
    uv run python -m analysis.l2_l4.phase_propulsion results/20260228T013353_rk4_flat
    uv run python -m analysis.l2_l4.phase_propulsion \
        results/20260228T013353_rk4_flat results/20260228T202903_rough_spatial_rk4
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from analysis.l2_l4._trial_filter import (
    SCENE_LABELS,
    active_mask,
    detect_terrain,
    find_npz,
    is_valid_trial,
    parse_key,
)

SCENE_ORDER = ["scene1", "scene2", "scene4", "scene_wheel"]
N_BINS = 36  # 10° bins


def analyze(run_dir: pathlib.Path):
    terrain = detect_terrain(run_dir)
    npz_path = find_npz(run_dir)
    npz = np.load(npz_path)

    # Discover trials
    trials: dict[tuple[str, float], list[int]] = {}
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            scene, freq, tidx, _ = parsed
            trials.setdefault((scene, freq), []).append(tidx)

    all_scenes = sorted(set(s for s, _ in trials),
                        key=lambda s: SCENE_ORDER.index(s) if s in SCENE_ORDER else 99)
    all_freqs = sorted(set(f for _, f in trials))

    print(f"\nPhase-Propulsion Analysis: {run_dir.name} ({terrain})")
    print(f"Phase bins: {N_BINS} ({360 // N_BINS}° each)")
    print(f"{'=' * 90}\n")

    print(f"{'Scene':<14} {'Freq':>5}  {'Phase R²':>9}  "
          f"{'Amplitude':>10}  {'Peak Phase':>11}  {'N':>3} {'skip':>4}")
    print("-" * 75)

    results = []

    for scene in all_scenes:
        label = SCENE_LABELS.get(scene, scene)
        for freq in all_freqs:
            tidxs = trials.get((scene, freq))
            if not tidxs:
                continue

            phase_r2s = []
            amplitudes = []
            peak_phases = []
            n_skipped = 0

            for t in tidxs:
                prefix = f"{scene}_f{freq:g}_t{t}"
                time = npz[f"{prefix}_time"]
                pos_x = npz[f"{prefix}_pos_x"]
                pitch = npz[f"{prefix}_pitch"]
                drive_key = f"{prefix}_drive_angle"

                if drive_key not in npz:
                    n_skipped += 1
                    continue

                if not is_valid_trial(pos_x, pitch, terrain, scene, freq):
                    n_skipped += 1
                    continue

                drive_angle = npz[drive_key]

                mask = active_mask(time, pos_x, terrain)
                if mask.sum() < 20:
                    n_skipped += 1
                    continue

                time_a = time[mask]
                dt = float(time_a[1] - time_a[0])
                if dt < 1e-10:
                    n_skipped += 1
                    continue

                vel_x_key = f"{prefix}_vel_x"
                if vel_x_key in npz:
                    vx = npz[vel_x_key][mask]
                else:
                    px = pos_x[mask]
                    vx = np.diff(px) / np.diff(time_a)
                    vx = np.append(vx, vx[-1])

                # Instantaneous acceleration
                ax = np.gradient(vx, dt)

                # Bin drive angle into [0, 2*pi)
                phase = drive_angle[mask] % (2 * np.pi)
                bin_edges = np.linspace(0, 2 * np.pi, N_BINS + 1)
                bin_idx = np.digitize(phase, bin_edges) - 1
                bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

                # Mean acceleration per bin
                bin_means = np.zeros(N_BINS)
                bin_counts = np.zeros(N_BINS)
                for b in range(N_BINS):
                    in_bin = bin_idx == b
                    if in_bin.sum() > 0:
                        bin_means[b] = ax[in_bin].mean()
                        bin_counts[b] = in_bin.sum()

                filled = bin_counts > 0
                if filled.sum() < N_BINS // 2:
                    n_skipped += 1
                    continue

                # Phase R²: fraction of acceleration variance explained by phase
                total_var = float(np.var(ax))
                if total_var < 1e-20:
                    n_skipped += 1
                    continue

                # Weighted variance of bin means (weight = bin count)
                weights = bin_counts[filled]
                bm = bin_means[filled]
                weighted_mean = np.average(bm, weights=weights)
                bin_var = float(np.average((bm - weighted_mean) ** 2, weights=weights))
                r2 = bin_var / total_var

                # Raw amplitude (mm/s²)
                amplitude = float(bm.max() - bm.min()) * 1000

                # Peak propulsion phase
                peak_bin = np.argmax(bin_means)
                peak_phase_deg = float(peak_bin) / N_BINS * 360

                phase_r2s.append(r2)
                amplitudes.append(amplitude)
                peak_phases.append(peak_phase_deg)

            if not phase_r2s:
                continue

            r2_mean = np.mean(phase_r2s)
            r2_std = np.std(phase_r2s)
            amp_mean = np.mean(amplitudes)
            pp_mean = np.mean(peak_phases)

            results.append({
                "scene": scene, "label": label, "freq": freq,
                "phase_r2_mean": r2_mean, "phase_r2_std": r2_std,
                "amplitude": amp_mean, "peak_phase": pp_mean,
                "n_trials": len(phase_r2s),
            })

            freq_str = f"{freq:g}Hz"
            print(
                f"{label:<14} {freq_str:>5}  "
                f"{r2_mean:>5.3f}+/-{r2_std:>4.3f}  "
                f"{amp_mean:>7.0f}mm/s²  "
                f"{pp_mean:>8.0f}deg  "
                f"{len(phase_r2s):>3} {n_skipped:>4}"
            )

    _print_summary(all_scenes, results)
    npz.close()
    return results


def _print_summary(all_scenes, results):
    print(f"\n{'--- Summary by morphology ---':^90}")
    print(f"{'Morphology':<14} {'Phase R²':>9}  {'Amplitude mm/s²':>16}  {'Peak Phase':>11}")
    print("-" * 55)
    for scene in all_scenes:
        scene_res = [r for r in results if r["scene"] == scene]
        if not scene_res:
            continue
        label = SCENE_LABELS.get(scene, scene)
        r2 = np.mean([r["phase_r2_mean"] for r in scene_res])
        amp = np.mean([r["amplitude"] for r in scene_res])
        pp = np.mean([r["peak_phase"] for r in scene_res])
        print(f"{label:<14} {r2:>9.3f}  {amp:>16.0f}  {pp:>8.0f}deg")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    all_results = {}
    for rd in args.run_dirs:
        terrain = detect_terrain(rd)
        all_results[terrain] = analyze(rd)

    if len(all_results) > 1:
        print(f"\n{'=' * 90}")
        print("Cross-terrain phase R² comparison (morphology average)")
        print(f"{'=' * 90}")

        terrains = list(all_results.keys())
        header = f"{'Morphology':<14}" + "".join(f"  {t:>10}" for t in terrains)
        print(header)
        print("-" * (14 + 12 * len(terrains)))

        all_scenes = set()
        for res in all_results.values():
            all_scenes.update(r["scene"] for r in res)
        all_scenes = sorted(all_scenes,
                            key=lambda s: SCENE_ORDER.index(s) if s in SCENE_ORDER else 99)

        for scene in all_scenes:
            label = SCENE_LABELS.get(scene, scene)
            row = f"{label:<14}"
            for t in terrains:
                scene_res = [r for r in all_results[t] if r["scene"] == scene]
                if scene_res:
                    r2 = np.mean([r["phase_r2_mean"] for r in scene_res])
                    row += f"  {r2:>10.3f}"
                else:
                    row += f"  {'n/a':>10}"
            print(row)


if __name__ == "__main__":
    main()
