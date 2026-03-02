#!/usr/bin/env python3
"""H3: Velocity ripple analysis.

Computes intra-revolution velocity oscillation. More spokes should produce
smoother forward velocity (lower ripple coefficient).

Ripple coefficient = std(vx within one revolution) / mean(vx within one revolution).
Also reports FFT amplitude at 1x drive frequency.

Usage:
    cd milliquad_opt
    uv run python -m analysis.l2_l4.velocity_ripple results/20260228T013353_rk4_flat
    uv run python -m analysis.l2_l4.velocity_ripple \
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

    print(f"\nVelocity Ripple Analysis: {run_dir.name} ({terrain})")
    print(f"{'=' * 100}\n")

    print(f"{'Scene':<14} {'Freq':>5}  {'Ripple Coeff':>13}  "
          f"{'Mean vx mm/s':>13}  {'Std vx mm/s':>12}  "
          f"{'FFT 1x amp':>11}  {'N':>3} {'skip':>4}")
    print("-" * 100)

    results = []

    for scene in all_scenes:
        label = SCENE_LABELS.get(scene, scene)
        for freq in all_freqs:
            tidxs = trials.get((scene, freq))
            if not tidxs:
                continue

            ripple_coeffs = []
            mean_vxs = []
            std_vxs = []
            fft_1x_amps = []
            n_skipped = 0

            for t in tidxs:
                prefix = f"{scene}_f{freq:g}_t{t}"
                time = npz[f"{prefix}_time"]
                pos_x = npz[f"{prefix}_pos_x"]
                pitch = npz[f"{prefix}_pitch"]

                if not is_valid_trial(pos_x, pitch, terrain, scene, freq):
                    n_skipped += 1
                    continue

                mask = active_mask(time, pos_x, terrain)
                if mask.sum() < 10:
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
                    vx = np.diff(pos_x[mask]) / np.diff(time_a)
                    vx = np.append(vx, vx[-1])

                mean_vx = float(np.mean(vx))
                std_vx = float(np.std(vx))

                period = 1.0 / freq
                steps_per_rev = max(1, int(period / dt))
                n_full_revs = len(vx) // steps_per_rev

                if n_full_revs < 2 or abs(mean_vx) < 1e-6:
                    n_skipped += 1
                    continue

                vx_revs = vx[:n_full_revs * steps_per_rev].reshape(n_full_revs, steps_per_rev)
                rev_stds = vx_revs.std(axis=1)
                ripple = float(np.mean(rev_stds) / abs(mean_vx))

                # FFT amplitude at 1x drive frequency
                N = len(vx)
                fft_vals = np.fft.rfft(vx - mean_vx)
                fft_freqs = np.fft.rfftfreq(N, d=dt)
                idx_1x = np.argmin(np.abs(fft_freqs - freq))
                amp_1x = 2.0 * np.abs(fft_vals[idx_1x]) / N

                ripple_coeffs.append(ripple)
                mean_vxs.append(mean_vx * 1000)
                std_vxs.append(std_vx * 1000)
                fft_1x_amps.append(float(amp_1x) * 1000)

            if not ripple_coeffs:
                continue

            rc_mean = np.mean(ripple_coeffs)
            rc_std = np.std(ripple_coeffs)
            mv_mean = np.mean(mean_vxs)
            sv_mean = np.mean(std_vxs)
            fa_mean = np.mean(fft_1x_amps)

            results.append({
                "scene": scene, "label": label, "freq": freq,
                "ripple_mean": rc_mean, "ripple_std": rc_std,
                "mean_vx": mv_mean, "std_vx": sv_mean,
                "fft_1x_amp": fa_mean,
                "n_trials": len(ripple_coeffs),
            })

            freq_str = f"{freq:g}Hz"
            print(
                f"{label:<14} {freq_str:>5}  "
                f"{rc_mean:>6.3f}+/-{rc_std:>5.3f}  "
                f"{mv_mean:>13.1f}  {sv_mean:>12.1f}  "
                f"{fa_mean:>11.2f}  "
                f"{len(ripple_coeffs):>3} {n_skipped:>4}"
            )

    _print_summary(all_scenes, results)
    npz.close()
    return results


def _print_summary(all_scenes, results):
    print(f"\n{'--- Summary by morphology (averaged across freqs) ---':^100}")
    print(f"{'Morphology':<14} {'Ripple Coeff':>13}  {'FFT 1x amp mm/s':>16}")
    print("-" * 48)
    for scene in all_scenes:
        scene_res = [r for r in results if r["scene"] == scene]
        if not scene_res:
            continue
        label = SCENE_LABELS.get(scene, scene)
        rc = np.mean([r["ripple_mean"] for r in scene_res])
        fa = np.mean([r["fft_1x_amp"] for r in scene_res])
        print(f"{label:<14} {rc:>13.3f}  {fa:>16.2f}")


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
        print(f"\n{'=' * 100}")
        print("Cross-terrain ripple comparison (morphology average)")
        print(f"{'=' * 100}")

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
                    rc = np.mean([r["ripple_mean"] for r in scene_res])
                    row += f"  {rc:>10.3f}"
                else:
                    row += f"  {'n/a':>10}"
            print(row)


if __name__ == "__main__":
    main()
