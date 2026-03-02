#!/usr/bin/env python3
"""H1: Contact duty factor analysis.

Computes per-morphology duty factor (fraction of active timesteps with >= 1 leg
in contact) and simultaneous contact count distribution.

Filters out stuck/inverted trials. Uses spatial gating for rough/step terrain.

Usage:
    cd milliquad_opt
    uv run python -m analysis.l2_l4.contact_duty_factor results/20260228T013353_rk4_flat
    uv run python -m analysis.l2_l4.contact_duty_factor \
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

    # Discover all trials
    trials: dict[tuple[str, float], list[int]] = {}
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            scene, freq, tidx, _ = parsed
            trials.setdefault((scene, freq), []).append(tidx)

    all_scenes = sorted(set(s for s, _ in trials),
                        key=lambda s: SCENE_ORDER.index(s) if s in SCENE_ORDER else 99)
    all_freqs = sorted(set(f for _, f in trials))

    print(f"\nDuty Factor Analysis: {run_dir.name} ({terrain})")
    print(f"NPZ: {npz_path.name}")
    print(f"{'=' * 105}\n")

    print(f"{'Scene':<14} {'Freq':>5}  {'Duty Factor':>12}  {'Mean #Contact':>14}  "
          f"{'0-leg':>6} {'1-leg':>6} {'2-leg':>6} {'3-leg':>6} {'4-leg':>6}  "
          f"{'N':>3} {'skip':>4}")
    print("-" * 105)

    results = []

    for scene in all_scenes:
        label = SCENE_LABELS.get(scene, scene)
        for freq in all_freqs:
            tidxs = trials.get((scene, freq))
            if not tidxs:
                continue

            duty_factors = []
            mean_contacts = []
            contact_hists = []
            n_skipped = 0

            for t in tidxs:
                prefix = f"{scene}_f{freq:g}_t{t}"
                time = npz[f"{prefix}_time"]
                pos_x = npz[f"{prefix}_pos_x"]
                pitch = npz[f"{prefix}_pitch"]
                contact = npz[f"{prefix}_leg_in_contact"]  # (T, 4) bool

                if not is_valid_trial(pos_x, pitch, terrain, scene, freq):
                    n_skipped += 1
                    continue

                mask = active_mask(time, pos_x, terrain)
                if mask.sum() < 10:
                    n_skipped += 1
                    continue

                contact_active = contact[mask]
                n_legs = contact_active.sum(axis=1)  # (N,) 0-4

                df = float((n_legs >= 1).mean())
                duty_factors.append(df)
                mean_contacts.append(float(n_legs.mean()))

                hist = np.bincount(n_legs, minlength=5)[:5]
                contact_hists.append(hist / hist.sum())

            if not duty_factors:
                continue

            df_mean = np.mean(duty_factors)
            df_std = np.std(duty_factors)
            mc_mean = np.mean(mean_contacts)
            mc_std = np.std(mean_contacts)
            hist_mean = np.mean(contact_hists, axis=0)

            results.append({
                "scene": scene, "label": label, "freq": freq,
                "duty_factor_mean": df_mean, "duty_factor_std": df_std,
                "mean_contacts": mc_mean, "mean_contacts_std": mc_std,
                "hist": hist_mean, "n_trials": len(duty_factors),
            })

            freq_str = f"{freq:g}Hz"
            print(
                f"{label:<14} {freq_str:>5}  "
                f"{df_mean:>5.3f} +/- {df_std:>5.3f}  "
                f"{mc_mean:>6.3f} +/- {mc_std:>5.3f}  "
                f"{hist_mean[0]:>5.1%} {hist_mean[1]:>5.1%} {hist_mean[2]:>5.1%} "
                f"{hist_mean[3]:>5.1%} {hist_mean[4]:>5.1%}  "
                f"{len(duty_factors):>3} {n_skipped:>4}"
            )

    _print_summary(all_scenes, results)
    npz.close()
    return results


def _print_summary(all_scenes, results):
    print(f"\n{'--- Summary by morphology (averaged across freqs) ---':^105}")
    print(f"{'Morphology':<14} {'Duty Factor':>12}  {'Mean #Contact':>14}")
    print("-" * 45)
    for scene in all_scenes:
        scene_res = [r for r in results if r["scene"] == scene]
        if not scene_res:
            continue
        label = SCENE_LABELS.get(scene, scene)
        df_all = np.mean([r["duty_factor_mean"] for r in scene_res])
        mc_all = np.mean([r["mean_contacts"] for r in scene_res])
        print(f"{label:<14} {df_all:>12.3f}  {mc_all:>14.3f}")


def _print_cross_terrain(all_results, key, label):
    print(f"\n{'=' * 80}")
    print(f"Cross-terrain {label.lower()} comparison (morphology average)")
    print(f"{'=' * 80}")

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
        lbl = SCENE_LABELS.get(scene, scene)
        row = f"{lbl:<14}"
        for t in terrains:
            scene_res = [r for r in all_results[t] if r["scene"] == scene]
            if scene_res:
                val = np.mean([r[key] for r in scene_res])
                row += f"  {val:>10.3f}"
            else:
                row += f"  {'n/a':>10}"
        print(row)


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
        _print_cross_terrain(all_results, "duty_factor_mean", "Duty Factor")


if __name__ == "__main__":
    main()
