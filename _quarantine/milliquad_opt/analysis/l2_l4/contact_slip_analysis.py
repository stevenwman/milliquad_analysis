#!/usr/bin/env python3
"""H2: Contact slip fraction analysis.

For each leg-terrain contact, computes slip ratio = F_tangent / (mu * F_normal).
Ratio = 1.0 means at friction cone boundary (slipping).

Reports fraction of contact-timesteps near or at slip per morphology.

Usage:
    cd milliquad_opt
    uv run python -m analysis.l2_l4.contact_slip_analysis results/20260228T013353_rk4_flat
    uv run python -m analysis.l2_l4.contact_slip_analysis \
        results/20260228T013353_rk4_flat results/20260228T202903_rough_spatial_rk4
"""

from __future__ import annotations

import argparse
import csv
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
SLIP_THRESHOLD = 0.95  # ratio above which we call "near-slip"


def _load_sliding_friction(run_dir: pathlib.Path) -> float:
    """Read sliding_friction from optimization_bests.csv (last row)."""
    import sys
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    if not rows:
        sys.exit(f"No rows in {bests_csv}")
    return float(rows[-1]["sliding_friction"])


def analyze(run_dir: pathlib.Path):
    terrain = detect_terrain(run_dir)
    npz_path = find_npz(run_dir)
    npz = np.load(npz_path)
    mu = _load_sliding_friction(run_dir)

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

    print(f"\nSlip Analysis: {run_dir.name} ({terrain})")
    print(f"Sliding friction mu = {mu:.4f}")
    print(f"Slip threshold: ratio >= {SLIP_THRESHOLD}")
    print(f"{'=' * 95}\n")

    print(f"{'Scene':<14} {'Freq':>5}  {'Slip Frac':>10}  {'Mean Ratio':>11}  "
          f"{'Med Ratio':>10}  {'Mean F_n':>8}  {'Mean F_t':>8}  {'N':>3} {'skip':>4}")
    print("-" * 95)

    results = []

    for scene in all_scenes:
        label = SCENE_LABELS.get(scene, scene)
        for freq in all_freqs:
            tidxs = trials.get((scene, freq))
            if not tidxs:
                continue

            slip_fracs = []
            mean_ratios = []
            median_ratios = []
            mean_fn_list = []
            mean_ft_list = []
            n_skipped = 0

            for t in tidxs:
                prefix = f"{scene}_f{freq:g}_t{t}"
                time = npz[f"{prefix}_time"]
                pos_x = npz[f"{prefix}_pos_x"]
                pitch = npz[f"{prefix}_pitch"]
                contact = npz[f"{prefix}_leg_in_contact"]
                fn = npz[f"{prefix}_leg_normal_force"]
                ft = npz[f"{prefix}_leg_tangent_force"]

                if not is_valid_trial(pos_x, pitch, terrain, scene, freq):
                    n_skipped += 1
                    continue

                mask = active_mask(time, pos_x, terrain)
                if mask.sum() < 10:
                    n_skipped += 1
                    continue

                contact_active = contact[mask]
                fn_active = fn[mask]
                ft_active = ft[mask]

                # Per-leg, per-timestep: only where leg is in contact
                in_contact = contact_active
                fn_contact = fn_active[in_contact]
                ft_contact = ft_active[in_contact]

                if len(fn_contact) == 0:
                    continue

                valid = fn_contact > 1e-10
                if valid.sum() == 0:
                    continue

                ratio = np.zeros_like(fn_contact)
                ratio[valid] = ft_contact[valid] / (mu * fn_contact[valid])

                slip_fracs.append(float((ratio[valid] >= SLIP_THRESHOLD).mean()))
                mean_ratios.append(float(ratio[valid].mean()))
                median_ratios.append(float(np.median(ratio[valid])))
                mean_fn_list.append(float(fn_contact[valid].mean()))
                mean_ft_list.append(float(ft_contact[valid].mean()))

            if not slip_fracs:
                continue

            sf_mean = np.mean(slip_fracs)
            sf_std = np.std(slip_fracs)
            mr_mean = np.mean(mean_ratios)
            med_mean = np.mean(median_ratios)
            fn_mean = np.mean(mean_fn_list)
            ft_mean = np.mean(mean_ft_list)

            results.append({
                "scene": scene, "label": label, "freq": freq,
                "slip_frac_mean": sf_mean, "slip_frac_std": sf_std,
                "mean_ratio": mr_mean, "median_ratio": med_mean,
                "mean_fn": fn_mean, "mean_ft": ft_mean,
                "n_trials": len(slip_fracs),
            })

            freq_str = f"{freq:g}Hz"
            print(
                f"{label:<14} {freq_str:>5}  "
                f"{sf_mean:>5.1%}+/-{sf_std:>4.1%}  "
                f"{mr_mean:>11.3f}  {med_mean:>10.3f}  "
                f"{fn_mean:>7.1e}  {ft_mean:>7.1e}  "
                f"{len(slip_fracs):>3} {n_skipped:>4}"
            )

    _print_summary(all_scenes, results)
    npz.close()
    return results


def _print_summary(all_scenes, results):
    print(f"\n{'--- Summary by morphology ---':^95}")
    print(f"{'Morphology':<14} {'Slip Frac':>10}  {'Mean Ratio':>11}  {'Mean F_n (N)':>12}  {'Mean F_t (N)':>12}")
    print("-" * 65)
    for scene in all_scenes:
        scene_res = [r for r in results if r["scene"] == scene]
        if not scene_res:
            continue
        label = SCENE_LABELS.get(scene, scene)
        sf = np.mean([r["slip_frac_mean"] for r in scene_res])
        mr = np.mean([r["mean_ratio"] for r in scene_res])
        fn = np.mean([r["mean_fn"] for r in scene_res])
        ft = np.mean([r["mean_ft"] for r in scene_res])
        print(f"{label:<14} {sf:>10.1%}  {mr:>11.3f}  {fn:>12.2e}  {ft:>12.2e}")


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
        print(f"\n{'=' * 95}")
        print("Cross-terrain slip fraction comparison (morphology average)")
        print(f"{'=' * 95}")

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
                    sf = np.mean([r["slip_frac_mean"] for r in scene_res])
                    row += f"  {sf:>9.1%}"
                else:
                    row += f"  {'n/a':>10}"
            print(row)


if __name__ == "__main__":
    main()
