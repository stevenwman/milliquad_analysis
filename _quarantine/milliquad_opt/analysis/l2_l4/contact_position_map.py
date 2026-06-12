#!/usr/bin/env python3
"""H6: Contact position mapping analysis.

Reports where on the terrain legs make contact. Computes mean contact height
and spatial distribution of contacts per morphology.

Primarily useful for rough terrain where terrain geometry varies spatially.

Usage:
    cd milliquad_opt
    uv run python -m analysis.l2_l4.contact_position_map results/20260228T202903_rough_spatial_rk4
    uv run python -m analysis.l2_l4.contact_position_map \
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

    print(f"\nContact Position Analysis: {run_dir.name} ({terrain})")
    print(f"{'=' * 105}\n")

    print(f"{'Scene':<14} {'Freq':>5}  {'Mean z mm':>10}  {'Std z mm':>9}  "
          f"{'Mean y mm':>10}  {'Std y mm':>9}  "
          f"{'#contacts':>10}  {'N':>3} {'skip':>4}")
    print("-" * 105)

    results = []

    for scene in all_scenes:
        label = SCENE_LABELS.get(scene, scene)
        for freq in all_freqs:
            tidxs = trials.get((scene, freq))
            if not tidxs:
                continue

            mean_zs = []
            std_zs = []
            mean_ys = []
            std_ys = []
            contact_counts = []
            n_skipped = 0

            for t in tidxs:
                prefix = f"{scene}_f{freq:g}_t{t}"
                time = npz[f"{prefix}_time"]
                pos_x = npz[f"{prefix}_pos_x"]
                pitch = npz[f"{prefix}_pitch"]
                contact = npz[f"{prefix}_leg_in_contact"]       # (T, 4) bool
                contact_pos = npz[f"{prefix}_leg_contact_pos"]  # (T, 4, 3)

                if not is_valid_trial(pos_x, pitch, terrain, scene, freq):
                    n_skipped += 1
                    continue

                mask = active_mask(time, pos_x, terrain)
                if mask.sum() < 10:
                    n_skipped += 1
                    continue

                contact_a = contact[mask]       # (N, 4)
                cpos_a = contact_pos[mask]      # (N, 4, 3)

                # Extract contact positions where leg is actually in contact
                # cpos_a[i, j, :] is meaningful only when contact_a[i, j] is True
                in_contact = contact_a  # (N, 4) bool
                if not in_contact.any():
                    continue

                # Gather all contact z and y values (flat array)
                z_vals = cpos_a[:, :, 2][in_contact]  # z positions
                y_vals = cpos_a[:, :, 1][in_contact]  # y positions

                mean_zs.append(float(z_vals.mean()) * 1000)  # m → mm
                std_zs.append(float(z_vals.std()) * 1000)
                mean_ys.append(float(y_vals.mean()) * 1000)
                std_ys.append(float(y_vals.std()) * 1000)
                contact_counts.append(int(in_contact.sum()))

            if not mean_zs:
                continue

            mz = np.mean(mean_zs)
            sz = np.mean(std_zs)
            my = np.mean(mean_ys)
            sy = np.mean(std_ys)
            cc = np.mean(contact_counts)

            results.append({
                "scene": scene, "label": label, "freq": freq,
                "mean_z": mz, "std_z": sz,
                "mean_y": my, "std_y": sy,
                "contact_count": cc,
                "n_trials": len(mean_zs),
            })

            freq_str = f"{freq:g}Hz"
            print(
                f"{label:<14} {freq_str:>5}  "
                f"{mz:>10.3f}  {sz:>9.3f}  "
                f"{my:>10.3f}  {sy:>9.3f}  "
                f"{cc:>10.0f}  "
                f"{len(mean_zs):>3} {n_skipped:>4}"
            )

    _print_summary(all_scenes, results)
    npz.close()
    return results


def _print_summary(all_scenes, results):
    print(f"\n{'--- Summary by morphology ---':^105}")
    print(f"{'Morphology':<14} {'Mean z mm':>10}  {'Std z mm':>9}  {'Mean y mm':>10}  {'Std y mm':>9}")
    print("-" * 58)
    for scene in all_scenes:
        scene_res = [r for r in results if r["scene"] == scene]
        if not scene_res:
            continue
        label = SCENE_LABELS.get(scene, scene)
        mz = np.mean([r["mean_z"] for r in scene_res])
        sz = np.mean([r["std_z"] for r in scene_res])
        my = np.mean([r["mean_y"] for r in scene_res])
        sy = np.mean([r["std_y"] for r in scene_res])
        print(f"{label:<14} {mz:>10.3f}  {sz:>9.3f}  {my:>10.3f}  {sy:>9.3f}")


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
        print(f"\n{'=' * 105}")
        print("Cross-terrain contact height comparison (morphology average)")
        print(f"{'=' * 105}")

        terrains = list(all_results.keys())
        header = f"{'Morphology':<14}" + "".join(f"  {'z_' + t:>10}" for t in terrains)
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
                    mz = np.mean([r["mean_z"] for r in scene_res])
                    row += f"  {mz:>8.3f}mm"
                else:
                    row += f"  {'n/a':>10}"
            print(row)


if __name__ == "__main__":
    main()
