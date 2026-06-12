from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from pipeline_common import MM_SCALE, base_dir, load_trial_csv
from plotting_backend import FlatPlotInputs, MatplotlibBackend
from plotly_backend import PlotlyBackend


@dataclass(frozen=True)
class FlatCondition:
    files: list[str]
    trial_idx_1based: list[int]
    title: str
    points: int
    steady_t: float


def _extract_flat(file_name: str, points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dat = load_trial_csv(file_name)
    dat = dat[:points, :]
    vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))
    y_raw = 0.5 * (dat[:, 2] + dat[:, 6])
    y = (y_raw - y_raw[2] + np.min(y_raw)) * MM_SCALE
    theta = dat[:, 11]
    omega = dat[:, 10]
    t = dat[:, 0]
    return t, vx, y, theta, omega


def run_condition(cond: FlatCondition, backend: MatplotlibBackend) -> None:
    trials = [cond.files[i - 1] for i in cond.trial_idx_1based]
    t = None
    vx_list, y_list, theta_list, omega_list = [], [], [], []
    for name in trials:
        t_i, vx, y, theta, omega = _extract_flat(name, cond.points)
        if t is None:
            t = t_i
        vx_list.append(vx)
        y_list.append(y)
        theta_list.append(theta)
        omega_list.append(omega)
    assert t is not None

    # MATLAB assumes matched lengths, but some trials differ slightly.
    # Use the shared minimum length to preserve intent without failing.
    min_len = min(arr.shape[0] for arr in [t, *vx_list, *y_list, *theta_list, *omega_list])
    t = t[:min_len]
    vx_list = [v[:min_len] for v in vx_list]
    y_list = [v[:min_len] for v in y_list]
    theta_list = [v[:min_len] for v in theta_list]
    omega_list = [v[:min_len] for v in omega_list]

    vx_mat = np.column_stack(vx_list)
    y_mat = np.column_stack(y_list)
    theta_mat = np.column_stack(theta_list)
    omega_mat = np.column_stack(omega_list)

    vx_mean = np.mean(vx_mat, axis=1)
    y_mean = np.mean(y_mat, axis=1)
    theta_mean = np.mean(theta_mat, axis=1)
    omega_mean = np.mean(omega_mat, axis=1)

    idx_steady = t > cond.steady_t
    vx_trial_steady = np.array([np.nanmean(v[idx_steady]) for v in vx_list], dtype=float)
    vx_steady_mean = float(np.nanmean(vx_trial_steady))
    vx_steady_std = float(np.nanstd(vx_trial_steady, ddof=0))

    idx_cross = np.where(vx_mean >= vx_steady_mean)[0]
    t_reach_avg = float(t[idx_cross[0]]) if idx_cross.size > 0 else float("nan")

    y_steady = y_mat[idx_steady, :].reshape(-1)
    theta_steady = theta_mat[idx_steady, :].reshape(-1)
    omega_steady = omega_mat[idx_steady, :].reshape(-1)

    payload = FlatPlotInputs(
        t=t,
        trials=vx_list,
        y=y_list,
        theta=theta_list,
        omega=omega_list,
        speed_mean=vx_mean,
        y_mean=y_mean,
        theta_mean=theta_mean,
        omega_mean=omega_mean,
        speed_steady_mean=vx_steady_mean,
        speed_steady_std=vx_steady_std,
        t_reach_avg=t_reach_avg,
        y_steady_mean=float(np.nanmean(y_steady)),
        y_steady_std=float(np.nanstd(y_steady, ddof=0)),
        theta_steady_mean=float(np.nanmean(theta_steady)),
        theta_steady_std=float(np.nanstd(theta_steady, ddof=0)),
        omega_steady_mean=float(np.nanmean(omega_steady)),
        omega_steady_std=float(np.nanstd(omega_steady, ddof=0)),
        title_prefix=cond.title,
    )
    backend.plot_flat(payload, file_stem=cond.title.lower().replace(" ", "_").replace("-", ""))


def build_conditions() -> list[FlatCondition]:
    return [
        FlatCondition(["f10leg1-1.csv", "f10leg2-2.csv", "f10leg3-3.csv", "f10leg4-4.csv"], [1, 2, 3], "10Hz Leg", 2500, 0.3),
        FlatCondition(["f102leg1-1.csv", "f102leg2-2.csv", "f102leg3-3.csv", "f102leg4-4.csv"], [1, 2, 4], "10Hz 2-Legged", 1480, 0.3),
        FlatCondition(["f104leg1-1.csv", "f104leg2-2.csv", "f104leg3-3.csv", "f104leg4-4.csv"], [1, 2, 4], "10Hz 4-Legged", 1199, 0.3),
        FlatCondition(["f10w1-1.csv", "f10w2-2.csv", "f10w3-3.csv", "f10w4-4.csv"], [1, 2, 3], "10Hz wheel", 910, 0.3),
        FlatCondition(["f30leg1-1.csv", "f30leg2-2.csv", "f30leg3-3.csv", "f30leg4-4.csv"], [1, 2, 3, 4], "30Hz Leg", 1100, 0.15),
        FlatCondition(["f302leg1-1.csv", "f302leg2-2.csv", "f302leg3-3.csv", "f302leg4-4.csv"], [1, 2, 3], "30Hz 2-Legged", 760, 0.3),
        FlatCondition(["f304leg1-1.csv", "f304leg2-2.csv", "f304leg3-3.csv", "f304leg4-4.csv"], [1, 2, 3, 4], "30Hz 4-Legged", 550, 0.3),
        FlatCondition(["f30w1-1.csv", "f30w2-2.csv", "f30w3-3.csv", "f30w4-4.csv"], [1, 2, 3, 4], "30Hz wheel", 350, 0.3),
        FlatCondition(["f50leg1-1.csv", "f50leg2-2.csv", "f50leg3-3.csv"], [1, 2, 3], "50Hz Leg", 1960, 0.35),
        FlatCondition(["f502leg1-1.csv", "f502leg2-2.csv", "f502leg3-3.csv"], [1, 2, 3], "50Hz 2-Legged", 1280, 0.35),
        FlatCondition(["f504leg1-1.csv", "f504leg2-2.csv", "f504leg3-3.csv"], [1, 2, 3], "50Hz 4-Legged", 1060, 0.35),
        FlatCondition(["f50w1-1.csv", "f50w2-2.csv", "f50w3-3.csv"], [1, 2, 3], "50Hz wheel", 620, 0.25),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Flat-ground pipeline port of Flat.m/plotFlat.m")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG outputs")
    parser.add_argument("--backend", choices=["matplotlib", "plotly"], default="matplotlib")
    args = parser.parse_args()

    if args.backend == "plotly":
        out_dir = base_dir() / "plots" / "flat_html"
        backend = PlotlyBackend(out_dir=out_dir, show=args.show, save=not args.no_save)
    else:
        out_dir = base_dir() / "plots" / "flat"
        backend = MatplotlibBackend(out_dir=out_dir, show=args.show, save=not args.no_save)
    for cond in build_conditions():
        run_condition(cond, backend)
    print(f"Flat pipeline complete. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
