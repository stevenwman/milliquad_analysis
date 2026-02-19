from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from pipeline_common import MM_SCALE, base_dir, clean_for_interp, load_trial_csv
from plotting_backend import MatplotlibBackend, StepsPlotInputs


@dataclass(frozen=True)
class StepsCondition:
    files: list[str]
    trial_idx_1based: list[int]
    title: str


def _extract_steps(file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dat = load_trial_csv(file_name)
    t = dat[:, 0]
    vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))
    vy = 0.5 * ((-dat[:, 4] * MM_SCALE) + (-dat[:, 8] * MM_SCALE))
    v = np.sqrt(vx**2 + vy**2)
    y = 0.5 * (dat[:, 2] + dat[:, 6]) * MM_SCALE
    theta = dat[:, 11]
    omega = dat[:, 10]
    return t, v, y, theta, omega


def _common_time(t_list: list[np.ndarray]) -> np.ndarray:
    dt_trials = []
    t_starts = []
    t_ends = []
    for t in t_list:
        t = t[np.isfinite(t)]
        if t.size < 2:
            continue
        t = np.unique(np.sort(t))
        if t.size < 2:
            continue
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            dt_trials.append(np.median(dt))
            t_starts.append(t[0])
            t_ends.append(t[-1])
    dt_common = float(np.median(np.array(dt_trials)))
    t0 = float(np.min(np.array(t_starts)))
    t1 = float(np.max(np.array(t_ends)))
    return np.arange(t0, t1 + dt_common * 0.5, dt_common)


def _interp_to(t_common: np.ndarray, t_raw: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
    t_clean, x_clean = clean_for_interp(t_raw, x_raw)
    if t_clean.size < 2:
        return np.full_like(t_common, np.nan, dtype=float)
    return np.interp(t_common, t_clean, x_clean, left=np.nan, right=np.nan)


def _mean_omit_nan_rows(mat: np.ndarray) -> np.ndarray:
    out = np.full(mat.shape[0], np.nan, dtype=float)
    valid_rows = np.any(np.isfinite(mat), axis=1)
    if np.any(valid_rows):
        out[valid_rows] = np.nanmean(mat[valid_rows, :], axis=1)
    return out


def run_condition(cond: StepsCondition, backend: MatplotlibBackend) -> None:
    trials = [cond.files[i - 1] for i in cond.trial_idx_1based]
    t_list, v_list, y_list, theta_list, omega_list = [], [], [], [], []
    for name in trials:
        t, v, y, theta, omega = _extract_steps(name)
        t_list.append(t)
        v_list.append(v)
        y_list.append(y)
        theta_list.append(theta)
        omega_list.append(omega)

    t_common = _common_time(t_list)
    v_mat = np.column_stack([_interp_to(t_common, t_list[i], v_list[i]) for i in range(len(trials))])
    y_mat = np.column_stack([_interp_to(t_common, t_list[i], y_list[i]) for i in range(len(trials))])
    theta_mat = np.column_stack([_interp_to(t_common, t_list[i], theta_list[i]) for i in range(len(trials))])
    omega_mat = np.column_stack([_interp_to(t_common, t_list[i], omega_list[i]) for i in range(len(trials))])

    payload = StepsPlotInputs(
        t=t_common,
        speed_mat=v_mat,
        y_mat=y_mat,
        theta_mat=theta_mat,
        omega_mat=omega_mat,
        speed_mean=_mean_omit_nan_rows(v_mat),
        y_mean=_mean_omit_nan_rows(y_mat),
        theta_mean=_mean_omit_nan_rows(theta_mat),
        omega_mean=_mean_omit_nan_rows(omega_mat),
        title_prefix=cond.title,
    )
    backend.plot_steps(payload, file_stem=cond.title.lower().replace(" ", "_").replace("-", ""))


def build_conditions() -> list[StepsCondition]:
    return [
        StepsCondition(["s10leg1-1.csv", "s10leg2-2.csv", "s10leg3-3.csv"], [1, 2, 3], "10Hz Leg"),
        StepsCondition(["s102leg1-1.csv", "s102leg2-2.csv", "s102leg3-3.csv"], [1, 2, 3], "10Hz 2-Legged"),
        StepsCondition(["s104leg1-1.csv", "s104leg2-2.csv", "s104leg3-3.csv"], [1, 2, 3], "10Hz 4-Legged"),
        StepsCondition(["s20leg1-1.csv", "s20leg2-2.csv", "s20leg3-3.csv"], [1, 2, 3], "20Hz Leg"),
        StepsCondition(["s202leg1-1.csv", "s202leg2-2.csv", "s202leg3-3.csv"], [1, 2, 3], "20Hz 2-Legged"),
        StepsCondition(["s204leg1-1.csv", "s204leg2-2.csv", "s204leg3-3.csv"], [1, 2, 3], "20Hz 4-Legged"),
        StepsCondition(["s30leg1-1.csv", "s30leg2-2.csv", "s30leg3-3.csv"], [1, 2, 3], "30Hz Leg"),
        StepsCondition(["s302leg1-1.csv", "s302leg2-2.csv", "s302leg3-3.csv"], [1, 2, 3], "30Hz 2-Legged"),
        StepsCondition(["s304leg1-1.csv", "s304leg2-2.csv", "s304leg3-3.csv"], [1, 2, 3], "30Hz 4-Legged"),
        StepsCondition(["s30w1-1.csv", "s30w2-2.csv", "s30w3-3.csv"], [1, 2, 3], "30Hz wheel"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step pipeline port of Steps.m/plotSteps.m")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG outputs")
    args = parser.parse_args()

    out_dir = base_dir() / "plots" / "steps"
    backend = MatplotlibBackend(out_dir=out_dir, show=args.show, save=not args.no_save)
    for cond in build_conditions():
        run_condition(cond, backend)
    print(f"Steps pipeline complete. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
