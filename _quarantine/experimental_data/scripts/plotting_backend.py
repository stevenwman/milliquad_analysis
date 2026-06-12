from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FlatPlotInputs:
    t: np.ndarray
    trials: Sequence[np.ndarray]
    y: Sequence[np.ndarray]
    theta: Sequence[np.ndarray]
    omega: Sequence[np.ndarray]
    speed_mean: np.ndarray
    y_mean: np.ndarray
    theta_mean: np.ndarray
    omega_mean: np.ndarray
    speed_steady_mean: float
    speed_steady_std: float
    t_reach_avg: float
    y_steady_mean: float
    y_steady_std: float
    theta_steady_mean: float
    theta_steady_std: float
    omega_steady_mean: float
    omega_steady_std: float
    title_prefix: str


@dataclass(frozen=True)
class StepsPlotInputs:
    t: np.ndarray
    speed_mat: np.ndarray
    y_mat: np.ndarray
    theta_mat: np.ndarray
    omega_mat: np.ndarray
    speed_mean: np.ndarray
    y_mean: np.ndarray
    theta_mean: np.ndarray
    omega_mean: np.ndarray
    title_prefix: str


class MatplotlibBackend:
    def __init__(self, out_dir: Path, show: bool, save: bool) -> None:
        self.out_dir = out_dir
        self.show = show
        self.save = save
        self.colors = plt.get_cmap("tab10").colors

    def _finish(self, fig: plt.Figure, file_stem: str, right_margin: float = 1.0) -> None:
        fig.tight_layout(rect=(0.0, 0.0, right_margin, 1.0))
        if self.save:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_dir / f"{file_stem}.png", dpi=160)
        if self.show:
            plt.show()
        else:
            plt.close(fig)

    def plot_flat(self, p: FlatPlotInputs, file_stem: str) -> None:
        fig, axs = plt.subplots(4, 1, figsize=(11, 12))
        labels = [f"Trial {i+1}" for i in range(len(p.trials))]

        for i, arr in enumerate(p.trials):
            axs[0].plot(p.t, arr, color=self.colors[i % 10], linewidth=1.2, label=labels[i])
        lo = p.speed_steady_mean - p.speed_steady_std
        hi = p.speed_steady_mean + p.speed_steady_std
        axs[0].axhspan(lo, hi, color="tab:orange", alpha=0.30)
        axs[0].axhline(
            p.speed_steady_mean,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Steady mean={p.speed_steady_mean:.1f}, std={p.speed_steady_std:.1f}",
        )
        if np.isfinite(p.t_reach_avg):
            axs[0].axvline(
                p.t_reach_avg,
                color="blue",
                linestyle="--",
                linewidth=1.2,
                label=f"Reach mean @ {p.t_reach_avg * 1000.0:.2f} ms",
            )
        all_v = np.concatenate([arr[np.isfinite(arr)] for arr in p.trials])
        if all_v.size > 0:
            v_min = float(np.min(all_v))
            v_max = float(np.max(all_v))
            pad = max(20.0, 0.15 * (v_max - v_min))
            axs[0].set_ylim(v_min - pad, v_max + pad)
        axs[0].set_ylabel("v_x [mm/s]")
        axs[0].set_title(f"Forward Speed vs. Time - {p.title_prefix}")
        axs[0].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )
        axs[0].grid(True)

        self._plot_trials_only_panel(
            ax=axs[1],
            t=p.t,
            trials=p.y,
            ylabel="height [mm]",
            title=f"Body Height vs. Time - {p.title_prefix}",
            margin=1.0,
        )

        self._plot_trials_only_panel(
            ax=axs[2],
            t=p.t,
            trials=p.theta,
            ylabel="theta",
            title=f"Body Angle vs. Time - {p.title_prefix}",
            margin=8.0,
        )

        self._plot_steady_band(
            ax=axs[3],
            t=p.t,
            trials=p.omega,
            mean_curve=p.omega_mean,
            mean=p.omega_steady_mean,
            std=p.omega_steady_std,
            ylabel="omega",
            title=f"Angular Velocity vs. Time - {p.title_prefix}",
            margin=30000.0,
        )
        axs[3].set_xlabel("Time [s]")

        self._finish(fig, file_stem, right_margin=0.82)

    def _plot_trials_only_panel(
        self,
        ax: plt.Axes,
        t: np.ndarray,
        trials: Sequence[np.ndarray],
        ylabel: str,
        title: str,
        margin: float,
    ) -> None:
        for i, arr in enumerate(trials):
            ax.plot(t, arr, color=self.colors[i % 10], linewidth=1.2)
        finite_blocks = [arr[np.isfinite(arr)] for arr in trials if np.any(np.isfinite(arr))]
        if finite_blocks:
            lo = float(np.min(np.concatenate(finite_blocks)))
            hi = float(np.max(np.concatenate(finite_blocks)))
            pad = max(margin, 0.08 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    def _plot_steady_band(
        self,
        ax: plt.Axes,
        t: np.ndarray,
        trials: Sequence[np.ndarray],
        mean_curve: np.ndarray,
        mean: float,
        std: float,
        ylabel: str,
        title: str,
        margin: float,
    ) -> None:
        for i, arr in enumerate(trials):
            ax.plot(t, arr, color=self.colors[i % 10], linewidth=1.2)
        lo = mean - std
        hi = mean + std
        ax.axhspan(lo, hi, color="tab:orange", alpha=0.30)
        ax.axhline(mean, linestyle=":", linewidth=1.0, color="black")
        ax.axhline(hi, linestyle=":", linewidth=1.0, color="black")
        ax.axhline(lo, linestyle=":", linewidth=1.0, color="black")
        finite_blocks = [arr[np.isfinite(arr)] for arr in trials if np.any(np.isfinite(arr))]
        if finite_blocks:
            trial_min = float(np.min(np.concatenate(finite_blocks)))
            trial_max = float(np.max(np.concatenate(finite_blocks)))
            lo_ref = min(lo, trial_min)
            hi_ref = max(hi, trial_max)
            pad = max(margin, 0.08 * (hi_ref - lo_ref))
            ax.set_ylim(lo_ref - pad, hi_ref + pad)
        else:
            ax.set_ylim(lo - margin, hi + margin)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    def plot_steps(self, p: StepsPlotInputs, file_stem: str) -> None:
        fig, axs = plt.subplots(4, 1, figsize=(11, 12))
        n_trials = p.speed_mat.shape[1]
        labels = [f"Trial {i+1}" for i in range(n_trials)]

        mats = [p.speed_mat, p.y_mat, p.theta_mat, p.omega_mat]
        ylabels = ["v [mm/s]", "height [mm]", "theta", "rot"]
        titles = [
            f"Forward Speed vs. Time - {p.title_prefix}",
            "Body Height vs. Time",
            "Body Angle vs. Time",
            "Rotation vs. Time",
        ]

        for idx, ax in enumerate(axs):
            mat = mats[idx]
            for i in range(n_trials):
                ax.plot(p.t, mat[:, i], color=self.colors[i % 10], linewidth=1.0)
            ax.set_ylabel(ylabels[idx])
            ax.set_title(titles[idx])
            ax.grid(True)
            if idx == 0:
                handles = ax.lines[:n_trials]
                ax.legend(handles, labels, loc="best", fontsize=8)
        theta_finite = p.theta_mat[np.isfinite(p.theta_mat)]
        if theta_finite.size > 0:
            th_min = float(np.min(theta_finite))
            th_max = float(np.max(theta_finite))
            th_pad = max(3.0, 0.10 * (th_max - th_min))
            axs[2].set_ylim(th_min - th_pad, th_max + th_pad)
        axs[3].set_xlabel("Time [s]")
        self._finish(fig, file_stem)
