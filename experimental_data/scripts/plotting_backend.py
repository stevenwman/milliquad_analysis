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

    def _finish(self, fig: plt.Figure, file_stem: str) -> None:
        fig.tight_layout()
        if self.save:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_dir / f"{file_stem}.png", dpi=160)
        if self.show:
            plt.show()
        else:
            plt.close(fig)

    def plot_flat(self, p: FlatPlotInputs, file_stem: str) -> None:
        fig, axs = plt.subplots(4, 1, figsize=(8, 12))
        labels = [f"Trial {i+1}" for i in range(len(p.trials))]

        for i, arr in enumerate(p.trials):
            axs[0].plot(p.t, arr, color=self.colors[i % 10], linewidth=1.2, label=labels[i])
        axs[0].plot(p.t, p.speed_mean, color="black", linewidth=1.5, label="Mean")
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
        axs[0].set_ylim(-50, 600)
        axs[0].set_ylabel("v_x [mm/s]")
        axs[0].set_title(f"Forward Speed vs. Time - {p.title_prefix}")
        axs[0].legend(loc="best", fontsize=8)
        axs[0].grid(True)

        self._plot_steady_band(
            ax=axs[1],
            t=p.t,
            trials=p.y,
            mean_curve=p.y_mean,
            mean=p.y_steady_mean,
            std=p.y_steady_std,
            ylabel="height [mm]",
            title=f"Body Height vs. Time - {p.title_prefix}",
            margin=0.5,
        )

        self._plot_steady_band(
            ax=axs[2],
            t=p.t,
            trials=p.theta,
            mean_curve=p.theta_mean,
            mean=p.theta_steady_mean,
            std=p.theta_steady_std,
            ylabel="theta",
            title=f"Body Angle vs. Time - {p.title_prefix}",
            margin=5.0,
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
            margin=20000.0,
        )
        axs[3].set_xlabel("Time [s]")

        self._finish(fig, file_stem)

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
        ax.plot(t, mean_curve, color="black", linewidth=1.5)
        lo = mean - std
        hi = mean + std
        ax.axhspan(lo, hi, color="tab:orange", alpha=0.30)
        ax.axhline(mean, linestyle=":", linewidth=1.0, color="black")
        ax.axhline(hi, linestyle=":", linewidth=1.0, color="black")
        ax.axhline(lo, linestyle=":", linewidth=1.0, color="black")
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    def plot_steps(self, p: StepsPlotInputs, file_stem: str) -> None:
        fig, axs = plt.subplots(4, 1, figsize=(8, 12))
        n_trials = p.speed_mat.shape[1]
        labels = [f"Trial {i+1}" for i in range(n_trials)] + ["Mean"]

        mats = [p.speed_mat, p.y_mat, p.theta_mat, p.omega_mat]
        means = [p.speed_mean, p.y_mean, p.theta_mean, p.omega_mean]
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
            ax.plot(p.t, means[idx], color="black", linewidth=2.0)
            ax.set_ylabel(ylabels[idx])
            ax.set_title(titles[idx])
            ax.grid(True)
            if idx == 0:
                handles = ax.lines[: n_trials + 1]
                ax.legend(handles, labels, loc="best", fontsize=8)
        axs[2].set_ylim(-30, 15)
        axs[3].set_xlabel("Time [s]")
        self._finish(fig, file_stem)
