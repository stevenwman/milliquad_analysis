from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotting_backend import FlatPlotInputs, StepsPlotInputs


class PlotlyBackend:
    def __init__(self, out_dir: Path, show: bool, save: bool) -> None:
        self.out_dir = out_dir
        self.show = show
        self.save = save
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def _finish(self, fig: go.Figure, file_stem: str) -> None:
        fig.update_layout(
            template="plotly_white",
            width=1280,
            height=1150,
            legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
            margin=dict(l=70, r=220, t=80, b=60),
        )
        if self.save:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.write_html(self.out_dir / f"{file_stem}.html", include_plotlyjs="cdn")
        if self.show:
            fig.show()

    def plot_flat(self, p: FlatPlotInputs, file_stem: str) -> None:
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                f"Forward Speed vs. Time - {p.title_prefix}",
                f"Body Height vs. Time - {p.title_prefix}",
                f"Body Angle vs. Time - {p.title_prefix}",
                f"Angular Velocity vs. Time - {p.title_prefix}",
            ),
        )

        for i, arr in enumerate(p.trials):
            fig.add_trace(
                go.Scatter(x=p.t, y=arr, mode="lines", name=f"Trial {i+1}", line=dict(color=self.colors[i % len(self.colors)])),
                row=1,
                col=1,
            )
        v_lo = p.speed_steady_mean - p.speed_steady_std
        v_hi = p.speed_steady_mean + p.speed_steady_std
        fig.add_hrect(y0=v_lo, y1=v_hi, fillcolor="#ff7f0e", opacity=0.25, line_width=0, row=1, col=1)
        fig.add_hline(y=p.speed_steady_mean, line_dash="dash", line_color="red", row=1, col=1)
        if np.isfinite(p.t_reach_avg):
            fig.add_vline(x=p.t_reach_avg, line_dash="dash", line_color="blue", row=1, col=1)
        v_all = np.concatenate([arr[np.isfinite(arr)] for arr in p.trials if np.any(np.isfinite(arr))])
        if v_all.size > 0:
            vmin, vmax = float(np.min(v_all)), float(np.max(v_all))
            vpad = max(20.0, 0.15 * (vmax - vmin))
            fig.update_yaxes(range=[vmin - vpad, vmax + vpad], row=1, col=1)
        fig.update_yaxes(title_text="v_x [mm/s]", row=1, col=1)

        self._flat_trials_only_panel(fig, row=2, t=p.t, trials=p.y, ylabel="height [mm]", margin=1.0)
        self._flat_trials_only_panel(fig, row=3, t=p.t, trials=p.theta, ylabel="theta", margin=8.0)
        self._flat_band_panel(fig, row=4, t=p.t, trials=p.omega, mean=p.omega_steady_mean, std=p.omega_steady_std, ylabel="omega", margin=30000.0)
        fig.update_xaxes(title_text="Time [s]", row=4, col=1)

        self._finish(fig, file_stem)

    def _flat_band_panel(
        self,
        fig: go.Figure,
        row: int,
        t: np.ndarray,
        trials: list[np.ndarray],
        mean: float,
        std: float,
        ylabel: str,
        margin: float,
    ) -> None:
        for i, arr in enumerate(trials):
            fig.add_trace(
                go.Scatter(x=t, y=arr, mode="lines", showlegend=False, line=dict(color=self.colors[i % len(self.colors)])),
                row=row,
                col=1,
            )
        lo = mean - std
        hi = mean + std
        fig.add_hrect(y0=lo, y1=hi, fillcolor="#ff7f0e", opacity=0.25, line_width=0, row=row, col=1)
        fig.add_hline(y=mean, line_dash="dot", line_color="black", row=row, col=1)
        fig.add_hline(y=hi, line_dash="dot", line_color="black", row=row, col=1)
        fig.add_hline(y=lo, line_dash="dot", line_color="black", row=row, col=1)
        finite_blocks = [arr[np.isfinite(arr)] for arr in trials if np.any(np.isfinite(arr))]
        if finite_blocks:
            tmin = float(np.min(np.concatenate(finite_blocks)))
            tmax = float(np.max(np.concatenate(finite_blocks)))
            lo_ref = min(lo, tmin)
            hi_ref = max(hi, tmax)
            pad = max(margin, 0.08 * (hi_ref - lo_ref))
            fig.update_yaxes(range=[lo_ref - pad, hi_ref + pad], row=row, col=1)
        fig.update_yaxes(title_text=ylabel, row=row, col=1)

    def _flat_trials_only_panel(
        self,
        fig: go.Figure,
        row: int,
        t: np.ndarray,
        trials: list[np.ndarray],
        ylabel: str,
        margin: float,
    ) -> None:
        for i, arr in enumerate(trials):
            fig.add_trace(
                go.Scatter(x=t, y=arr, mode="lines", showlegend=False, line=dict(color=self.colors[i % len(self.colors)])),
                row=row,
                col=1,
            )
        finite_blocks = [arr[np.isfinite(arr)] for arr in trials if np.any(np.isfinite(arr))]
        if finite_blocks:
            lo = float(np.min(np.concatenate(finite_blocks)))
            hi = float(np.max(np.concatenate(finite_blocks)))
            pad = max(margin, 0.08 * (hi - lo))
            fig.update_yaxes(range=[lo - pad, hi + pad], row=row, col=1)
        fig.update_yaxes(title_text=ylabel, row=row, col=1)

    def plot_steps(self, p: StepsPlotInputs, file_stem: str) -> None:
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                f"Forward Speed vs. Time - {p.title_prefix}",
                "Body Height vs. Time",
                "Body Angle vs. Time",
                "Rotation vs. Time",
            ),
        )
        mats = [p.speed_mat, p.y_mat, p.theta_mat, p.omega_mat]
        ylabels = ["v [mm/s]", "height [mm]", "theta", "rot"]
        for row_idx, mat in enumerate(mats, start=1):
            n_trials = mat.shape[1]
            for i in range(n_trials):
                fig.add_trace(
                    go.Scatter(
                        x=p.t,
                        y=mat[:, i],
                        mode="lines",
                        name=f"Trial {i+1}" if row_idx == 1 else None,
                        showlegend=(row_idx == 1),
                        line=dict(color=self.colors[i % len(self.colors)]),
                    ),
                    row=row_idx,
                    col=1,
                )
            fig.update_yaxes(title_text=ylabels[row_idx - 1], row=row_idx, col=1)

        theta_finite = p.theta_mat[np.isfinite(p.theta_mat)]
        if theta_finite.size > 0:
            th_min = float(np.min(theta_finite))
            th_max = float(np.max(theta_finite))
            th_pad = max(3.0, 0.10 * (th_max - th_min))
            fig.update_yaxes(range=[th_min - th_pad, th_max + th_pad], row=3, col=1)

        fig.update_xaxes(title_text="Time [s]", row=4, col=1)
        self._finish(fig, file_stem)
