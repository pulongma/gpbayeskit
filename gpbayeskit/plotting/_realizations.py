"""
gpbayeskit.plotting._realizations
====================================
Visualise spatio-temporal GP realisations across time.

Functions
---------
plot_st_realizations
    Panel of spatial heatmaps — one per time step — showing how the
    simulated random field advects and evolves.  A velocity arrow and an
    optional tracer point (a reference location that drifts with the
    advection) are drawn on every subplot to make the movement tangible.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from ..simulation._spatiotemporal import STSimulationResult


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_st_realizations(
    result: STSimulationResult,
    ncols: int | None = None,
    cmap: str = "RdBu_r",
    shared_scale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    show_velocity: bool = True,
    velocity_scale: float = 0.25,
    show_tracer: bool = True,
    tracer_origin: tuple[float, float] | None = None,
    tracer_color: str = "gold",
    tracer_size: float = 60.0,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    subplot_titles: Sequence[str] | None = None,
    add_colorbar: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Panel of 2-D spatial heatmaps of a spatio-temporal GP realisation.

    One subplot is drawn per time step.  A velocity arrow on each panel
    shows the advection direction and relative magnitude; an optional
    tracer point drifts with the advection velocity to make the movement
    of the spatial field visually apparent.

    Parameters
    ----------
    result          : STSimulationResult
                      Output of ``simulate_st``.
    ncols           : int or None
                      Number of subplot columns.  Defaults to
                      ``min(n_t, 4)``.
    cmap            : str
                      Matplotlib colormap (diverging recommended).
    shared_scale    : bool
                      If True, all subplots share the same colour scale.
    vmin, vmax      : float or None
                      Override the automatic shared colour scale.
    show_velocity   : bool
                      Draw an advection arrow on each panel.
    velocity_scale  : float
                      Length of the velocity arrow as a fraction of the
                      spatial domain width.  Arrow length = lam0 * dt
                      (one time-step displacement); ``velocity_scale``
                      rescales it so the arrow fits the plot.
    show_tracer     : bool
                      Draw a marker that moves by  lam_vec * (t − t₀)
                      at each time step, tracking the advection of a
                      reference parcel of the medium.
    tracer_origin   : (x, y) or None
                      Starting position of the tracer in domain
                      coordinates.  Defaults to the domain centre.
    tracer_color    : str   Colour of the tracer marker.
    tracer_size     : float Marker area for the tracer scatter point.
    figsize         : (width, height) or None
                      Auto-computed if None.
    title           : str or None
                      Figure-level suptitle.  Auto-generated if None.
    subplot_titles  : sequence of str or None
                      Per-subplot titles.  Defaults to ``"t = {value:.3g}"``.
    add_colorbar    : bool
                      Whether to add a shared colorbar.

    Returns
    -------
    fig  : matplotlib Figure
    axes : flattened ndarray of Axes
    """
    n_t = result.n_t
    lam_vec = result.lam_vec
    lam0    = float(result.params.get("lam0", 0.0))
    theta0  = float(result.params.get("theta0", 0.0))
    t       = result.t
    S1      = result.S1      # (n_h, n_w)
    S2      = result.S2
    field   = result.field   # (n_t, n_h, n_w)

    lo1, hi1 = float(S1.min()), float(S1.max())
    lo2, hi2 = float(S2.min()), float(S2.max())
    domain_w = hi1 - lo1
    domain_h = hi2 - lo2
    centre   = (0.5 * (lo1 + hi1), 0.5 * (lo2 + hi2))

    # ── Colour scale ──────────────────────────────────────────────────────────
    if shared_scale:
        _vmin = vmin if vmin is not None else float(field.min())
        _vmax = vmax if vmax is not None else float(field.max())
        # Symmetric around zero for diverging cmaps
        abs_max  = max(abs(_vmin), abs(_vmax))
        _vmin, _vmax = -abs_max, abs_max
        norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
    else:
        norm = None

    # ── Layout ────────────────────────────────────────────────────────────────
    _ncols = ncols if ncols is not None else min(n_t, 4)
    nrows  = math.ceil(n_t / _ncols)
    if figsize is None:
        cb_extra = 0.8 if add_colorbar else 0.0
        figsize  = (4.5 * _ncols + cb_extra, 4.0 * nrows)

    fig, axes = plt.subplots(
        nrows, _ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    # ── Tracer origin ─────────────────────────────────────────────────────────
    if tracer_origin is None:
        tx0, ty0 = centre
    else:
        tx0, ty0 = tracer_origin

    t0 = float(t[0])

    # ── Draw each time slice ──────────────────────────────────────────────────
    pc_last = None

    for k in range(n_t):
        ax  = axes_flat[k]
        snp = field[k]     # (n_h, n_w)

        if shared_scale:
            pc = ax.pcolormesh(S1, S2, snp, cmap=cmap, norm=norm,
                               shading="auto")
        else:
            abs_max_k = max(abs(snp.min()), abs(snp.max())) or 1e-6
            pc = ax.pcolormesh(S1, S2, snp, cmap=cmap,
                               vmin=-abs_max_k, vmax=abs_max_k,
                               shading="auto")
            if add_colorbar and not shared_scale:
                fig.colorbar(pc, ax=ax, pad=0.02, shrink=0.9)

        ax.set_aspect("equal")
        ax.set_xlim(lo1, hi1)
        ax.set_ylim(lo2, hi2)
        ax.set_xlabel(r"$s_1$", fontsize=10)
        ax.set_ylabel(r"$s_2$", fontsize=10)

        # Per-subplot title
        if subplot_titles is not None:
            ax.set_title(subplot_titles[k], fontsize=11)
        else:
            ax.set_title(f"$t = {t[k]:.3g}$", fontsize=11)

        # ── Tracer: reference parcel drifting with advection ─────────────────
        if show_tracer:
            dt   = float(t[k]) - t0
            tx   = tx0 + lam_vec[0] * dt
            ty   = ty0 + lam_vec[1] * dt
            ax.scatter(tx, ty, s=tracer_size, color=tracer_color,
                       edgecolors="k", linewidths=0.8, zorder=5)
            # Faint trail: show tracer positions for all previous time steps
            for kk in range(k):
                dt_kk = float(t[kk]) - t0
                ax.scatter(
                    tx0 + lam_vec[0] * dt_kk,
                    ty0 + lam_vec[1] * dt_kk,
                    s=tracer_size * 0.3, color=tracer_color,
                    edgecolors="k", linewidths=0.5, alpha=0.35, zorder=4,
                )

        # ── Velocity arrow ────────────────────────────────────────────────────
        if show_velocity and lam0 > 0:
            # Arrow placed in the lower-right corner of the axes
            arrow_len = velocity_scale * domain_w
            ax_x      = lo1 + 0.82 * domain_w
            ax_y      = lo2 + 0.12 * domain_h
            dx        = np.cos(theta0) * arrow_len
            dy        = np.sin(theta0) * arrow_len
            ax.annotate(
                "", xy=(ax_x + dx, ax_y + dy), xytext=(ax_x, ax_y),
                arrowprops=dict(
                    arrowstyle="-|>", color="white",
                    lw=2.0, mutation_scale=14,
                ),
                zorder=6,
            )
            ax.annotate(
                "", xy=(ax_x + dx, ax_y + dy), xytext=(ax_x, ax_y),
                arrowprops=dict(
                    arrowstyle="-|>", color="k",
                    lw=0.8, mutation_scale=12,
                ),
                zorder=7,
            )

        pc_last = pc

    # ── Hide unused axes ──────────────────────────────────────────────────────
    for ax in axes_flat[n_t:]:
        ax.axis("off")

    # ── Shared colorbar ───────────────────────────────────────────────────────
    if add_colorbar and shared_scale and pc_last is not None:
        cb = fig.colorbar(pc_last, ax=axes_flat[:n_t], shrink=0.8,
                          pad=0.02, aspect=30)
        cb.set_label(r"$y(\mathbf{s}, t)$", fontsize=11)

    # ── Figure-level title ────────────────────────────────────────────────────
    if title is None:
        deg    = float(np.degrees(theta0))
        title  = (
            f"{result.model}"
            r"  |  $\phi$=" + f"{result.params['phi']:.2g}"
            r",  $\lambda_0$=" + f"{lam0:.2g}"
            f",  dir={deg:.0f}°"
            r",  $\sigma^2$=" + f"{result.sigma2:.2g}"
        )
    fig.suptitle(title, fontsize=12, y=1.01)

    # ── Legend for tracer ─────────────────────────────────────────────────────
    if show_tracer and lam0 > 0:
        patch = mpatches.Patch(
            facecolor=tracer_color, edgecolor="k",
            linewidth=0.8, label="Advection tracer",
        )
        fig.legend(handles=[patch], loc="lower center",
                   ncol=1, fontsize=9, framealpha=0.7,
                   bbox_to_anchor=(0.5, -0.03))

    return fig, axes_flat


__all__ = ["plot_st_realizations"]
