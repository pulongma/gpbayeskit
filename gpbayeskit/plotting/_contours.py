"""
gpbayeskit.plotting._contours
------------------------------------
Space-time covariance contour plots for Lagrangian and frozen-field GP models.

Functions
---------
plot_cov_st_1d   Filled contour of C(h, u) for a 1-D spatial domain.
                 x-axis = scalar spatial lag h, y-axis = temporal lag u.
                 Advection appears as a tilted ridge.

plot_cov_st_2d   Panel of spatial covariance contours C(h₁, h₂; u) at
                 specified temporal lags.  One subplot per u value.
                 A white arrow shows the advection displacement u·λ.

Both functions accept either a fitted ``SpatioTemporalGP`` (reads params_
and st_model automatically) or a plain parameter dict with an explicit
``model`` argument.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..models.spatiotemporal import SpatioTemporalGP
from ..kernels._lagrangian import LagrangianKernelBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

_ST_DEFAULTS = dict(
    phi=1.0, nu=1.5, tail=0.5,
    lam0=0.0, theta0=0.0,
    rho=1.0, lambda1=1.0, lambda2=1.0, theta_Lambda=0.0,
)


def _st_params_from(model_or_params) -> dict:
    """
    Return a complete parameter dict from either a fitted SpatioTemporalGP
    or a plain dict.  Missing keys are filled with sensible defaults.
    """
    if isinstance(model_or_params, SpatioTemporalGP):
        if model_or_params.params_ is None:
            raise RuntimeError("SpatioTemporalGP must be fitted before plotting.")
        p = dict(_ST_DEFAULTS)
        p.update(model_or_params.params_)
        return p
    if isinstance(model_or_params, dict):
        p = dict(_ST_DEFAULTS)
        p.update(model_or_params)
        return p
    raise TypeError(
        "model_or_params must be a fitted SpatioTemporalGP or a parameter dict."
    )


def _kb_kwargs(params: dict) -> dict:
    """Build the keyword dict expected by LagrangianKernelBuilder.__call__."""
    return dict(
        phi=float(params["phi"]),
        nu=float(params.get("nu", 1.5)),
        tail=float(params.get("tail", 0.5)),
        lam0=float(params.get("lam0", 0.0)),
        theta0=float(params.get("theta0", 0.0)),
        rho=float(params.get("rho", 1.0)),
        lambda1=float(params.get("lambda1", 1.0)),
        lambda2=float(params.get("lambda2", 1.0)),
        theta_Lambda=float(params.get("theta_Lambda", 0.0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# plot_cov_st_1d — C(h, u) for a 1-D spatial domain
# ─────────────────────────────────────────────────────────────────────────────

def plot_cov_st_1d(
    model_or_params,
    h_range: tuple[float, float],
    u_range: tuple[float, float],
    model: str = "lagrangian_matern",
    n_h: int = 200,
    n_u: int = 200,
    levels: int | Sequence[float] = 12,
    cmap: str = "RdYlBu_r",
    add_colorbar: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Filled contour plot of the spatio-temporal covariance C(h, u) for a
    **1-D spatial domain**.

    The x-axis is the scalar spatial lag h, the y-axis is the temporal lag
    u, and colour encodes the covariance value.  Advection appears as a
    tilted ridge running across the plot.

    Parameters
    ----------
    model_or_params : SpatioTemporalGP or dict
        Fitted GP model (uses model.params_ and model.st_model) or a
        parameter dict.  When a dict is passed, ``model`` selects the kernel.
    h_range : (lo, hi)
        Range of scalar spatial lags.  Symmetric ranges like (-3, 3) show
        the advection tilt clearly.
    u_range : (lo, hi)
        Range of temporal lags.  Use (0, T) for non-negative time only.
    model : str
        One of "frozen_matern", "frozen_ch", "lagrangian_matern",
        "lagrangian_ch".  Ignored when model_or_params is a SpatioTemporalGP.
    n_h, n_u : int
        Grid resolution along the spatial and temporal axes.
    levels : int or sequence of float
        Number of contour levels or explicit level values.
    cmap : str
        Matplotlib colormap name.
    add_colorbar : bool
        Whether to add a colorbar.
    ax : plt.Axes or None
        Axes to draw on; creates a new figure if None.
    title : str or None
        Axes title.  Auto-generated from key parameter values if None.

    Returns
    -------
    fig : matplotlib Figure
    ax  : matplotlib Axes
    """
    if isinstance(model_or_params, SpatioTemporalGP):
        st_model = model_or_params.st_model
    else:
        st_model = model

    params = _st_params_from(model_or_params)
    kb     = LagrangianKernelBuilder(model=st_model)

    h_grid = np.linspace(h_range[0], h_range[1], n_h)
    u_grid = np.linspace(u_range[0], u_range[1], n_u)
    H_mesh, U_mesh = np.meshgrid(h_grid, u_grid)        # (n_u, n_h)

    # The builder always works in 2-D space (lam_vec and Lambda are 2×2).
    # Embed the 1-D spatial lag along h₁ with h₂ = 0.
    h1_flat = H_mesh.ravel()
    S1 = np.column_stack([h1_flat, np.zeros_like(h1_flat)])  # (n_h*n_u, 2)
    T1 = U_mesh.ravel()
    S0 = np.zeros((1, 2))
    T0 = np.zeros(1)

    C_flat = kb(S0, S1, T0, T1, **_kb_kwargs(params))   # (1, n_h*n_u)
    C = C_flat.reshape(n_u, n_h)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    cf = ax.contourf(H_mesh, U_mesh, C, levels=levels, cmap=cmap)
    ax.contour(H_mesh, U_mesh, C, levels=cf.levels,
               colors="k", linewidths=0.4, alpha=0.4)

    ax.set_xlabel(r"Spatial lag $h$", fontsize=12)
    ax.set_ylabel(r"Temporal lag $u$", fontsize=12)

    if add_colorbar:
        cb = fig.colorbar(cf, ax=ax, pad=0.02)
        cb.set_label("Covariance", fontsize=10)

    if title is None:
        lam0 = float(params.get("lam0", 0.0))
        title = (
            f"{st_model}   "
            r"$\phi$=" + f"{params['phi']:.2g},"
            r"  $\lambda_0$=" + f"{lam0:.2g}"
        )
    ax.set_title(title, fontsize=11)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# plot_cov_st_2d — C(h₁, h₂; u) panel for a 2-D spatial domain
# ─────────────────────────────────────────────────────────────────────────────

def plot_cov_st_2d(
    model_or_params,
    h_range: tuple[float, float],
    u_values: Sequence[float],
    model: str = "lagrangian_matern",
    n_h: int = 80,
    levels: int | Sequence[float] = 10,
    cmap: str = "RdYlBu_r",
    add_colorbar: bool = True,
    shared_scale: bool = True,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    titles: Sequence[str] | None = None,
    title: str | None = None,
    suptitle: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Panel of spatial covariance contours C(h₁, h₂; u) at specified temporal
    lags for a **2-D spatial domain**.

    Each subplot shows the full spatial covariance field centred at the
    origin for one value of the temporal lag u.  A white arrow marks the
    advection displacement u·λ.  The panel reveals how anisotropy, tilt,
    and variance evolve over time.

    Parameters
    ----------
    model_or_params : SpatioTemporalGP or dict
        Fitted GP model or parameter dict.  When a dict is passed, ``model``
        selects the kernel type.
    h_range : (lo, hi)
        Symmetric range for both spatial lag dimensions, e.g. (-2, 2).
    u_values : sequence of float
        Temporal lags at which to show the spatial covariance slice.
        Example: [0, 0.5, 1.0, 2.0]
    model : str
        Kernel model name.  Ignored when model_or_params is a SpatioTemporalGP.
    n_h : int
        Grid resolution for each spatial dimension.
    levels : int or sequence of float
        Contour levels (shared across subplots when shared_scale=True).
    cmap : str
        Matplotlib colormap name.
    add_colorbar : bool
        Whether to add a colorbar (shared when shared_scale=True,
        per-subplot otherwise).
    shared_scale : bool
        If True, all subplots use the same colour scale (min/max over all u).
        If False, each subplot autoscales independently.
    ncols : int or None
        Number of subplot columns.  Defaults to min(len(u_values), 4).
    figsize : (width, height) or None
        Figure size; auto-computed if None.
    titles : sequence of str or None
        Per-subplot titles (one per u value).  Defaults to "$u = {value}$".
    title : str or None
        Figure-level title.  Auto-generated from key parameters if None.
        Alias for ``suptitle``; if both are given, ``title`` takes precedence.
    suptitle : str or None
        Alias for ``title`` retained for backward compatibility.

    Returns
    -------
    fig  : matplotlib Figure
    axes : (nrows * ncols,) flattened ndarray of Axes
    """
    if isinstance(model_or_params, SpatioTemporalGP):
        st_model = model_or_params.st_model
    else:
        st_model = model

    params   = _st_params_from(model_or_params)
    kb       = LagrangianKernelBuilder(model=st_model)
    u_values = list(u_values)
    n_u      = len(u_values)
    kw       = _kb_kwargs(params)

    # Spatial grid — same for every u
    h_lin  = np.linspace(h_range[0], h_range[1], n_h)
    H1, H2 = np.meshgrid(h_lin, h_lin)                  # (n_h, n_h)
    S1     = np.column_stack([H1.ravel(), H2.ravel()])   # (n_h², 2)
    S0     = np.zeros((1, 2))

    # Pre-compute all covariance grids
    grids = []
    for u in u_values:
        T0 = np.zeros(1)
        T1 = np.full(len(S1), float(u))
        C_flat = kb(S0, S1, T0, T1, **kw)               # (1, n_h²)
        grids.append(C_flat.reshape(n_h, n_h))

    # Colour scale
    if shared_scale:
        vmin = min(g.min() for g in grids)
        vmax = max(g.max() for g in grids)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        vmin = vmax = norm = None

    # Layout
    _ncols  = ncols if ncols is not None else min(n_u, 4)
    nrows   = math.ceil(n_u / _ncols)
    if figsize is None:
        figsize = (4.5 * _ncols + (1.0 if add_colorbar else 0), 4.0 * nrows)

    fig, axes = plt.subplots(
        nrows, _ncols, figsize=figsize,
        squeeze=False, constrained_layout=True,
    )
    axes_flat = axes.ravel()

    lam0   = float(params.get("lam0", 0.0))
    theta0 = float(params.get("theta0", 0.0))
    cf_last = None

    for i, (u, grid) in enumerate(zip(u_values, grids)):
        ax = axes_flat[i]

        cf_kw = dict(levels=levels, cmap=cmap)
        if shared_scale:
            cf = ax.contourf(H1, H2, grid, norm=norm, **cf_kw)
        else:
            cf = ax.contourf(H1, H2, grid, **cf_kw)
            if add_colorbar:
                fig.colorbar(cf, ax=ax, pad=0.02)

        ax.contour(H1, H2, grid, levels=cf.levels,
                   colors="k", linewidths=0.4, alpha=0.4)

        # Advection arrow: tip at u·λ
        if lam0 > 0 and u != 0:
            dx = u * lam0 * np.cos(theta0)
            dy = u * lam0 * np.sin(theta0)
            ax.annotate(
                "", xy=(dx, dy), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.8),
            )

        ax.set_aspect("equal")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_xlabel(r"$h_1$", fontsize=11)
        ax.set_ylabel(r"$h_2$", fontsize=11)
        ax.set_title(titles[i] if titles is not None else f"$u = {u:.3g}$",
                     fontsize=11)
        cf_last = cf

    # Hide unused axes
    for ax in axes_flat[n_u:]:
        ax.axis("off")

    # Shared colorbar
    if add_colorbar and shared_scale and cf_last is not None:
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes_flat[:n_u], shrink=0.7, pad=0.02,
        )
        cb.set_label("Covariance", fontsize=10)

    # title takes precedence over suptitle for consistency with plot_cov_st_1d
    if title is not None:
        suptitle = title

    if suptitle is None:
        suptitle = (
            f"{st_model}   "
            r"$\phi$=" + f"{params['phi']:.2g},"
            r"  $\lambda_0$=" + f"{lam0:.2g},"
            r"  $\rho$=" + f"{float(params.get('rho', 1.0)):.2g}"
        )
    fig.suptitle(suptitle, fontsize=12, y=1.01)

    return fig, axes_flat


__all__ = ["plot_cov_st_1d", "plot_cov_st_2d"]