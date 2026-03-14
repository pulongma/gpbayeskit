"""
gpbayeskit.plotting._likelihood
--------------------------------------
Integrated log-likelihood profile plots for spatial GP models.

Functions
---------
plot_loglik        1-D profile of the integrated log-likelihood vs. one parameter.
plot_loglik_panel  Grid of profile plots from a list of specs.
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt

from ..models.spatial import SpatialGP


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_SYMBOL: dict[str, str] = {
    "phi":    r"$\phi$",
    "nu":     r"$\nu$",
    "nugget": r"$\tau^2$",
    "tail":   r"$\alpha$",
}

_PARAMS_BY_FAMILY: dict[str, list[str]] = {
    "matern":   ["phi", "nu", "nugget"],
    "exp":      ["phi", "nugget"],
    "matern32": ["phi", "nugget"],
    "matern52": ["phi", "nugget"],
    "gauss":    ["phi", "nugget"],
    "ch":       ["phi", "nu", "nugget", "tail"],
}


def _relevant_params(kernel_type: str) -> list[str]:
    return _PARAMS_BY_FAMILY.get(kernel_type.lower(), ["phi", "nu", "nugget"])


def _format_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.2e}" if (abs(v) < 1e-3 or abs(v) > 1e3) else f"{v:.3g}"
    return str(v)


def _auto_title(param: str, fixed_params: dict, kernel_type: str) -> str:
    """List fixed parameter values in the axes title, excluding the varied param."""
    relevant = _relevant_params(kernel_type)
    fixed = {k: v for k, v in fixed_params.items()
             if k in relevant and k != param}
    if not fixed:
        return ""
    return ", ".join(
        f"{_PARAM_SYMBOL.get(k, k)}={_format_value(v)}"
        for k, v in fixed.items()
    )


def _eval_loglik(
    X, y, phi, nu, nugget, tail,
    kernel_type, kernel_form, H,
) -> float:
    """Evaluate integrated log marginal likelihood via a temporary SpatialGP."""
    gp = SpatialGP(X, y, kernel_type=kernel_type, kernel_form=kernel_form, H=H)
    return gp.log_marginal_likelihood(phi=phi, nu=nu, nugget=nugget, tail=tail)


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_loglik(
    X: np.ndarray,
    y: np.ndarray,
    param: str,
    param_range: tuple[float, float],
    fixed_params: dict,
    kernel_type: str = "matern",
    kernel_form: str = "isotropic",
    H: np.ndarray | None = None,
    n_points: int = 100,
    log_x: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[np.ndarray, np.ndarray, plt.Axes]:
    """
    Plot the integrated log marginal likelihood profile for one parameter.

    Parameters
    ----------
    X, y         : training data
    param        : parameter to vary — one of {"phi", "nu", "nugget", "tail"}
    param_range  : (lo, hi) tuple for the grid
    fixed_params : dict of remaining parameter values (held fixed)
    kernel_type  : kernel family
    kernel_form  : anisotropy form
    H            : mean-basis matrix (None → intercept only)
    n_points     : number of grid points
    log_x        : if True, use a log-spaced grid and log x-axis
    ax           : existing Axes to draw on (creates a new figure if None)

    Returns
    -------
    grid   : (n_points,)  parameter grid
    values : (n_points,)  log-likelihood values
    ax     : Axes object
    """
    valid_params = {"phi", "nu", "nugget", "tail"}
    if param not in valid_params:
        raise ValueError(f"param must be one of {valid_params}, got '{param}'.")
    if param == "tail" and kernel_type.lower() != "ch":
        raise ValueError("'tail' is only relevant for kernel_type='ch'.")

    lo, hi = param_range
    grid = np.geomspace(lo, hi, n_points) if log_x else np.linspace(lo, hi, n_points)

    values = np.array([
        _eval_loglik(
            X=X, y=y,
            phi=fixed_params.get("phi", 0.5)      if param != "phi"    else g,
            nu=fixed_params.get("nu", 2.5)         if param != "nu"     else g,
            nugget=fixed_params.get("nugget", 0.0) if param != "nugget" else g,
            tail=fixed_params.get("tail", 0.5)     if param != "tail"   else g,
            kernel_type=kernel_type,
            kernel_form=kernel_form,
            H=H,
        )
        for g in grid
    ])

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(grid, values, linewidth=2)
    ax.set_xlabel(_PARAM_SYMBOL.get(param, param), fontsize=12)
    ax.set_ylabel("Integrated log-likelihood", fontsize=11)
    ax.set_title(_auto_title(param, fixed_params, kernel_type))
    if log_x:
        ax.set_xscale("log")

    return grid, values, ax


def plot_loglik_panel(
    X: np.ndarray,
    y: np.ndarray,
    specs: list[dict],
    kernel_type: str = "matern",
    kernel_form: str = "isotropic",
    H: np.ndarray | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a panel of log-likelihood profiles.

    Parameters
    ----------
    X, y        : training data
    specs       : list of dicts, each with keys:
                    param        (str)   — parameter to vary
                    param_range  (tuple) — (lo, hi)
                    fixed_params (dict)  — remaining parameter values
                    n_points     (int, optional, default 100)
                    log_x        (bool, optional, default False)
                    title        (str, optional) — override auto title
    kernel_type : kernel family (applied to all panels)
    kernel_form : anisotropy form (applied to all panels)
    H           : mean-basis matrix
    ncols       : number of columns in the panel grid
    figsize     : figure size; auto-computed if None

    Returns
    -------
    fig  : matplotlib Figure
    axes : flattened array of Axes
    """
    nplots = len(specs)
    nrows  = math.ceil(nplots / ncols)
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, spec in zip(axes_flat, specs):
        plot_loglik(
            X=X, y=y,
            param=spec["param"],
            param_range=spec["param_range"],
            fixed_params=spec["fixed_params"],
            kernel_type=kernel_type,
            kernel_form=kernel_form,
            H=H,
            n_points=spec.get("n_points", 100),
            log_x=spec.get("log_x", False),
            ax=ax,
        )
        if "title" in spec:
            ax.set_title(spec["title"])

    for ax in axes_flat[nplots:]:
        ax.axis("off")

    plt.tight_layout()
    return fig, axes_flat


__all__ = ["plot_loglik", "plot_loglik_panel"]