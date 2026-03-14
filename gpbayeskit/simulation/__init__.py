"""
simulate.py
-----------
Simulate realisations from a spatial Gaussian process.

    y ~ N(H beta,  sigma2 * (R(phi, nu) + nugget * I))

where R is a correlation matrix built from a ``KernelBuilder``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky

from gpbayeskit.kernels import KernelBuilder
from gpbayeskit.utils import validate_H


# ─────────────────────────────────────────────────────────────────────────────
# Return container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Container for a single GP simulation draw.

    Attributes
    ----------
    y       : (n, 1)  observed draw  (with nugget noise)
    mean    : (n, 1)  mean vector H @ beta
    cov     : (n, n)  full covariance matrix sigma2 * (R + nugget * I)
    latent  : (n, 1) or None
              Latent (noise-free) draw from  N(H beta, sigma2 * R).
              Only set when ``simulate(..., return_latent=True)``.
    """
    y:      np.ndarray
    mean:   np.ndarray
    cov:    np.ndarray
    latent: np.ndarray | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def simulate(
    X: np.ndarray,
    kernel_type: str = "matern",
    kernel_form: str = "isotropic",
    phi: float | np.ndarray = 0.2,
    nu: float = 2.5,
    nugget: float = 1e-4,
    sigma2: float = 1.0,
    tail: float = 0.5,
    beta: np.ndarray | None = None,
    H: np.ndarray | None = None,
    seed: int | None = None,
    return_latent: bool = False,
) -> SimulationResult:
    """
    Simulate one realisation from a spatial GP.

    Parameters
    ----------
    X            : (n, d)  Input locations.
    kernel_type  : str     Kernel family — one of
                           {"matern", "exp", "matern32", "matern52", "ch", "gauss"}.
    kernel_form  : str     Anisotropy form — one of
                           {"isotropic", "ard", "tensor"}.
    phi          : float or (d,) ndarray
                   Range / scale parameter(s).  Must match *kernel_form*:
                   scalar for isotropic, d-vector for ard / tensor.
    nu           : float   Smoothness parameter.
    nugget       : float   Nugget ratio τ² ≥ 0.  Added as
                           sigma2 * (R + nugget * I).
    sigma2       : float   Marginal variance scale (> 0).
    tail         : float   Tail / shape parameter (CH kernel only).
    beta         : (p,) ndarray or None
                   Mean coefficients.  None → zero mean.
    H            : (n, p) ndarray or None
                   Mean-basis matrix.  None → intercept-only column.
    seed         : int or None
                   Random seed for reproducibility.
    return_latent: bool
                   If True, also simulate a noise-free (no-nugget) draw
                   stored in ``SimulationResult.latent``.

    Returns
    -------
    SimulationResult
        ``.y``      — observed draw  (n, 1)
        ``.mean``   — mean vector    (n, 1)
        ``.cov``    — covariance     (n, n)
        ``.latent`` — latent draw    (n, 1) or None

    Examples
    --------
    >>> import numpy as np
    >>> from gpbayeskit.simulation import simulate
    >>> rng  = np.random.default_rng(0)
    >>> X    = rng.uniform(0, 1, (50, 2))
    >>> sim  = simulate(X, kernel_type="matern52", phi=0.3, sigma2=2.0, seed=42)
    >>> sim.y.shape
    (50, 1)
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n, d).")

    n, d = X.shape

    if nugget < 0:
        raise ValueError(f"nugget must be non-negative, got {nugget}.")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}.")

    H = validate_H(H, n)
    p = H.shape[1]

    if beta is None:
        beta = np.zeros((p, 1), dtype=float)
    else:
        beta = np.asarray(beta, dtype=float).reshape(-1, 1)
        if beta.shape[0] != p:
            raise ValueError(
                f"beta must have length {p} (number of columns in H), "
                f"got {beta.shape[0]}."
            )

    mean_vec = H @ beta   # (n, 1)

    kb = KernelBuilder(kernel_type=kernel_type, kernel_form=kernel_form)
    R  = kb(X, X, dim=d, phi=phi, nu=nu, tail=tail)
    R  = _symmetrise(R)

    K = sigma2 * (R + nugget * np.eye(n))
    K = _symmetrise(K)

    L = _safe_cholesky(K, label="observed covariance")

    z = rng.standard_normal((n, 1))
    y = mean_vec + L @ z   # (n, 1)

    latent = None
    if return_latent:
        K_lat = _symmetrise(sigma2 * R)
        L_lat = _safe_cholesky(K_lat, label="latent covariance", fallback_jitter=1e-12)
        z_lat = rng.standard_normal((n, 1))
        latent = mean_vec + L_lat @ z_lat

    return SimulationResult(y=y, mean=mean_vec, cov=K, latent=latent)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _symmetrise(M: np.ndarray) -> np.ndarray:
    """Force exact symmetry to guard against floating-point asymmetry."""
    return 0.5 * (M + M.T)


def _safe_cholesky(
    M: np.ndarray,
    label: str = "matrix",
    fallback_jitter: float | None = None,
) -> np.ndarray:
    """
    Cholesky decomposition with an optional single-step jitter fallback.

    If *fallback_jitter* is given and the first attempt fails, a small
    diagonal term is added and the decomposition is retried once; a
    warning is issued so the caller is aware of the stabilisation.
    """
    try:
        return cholesky(M, lower=True)
    except np.linalg.LinAlgError:
        if fallback_jitter is not None:
            warnings.warn(
                f"{label} is numerically singular; "
                f"adding jitter {fallback_jitter:.2e} to diagonal.",
                RuntimeWarning,
                stacklevel=3,
            )
            try:
                return cholesky(M + fallback_jitter * np.eye(len(M)), lower=True)
            except np.linalg.LinAlgError:
                pass
        raise np.linalg.LinAlgError(
            f"{label} is not positive definite. "
            "Try a larger nugget or different kernel parameters."
        )

# ─── Spatio-temporal simulation ───────────────────────────────────────────────
from gpbayeskit.simulation._spatiotemporal import simulate_st, STSimulationResult  
