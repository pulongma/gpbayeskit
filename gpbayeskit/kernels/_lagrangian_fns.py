"""
gpbayeskit.kernels._lagrangian
==================================
The four Lagrangian / frozen-field spatio-temporal covariance functions
and their vectorised matrix wrappers.

All covariance functions share the signature:

    C(h, u, *, <model params>) -> float

where
    h  : spatial lag vector ∈ ℝ^d
    u  : temporal lag (scalar)

The Lagrangian distance used by the full Lagrangian models is:

    h_u = sqrt( (h − u·λ)ᵀ (u²Λ + I)⁻¹ (h − u·λ) )

The frozen-field models use the simpler Euclidean advection-corrected lag:

    |h − u·λ|

Model overview
--------------
frozen_matern     σ² M(|h − uλ| / φ; ν)
frozen_ch         σ² CH(|h − uλ|; ν, α, β)
lagrangian_matern σ² |I + ρ²Λ/2|^{-½} M(h_u / φ; ν)
lagrangian_ch     σ² |I + ρ²Λ/2|^{-½} CH(h_u; ν, α, β)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import det, inv

from gpbayeskit.kernels._functions import matern_kernel, ch_kernel

__all__ = [
    "frozen_matern",
    "frozen_ch",
    "lagrangian_matern",
    "lagrangian_ch",
    "frozen_matern_matrix",
    "frozen_ch_matrix",
    "lagrangian_matern_matrix",
    "lagrangian_ch_matrix",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _det_prefactor(Lambda: np.ndarray, rho: float, d: int) -> float:
    """
    Compute the determinant prefactor  |I_d + ρ²Λ/2|^{-1/2}.

    Parameters
    ----------
    Lambda : ndarray, shape (d, d)
    rho    : temporal range parameter ρ
    d      : spatial dimension

    Returns
    -------
    float
    """
    M = np.eye(d) + (rho ** 2 / 2.0) * Lambda
    return float(det(M) ** (-0.5))


def _lagrangian_distance(
    h: np.ndarray,
    u: float,
    lam_vec: np.ndarray,
    Lambda: np.ndarray,
) -> float:
    """
    Compute the Lagrangian distance:

        h_u = sqrt( (h − u·λ)ᵀ (u²Λ + I)⁻¹ (h − u·λ) )

    Parameters
    ----------
    h       : spatial lag vector, shape (d,)
    u       : temporal lag (scalar)
    lam_vec : advection vector λ, shape (d,)
    Lambda  : anisotropy matrix Λ, shape (d, d)

    Returns
    -------
    float  h_u ≥ 0
    """
    d = len(h)
    diff  = h - u * lam_vec
    A_inv = inv(u ** 2 * Lambda + np.eye(d))
    quad  = float(diff @ A_inv @ diff)
    return np.sqrt(max(quad, 0.0))   # guard against tiny floating-point negatives


# ── Frozen-field models ───────────────────────────────────────────────────────

def frozen_matern(
    h: np.ndarray,
    u: float,
    *,
    sigma2: float,
    nu: float,
    phi: float,
    lam_vec: np.ndarray,
) -> float:
    """
    Frozen-field Matérn covariance:

        C(h, u) = σ² · M( |h − u·λ| / φ ; ν )

    The spatial field is advected rigidly by the constant velocity λ;
    no temporal deformation occurs.

    Parameters
    ----------
    h       : spatial lag vector ∈ ℝ^d
    u       : temporal lag (scalar)
    sigma2  : marginal variance σ² > 0
    nu      : Matérn smoothness ν > 0
    phi     : spatial range φ > 0
    lam_vec : advection vector λ ∈ ℝ^d  (build with `lam_vec_from_polar`)

    Returns
    -------
    float
        Covariance value C(h, u).
    """
    r = np.linalg.norm(h - u * lam_vec)
    return sigma2 * matern_kernel(r / phi, nu)


def frozen_ch(
    h: np.ndarray,
    u: float,
    *,
    sigma2: float,
    nu: float,
    alpha: float,
    beta: float,
    lam_vec: np.ndarray,
) -> float:
    """
    Frozen-field CH covariance:

        C(h, u) = σ² · CH( |h − u·λ| ; ν, α, β )

    Parameters
    ----------
    h       : spatial lag vector ∈ ℝ^d
    u       : temporal lag (scalar)
    sigma2  : marginal variance σ² > 0
    nu      : smoothness parameter ν > 0
    alpha   : tail-decay parameter α > 0
    beta    : spatial scale parameter β > 0
    lam_vec : advection vector λ ∈ ℝ^d

    Returns
    -------
    float
        Covariance value C(h, u).
    """
    r = np.linalg.norm(h - u * lam_vec)
    return sigma2 * ch_kernel(r, nu, alpha, beta)


# ── Full Lagrangian models ────────────────────────────────────────────────────

def lagrangian_matern(
    h: np.ndarray,
    u: float,
    *,
    sigma2: float,
    rho: float,
    nu: float,
    phi: float,
    lam_vec: np.ndarray,
    Lambda: np.ndarray,
) -> float:
    """
    Lagrangian Matérn covariance:

        C(h, u) = σ² |I + ρ²Λ/2|^{-½} · M( h_u / φ ; ν )

    where the Lagrangian distance is:

        h_u = sqrt( (h − u·λ)ᵀ (u²Λ + I)⁻¹ (h − u·λ) )

    Parameters
    ----------
    h       : spatial lag vector ∈ ℝ^d
    u       : temporal lag (scalar)
    sigma2  : marginal variance σ² > 0
    rho     : temporal range parameter ρ > 0
    nu      : Matérn smoothness ν > 0
    phi     : spatial range φ > 0
    lam_vec : advection vector λ ∈ ℝ^d  (build with `lam_vec_from_polar`)
    Lambda  : d×d anisotropy matrix Λ    (build with `build_Lambda`)

    Returns
    -------
    float
        Covariance value C(h, u).
    """
    d          = len(h)
    hu         = _lagrangian_distance(h, u, lam_vec, Lambda)
    prefactor  = _det_prefactor(Lambda, rho, d)
    return sigma2 * prefactor * matern_kernel(hu / phi, nu)


def lagrangian_ch(
    h: np.ndarray,
    u: float,
    *,
    sigma2: float,
    rho: float,
    nu: float,
    alpha: float,
    beta: float,
    lam_vec: np.ndarray,
    Lambda: np.ndarray,
) -> float:
    """
    Lagrangian CH covariance:

        C(h, u) = σ² |I + ρ²Λ/2|^{-½} · CH( h_u ; ν, α, β )

    where the Lagrangian distance h_u is defined as in `lagrangian_matern`.

    Parameters
    ----------
    h       : spatial lag vector ∈ ℝ^d
    u       : temporal lag (scalar)
    sigma2  : marginal variance σ² > 0
    rho     : temporal range parameter ρ > 0
    nu      : smoothness parameter ν > 0
    alpha   : tail-decay parameter α > 0
    beta    : spatial scale parameter β > 0
    lam_vec : advection vector λ ∈ ℝ^d
    Lambda  : d×d anisotropy matrix Λ

    Returns
    -------
    float
        Covariance value C(h, u).
    """
    d         = len(h)
    hu        = _lagrangian_distance(h, u, lam_vec, Lambda)
    prefactor = _det_prefactor(Lambda, rho, d)
    return sigma2 * prefactor * ch_kernel(hu, nu, alpha, beta)


# ── Vectorised matrix wrappers ────────────────────────────────────────────────

def _make_matrix(cov_fn, h_list, u_list, kwargs):
    """Generic (N, d) × (M,) → (N, M) evaluator."""
    h_list = np.asarray(h_list)
    u_list = np.asarray(u_list)
    N, M   = len(h_list), len(u_list)
    C      = np.empty((N, M))
    for i, h in enumerate(h_list):
        for j, u in enumerate(u_list):
            C[i, j] = cov_fn(h, float(u), **kwargs)
    return C


def frozen_matern_matrix(
    h_list: np.ndarray,
    u_list: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Evaluate :func:`frozen_matern` over arrays of spatial and temporal lags.

    Parameters
    ----------
    h_list : ndarray, shape (N, d)
        Spatial lag vectors.
    u_list : ndarray, shape (M,)
        Temporal lags.
    **kwargs
        Forwarded to :func:`frozen_matern`.

    Returns
    -------
    C : ndarray, shape (N, M)
    """
    return _make_matrix(frozen_matern, h_list, u_list, kwargs)


def frozen_ch_matrix(
    h_list: np.ndarray,
    u_list: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Evaluate :func:`frozen_ch` over arrays of spatial and temporal lags.

    Parameters
    ----------
    h_list : ndarray, shape (N, d)
    u_list : ndarray, shape (M,)
    **kwargs
        Forwarded to :func:`frozen_ch`.

    Returns
    -------
    C : ndarray, shape (N, M)
    """
    return _make_matrix(frozen_ch, h_list, u_list, kwargs)


def lagrangian_matern_matrix(
    h_list: np.ndarray,
    u_list: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Evaluate :func:`lagrangian_matern` over arrays of spatial and temporal lags.

    Parameters
    ----------
    h_list : ndarray, shape (N, d)
    u_list : ndarray, shape (M,)
    **kwargs
        Forwarded to :func:`lagrangian_matern`.

    Returns
    -------
    C : ndarray, shape (N, M)
    """
    return _make_matrix(lagrangian_matern, h_list, u_list, kwargs)


def lagrangian_ch_matrix(
    h_list: np.ndarray,
    u_list: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Evaluate :func:`lagrangian_ch` over arrays of spatial and temporal lags.

    Parameters
    ----------
    h_list : ndarray, shape (N, d)
    u_list : ndarray, shape (M,)
    **kwargs
        Forwarded to :func:`lagrangian_ch`.

    Returns
    -------
    C : ndarray, shape (N, M)
    """
    return _make_matrix(lagrangian_ch, h_list, u_list, kwargs)