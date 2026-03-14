"""
gpbayeskit.simulation._spatiotemporal
======================================
Simulate realisations from a spatio-temporal GP on a regular 2-D spatial
grid × a sequence of time points.

The full joint covariance is built across all (space × time) locations and
a single multivariate normal draw is made — so the simulated field is jointly
consistent across space and time.

Covariance formulas
-------------------
Let h = s₁ − s₂ (spatial lag), u = t₁ − t₂ (temporal lag),
  λ   = λ₀ [cos θ₀, sin θ₀]  (advection vector),
  Λ   = U diag(λ₁, λ₂) Uᵀ   (deformation matrix),
  A(u) = (u/ρ)² Λ + I        (space-time deformation at lag u).

frozen_matern:
    C(h, u) = σ² M( |h − uλ| / φ ; ν )

frozen_ch:
    C(h, u) = σ² CH( |h − uλ| ; ν, α, φ )

lagrangian_matern:
    C(h, u) = σ² |A(u)|^{−½} M( h_u / φ ; ν )
    h_u     = sqrt( (h − uλ)ᵀ A(u)⁻¹ (h − uλ) )

lagrangian_ch:
    C(h, u) = σ² |A(u)|^{−½} CH( h_u ; ν, α, φ )

The Lagrangian formula uses the **u-dependent** prefactor |A(u)|^{-½}
(not the constant prefactor in the package's SpatioTemporalGP kernel).
This ensures C(h, 0) = σ² M(|h|/φ; ν), C(0, 0) = σ², and that the
full space-time covariance matrix is positive semi-definite (Stein 2005).

The ρ parameter controls how quickly spatial deformation grows with |u|:
larger ρ → slower deformation onset (approaches the frozen field as ρ → ∞).

Vectorisation
-------------
For each unique lag u = tᵢ − tⱼ the matrix A(u) is inverted once and all
n_s² spatial pairs are processed in a single NumPy pass.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky
from scipy.special import kv, gamma as _gamma, hyperu as _hyperu

from gpbayeskit.kernels._parametrisation import lam_vec_from_polar, build_Lambda


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised Matérn
# ─────────────────────────────────────────────────────────────────────────────

def _matern_vec(r: np.ndarray, nu: float) -> np.ndarray:
    r   = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    m   = r > 0.0
    rm  = r[m]
    out[m] = (2.0 ** (1.0 - nu) / _gamma(nu)) * (rm ** nu) * kv(nu, rm)
    return out




def _ch_vec(r: np.ndarray, nu: float, tail: float, phi: float) -> np.ndarray:
    """CH kernel CH(r; ν, α, β=φ) evaluated element-wise over an array of distances."""
    r   = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    m   = r > 0.0
    rm  = r[m]
    prefactor = _gamma(nu + tail) / _gamma(nu)
    out[m] = prefactor * _hyperu(tail, 1.0 - nu, (rm / phi) ** 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class STSimulationResult:
    """
    Container for a spatio-temporal GP simulation draw on a regular grid.

    Attributes
    ----------
    field   : (n_t, n_h, n_w)  Simulated snapshots.  ``field[k]`` is the
              2-D spatial field at time ``t[k]``.
    S1      : (n_h, n_w)  First spatial coordinate meshgrid.
    S2      : (n_h, n_w)  Second spatial coordinate meshgrid.
    t       : (n_t,)      Time points.
    params  : dict        Kernel parameters used.
    sigma2  : float       Marginal variance.
    nugget  : float       Nugget ratio.
    model   : str         Kernel model name.
    """
    field:  np.ndarray
    S1:     np.ndarray
    S2:     np.ndarray
    t:      np.ndarray
    params: dict
    sigma2: float
    nugget: float
    model:  str

    @property
    def n_t(self) -> int:
        return len(self.t)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.field.shape[1], self.field.shape[2]

    @property
    def lam_vec(self) -> np.ndarray:
        """Advection vector λ = λ₀ [cos θ₀, sin θ₀]."""
        return lam_vec_from_polar(
            float(self.params.get("lam0", 0.0)),
            float(self.params.get("theta0", 0.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Block builders
# ─────────────────────────────────────────────────────────────────────────────

def _frozen_block(S, u, phi, nu, lam_vec, use_ch=False, tail=0.5):
    h     = S[:, None, :] - S[None, :, :]          # (n_s, n_s, 2)
    h_adv = h - u * lam_vec
    dist  = np.sqrt(np.maximum((h_adv ** 2).sum(axis=-1), 0.0))
    r_flat = dist.ravel()
    if use_ch:
        return _ch_vec(r_flat, nu, tail, phi).reshape(len(S), len(S))
    return _matern_vec((r_flat / phi), nu).reshape(len(S), len(S))


def _lagrangian_block(S, u, phi, nu, lam_vec, Lambda, rho, use_ch=False, tail=0.5):
    u_sc      = u / rho if rho > 0 else 0.0
    A         = u_sc ** 2 * Lambda + np.eye(2)
    A_inv     = np.linalg.inv(A)
    prefactor = float(np.linalg.det(A) ** (-0.5))

    h         = S[:, None, :] - S[None, :, :]
    h_shifted = h - u * lam_vec
    Ah        = h_shifted @ A_inv
    hu        = np.sqrt(np.maximum((Ah * h_shifted).sum(axis=-1), 0.0))
    hu_flat   = hu.ravel()
    if use_ch:
        return prefactor * _ch_vec(hu_flat, nu, tail, phi).reshape(len(S), len(S))
    return prefactor * _matern_vec((hu_flat / phi), nu).reshape(len(S), len(S))


def _build_st_cov(S, t, sigma2, nugget, phi, nu, lam_vec,
                  is_lagrangian, Lambda, rho, use_ch=False, tail=0.5):
    n_s, n_t = len(S), len(t)
    K        = np.zeros((n_s * n_t, n_s * n_t))

    for i in range(n_t):
        for j in range(i, n_t):
            u = float(t[i] - t[j])
            if is_lagrangian:
                block = _lagrangian_block(S, u, phi, nu, lam_vec, Lambda, rho,
                                          use_ch=use_ch, tail=tail)
            else:
                block = _frozen_block(S, u, phi, nu, lam_vec,
                                     use_ch=use_ch, tail=tail)
            rs, cs = i * n_s, j * n_s
            re, ce = rs + n_s, cs + n_s
            K[rs:re, cs:ce] = block
            if i != j:
                K[cs:ce, rs:re] = block.T

    K = sigma2 * K
    np.fill_diagonal(K, np.diag(K) + sigma2 * nugget)
    return 0.5 * (K + K.T)


def _safe_chol(K, nugget):
    n, jitter = len(K), 0.0
    for _ in range(12):
        try:
            return cholesky(K + jitter * np.eye(n), lower=True)
        except np.linalg.LinAlgError:
            if jitter == 0.0:
                jitter = 1e-10
                warnings.warn(
                    "Space-time covariance is borderline PD; adding jitter. "
                    f"Consider increasing nugget (current: {nugget:.2e}).",
                    RuntimeWarning, stacklevel=4,
                )
            else:
                jitter *= 10.0
    raise np.linalg.LinAlgError(
        "Could not obtain positive-definite Cholesky. "
        "Try a larger nugget or smaller deformation parameters."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_VALID  = frozenset({"frozen_matern", "frozen_ch", "lagrangian_matern", "lagrangian_ch"})
_LAG    = frozenset({"lagrangian_matern", "lagrangian_ch"})
_CH     = frozenset({"frozen_ch", "lagrangian_ch"})


def simulate_st(
    n_space: int | tuple[int, int] = 20,
    space_range: tuple[float, float] = (0.0, 1.0),
    t_values=None,
    model: str = "lagrangian_matern",
    sigma2: float = 1.0,
    phi: float = 0.3,
    nu: float = 1.5,
    nugget: float = 1e-4,
    lam0: float = 0.3,
    theta0: float = 0.0,
    rho: float = 1.0,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    theta_Lambda: float = 0.0,
    tail: float = 1.0,
    seed=None,
) -> STSimulationResult:
    """
    Simulate one realisation of a spatio-temporal GP on a regular 2-D grid.

    Parameters
    ----------
    n_space      : int or (n_h, n_w)
                   Spatial grid resolution.  Keep ≤ 25 for fast computation.
    space_range  : (lo, hi)  Spatial domain extent for both axes.
    t_values     : array_like or None
                   Time points.  Defaults to np.linspace(0, 1, 6).
    model        : "frozen_matern" | "lagrangian_matern" | "frozen_ch" | "lagrangian_ch"
    sigma2       : float  Marginal variance σ².
    phi          : float  Spatial range φ.
    nu           : float  Smoothness ν.
    nugget       : float  Nugget ratio (added as σ² × nugget × I).
    lam0         : float  Advection speed λ₀ (domain units / time unit).
    theta0       : float  Advection direction θ₀ (radians, 0 = East).
    rho          : float  Temporal deformation scale ρ > 0 (Lagrangian only).
                          Large ρ → slow deformation; ρ → ∞ → frozen field.
    lambda1      : float  Larger eigenvalue of deformation matrix Λ.
    lambda2      : float  Smaller eigenvalue of Λ.
    theta_Lambda : float  Rotation angle of Λ (radians).
    tail         : float  Tail-decay parameter α > 0 (CH models only;
                          ignored for Matérn models).  Default 0.5.
    seed         : int or None  Random seed.

    Returns
    -------
    STSimulationResult
        .field  (n_t, n_h, n_w) — spatial snapshots at each time step
        .S1, .S2  (n_h, n_w)   — spatial coordinate meshgrids
        .t      (n_t,)          — time points
        .lam_vec (2,)           — property: advection vector λ₀[cos θ₀, sin θ₀]
    """
    m = model.lower()
    if m not in _VALID:
        raise ValueError(f"model '{model}' not in {sorted(_VALID)}.")

    t = np.linspace(0.0, 1.0, 6) if t_values is None else np.asarray(t_values, float).ravel()

    n_h, n_w = (int(n_space), int(n_space)) if np.isscalar(n_space) \
               else (int(n_space[0]), int(n_space[1]))

    lo, hi   = space_range
    S1m, S2m = np.meshgrid(np.linspace(lo, hi, n_w), np.linspace(lo, hi, n_h))
    S        = np.column_stack([S1m.ravel(), S2m.ravel()])   # (n_s, 2)

    lam_vec = lam_vec_from_polar(lam0, theta0)
    is_lag  = m in _LAG
    use_ch  = m in _CH
    Lambda  = build_Lambda(lambda1, lambda2, theta_Lambda) if is_lag else None

    K = _build_st_cov(S, t, sigma2, nugget, phi, nu, lam_vec,
                      is_lag, Lambda, rho, use_ch=use_ch, tail=tail)
    L = _safe_chol(K, nugget)

    rng   = np.random.default_rng(seed)
    field = (L @ rng.standard_normal(len(K))).reshape(len(t), n_h, n_w)

    return STSimulationResult(
        field=field, S1=S1m, S2=S2m, t=t,
        params=dict(phi=phi, nu=nu, tail=tail, lam0=lam0, theta0=theta0,
                    rho=rho, lambda1=lambda1, lambda2=lambda2, theta_Lambda=theta_Lambda),
        sigma2=sigma2, nugget=nugget, model=m,
    )


__all__ = ["simulate_st", "STSimulationResult"]

