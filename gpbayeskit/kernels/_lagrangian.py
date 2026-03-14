"""
gpbayeskit.kernels._lagrangian
================================
``LagrangianKernelBuilder`` — spatio-temporal kernel builder that mirrors
``KernelBuilder`` for the four Lagrangian / frozen-field covariance models.

Models
------
  frozen_matern      σ² M( |h − uλ| / φ ; ν )
  frozen_ch          σ² CH( |h − uλ| ; ν, α, β )
  lagrangian_matern  σ² |I + ρ²Λ/2|^{−½} M( h_u / φ ; ν )
  lagrangian_ch      σ² |I + ρ²Λ/2|^{−½} CH( h_u ; ν, α, β )

where the Lagrangian distance is

    h_u = sqrt( (h − u·λ)ᵀ (u²Λ + I)⁻¹ (h − u·λ) )

and the advection vector and anisotropy matrix are parameterised as

    λ  = λ₀ [cos θ₀, sin θ₀]
    Λ  = U diag(λ₁, λ₂) Uᵀ,   U = rotation matrix for angle θ_Λ

The builder's ``__call__`` takes *separate* spatial and temporal arrays
(S1, S2 of shape (n1/n2, d_space) and T1, T2 of shape (n1/n2,)) and returns
an (n1, n2) correlation matrix.  The sigma² scaling is handled upstream by
``SpatioTemporalGP``.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import det, inv

from gpbayeskit.kernels._functions import matern_kernel, ch_kernel
from gpbayeskit.kernels._parametrisation import (
    lam_vec_from_polar,
    build_Lambda,
)


# ── Valid model names ─────────────────────────────────────────────────────────

_VALID_MODELS: frozenset[str] = frozenset(
    {"frozen_matern", "frozen_ch", "lagrangian_matern", "lagrangian_ch"}
)

_FROZEN_MODELS:    frozenset[str] = frozenset({"frozen_matern", "frozen_ch"})
_MATERN_MODELS:    frozenset[str] = frozenset({"frozen_matern", "lagrangian_matern"})
_CH_MODELS:        frozenset[str] = frozenset({"frozen_ch", "lagrangian_ch"})
_LAGRANGIAN_MODELS: frozenset[str] = frozenset({"lagrangian_matern", "lagrangian_ch"})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _det_prefactor(Lambda: np.ndarray, rho: float, d: int) -> float:
    """Compute |I_d + ρ²Λ/2|^{−1/2}."""
    M = np.eye(d) + (rho ** 2 / 2.0) * Lambda
    return float(det(M) ** (-0.5))


def _lagrangian_distance(
    h: np.ndarray,
    u: float,
    lam_vec: np.ndarray,
    Lambda: np.ndarray,
) -> float:
    """
    Compute h_u = sqrt( (h − u·λ)ᵀ (u²Λ + I)⁻¹ (h − u·λ) ).
    """
    d     = len(h)
    diff  = h - u * lam_vec
    A_inv = inv(u ** 2 * Lambda + np.eye(d))
    return np.sqrt(max(float(diff @ A_inv @ diff), 0.0))


# ── LagrangianKernelBuilder ───────────────────────────────────────────────────

class LagrangianKernelBuilder:
    """
    Spatio-temporal kernel builder for Lagrangian and frozen-field models.

    Mirrors the interface of ``KernelBuilder``: construct once with the model
    name, then call with data arrays and parameter values to obtain a
    correlation matrix.

    Parameters
    ----------
    model : str
        One of ``"frozen_matern"``, ``"frozen_ch"``,
        ``"lagrangian_matern"``, ``"lagrangian_ch"``.

    Examples
    --------
    >>> import numpy as np
    >>> kb = LagrangianKernelBuilder("lagrangian_matern")
    >>> R  = kb(S1, S2, T1, T2, phi=1.5, nu=1.5, rho=0.8,
    ...         lam0=0.5, theta0=np.deg2rad(30),
    ...         lambda1=0.8, lambda2=0.2, theta_Lambda=np.deg2rad(45))
    """

    def __init__(self, model: str = "lagrangian_matern") -> None:
        m = model.lower()
        if m not in _VALID_MODELS:
            raise ValueError(
                f"model '{model}' is not recognised. "
                f"Choose from {sorted(_VALID_MODELS)}."
            )
        self.model = m

    def __repr__(self) -> str:
        return f"LagrangianKernelBuilder(model='{self.model}')"

    # ── Model-type predicates ─────────────────────────────────────────────────

    @property
    def is_frozen(self) -> bool:
        """True for frozen-field models (no Λ or ρ parameters)."""
        return self.model in _FROZEN_MODELS

    @property
    def uses_tail(self) -> bool:
        """True for CH models (have a tail/shape parameter)."""
        return self.model in _CH_MODELS

    # ── Single-pair covariance (normalised by sigma²) ─────────────────────────

    def _eval_pair(
        self,
        h: np.ndarray,
        u: float,
        phi: float,
        nu: float,
        tail: float,
        lam_vec: np.ndarray,
        # Lagrangian-only (ignored for frozen models)
        rho: float,
        Lambda: np.ndarray | None,
    ) -> float:
        """
        Evaluate the *correlation* (C / σ²) for a single (h, u) pair.
        """
        if self.is_frozen:
            r = float(np.linalg.norm(h - u * lam_vec))
            if self.model == "frozen_matern":
                return matern_kernel(r / phi, nu)
            return ch_kernel(r, nu, tail, phi)   # beta = phi for CH frozen

        # --- full Lagrangian ---
        d         = len(h)
        hu        = _lagrangian_distance(h, u, lam_vec, Lambda)
        prefactor = _det_prefactor(Lambda, rho, d)

        if self.model == "lagrangian_matern":
            return prefactor * matern_kernel(hu / phi, nu)
        return prefactor * ch_kernel(hu, nu, tail, phi)   # beta = phi

    # ── Public interface ──────────────────────────────────────────────────────

    def __call__(
        self,
        S1: np.ndarray,
        S2: np.ndarray,
        T1: np.ndarray,
        T2: np.ndarray,
        *,
        phi: float = 1.0,
        nu: float = 1.5,
        tail: float = 0.5,
        lam0: float = 0.0,
        theta0: float = 0.0,
        rho: float = 1.0,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        theta_Lambda: float = 0.0,
    ) -> np.ndarray:
        """
        Build the (n1, n2) correlation matrix between two sets of
        spatio-temporal locations.

        Parameters
        ----------
        S1 : (n1, d_space)  Spatial locations — first set.
        S2 : (n2, d_space)  Spatial locations — second set.
        T1 : (n1,)          Temporal coordinates — first set.
        T2 : (n2,)          Temporal coordinates — second set.
        phi          : Spatial range / scale  φ > 0.
        nu           : Smoothness  ν > 0  (Matérn / CH).
        tail         : Tail parameter  α > 0  (CH only; ignored otherwise).
        lam0         : Advection magnitude  λ₀ ≥ 0.
        theta0       : Advection direction  θ₀ (radians).
        rho          : Temporal range  ρ > 0  (Lagrangian models only).
        lambda1      : Larger eigenvalue of Λ  (Lagrangian models only).
        lambda2      : Smaller eigenvalue of Λ  (Lagrangian models only).
        theta_Lambda : Rotation angle of Λ (radians, Lagrangian models only).

        Returns
        -------
        R : (n1, n2)  correlation matrix with R[i,i] = 1 when S1==S2, T1==T2.
        """
        S1 = np.asarray(S1, dtype=float)
        S2 = np.asarray(S2, dtype=float)
        T1 = np.asarray(T1, dtype=float).ravel()
        T2 = np.asarray(T2, dtype=float).ravel()

        n1, n2   = S1.shape[0], S2.shape[0]
        d_space  = S1.shape[1]

        # Build derived parameters
        lam_vec = lam_vec_from_polar(lam0, theta0)

        if not self.is_frozen:
            Lambda = build_Lambda(lambda1, lambda2, theta_Lambda)
        else:
            Lambda = None

        # Evaluate pairwise
        R = np.empty((n1, n2))
        for i in range(n1):
            for j in range(n2):
                h = S1[i] - S2[j]
                u = float(T1[i] - T2[j])
                R[i, j] = self._eval_pair(
                    h, u,
                    phi=phi, nu=nu, tail=tail,
                    lam_vec=lam_vec,
                    rho=rho, Lambda=Lambda,
                )
        return R