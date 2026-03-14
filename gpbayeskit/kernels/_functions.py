"""
gpbayeskit.kernels._kernels
==============================
Isotropic correlation kernels used by the Lagrangian covariance models.

Matérn kernel
-------------
    M(r; ν) = 2^{1-ν} / Γ(ν) · rᵛ · Kᵥ(r)

where Kᵥ is the modified Bessel function of the second kind.
Returns 1 at r = 0 by continuity.

CH kernel (Cauchy-type / confluent hypergeometric)
---------------------------------------------------
    CH(r; ν, α, β) = Γ(ν+α) / Γ(ν) · U(α, 1-ν, (r/β)²)

where U is Tricomi's confluent hypergeometric function.
"""

from __future__ import annotations

import numpy as np
from scipy.special import kv, gamma, hyperu

__all__ = ["matern_kernel", "ch_kernel"]


def matern_kernel(r: float, nu: float) -> float:
    """
    Matérn isotropic correlation kernel (without explicit range parameter):

        M(r; ν) = 2^{1-ν} / Γ(ν) · rᵛ · Kᵥ(r)

    The spatial range φ is applied by the caller as M(r/φ; ν).

    Parameters
    ----------
    r : float
        Scaled distance r ≥ 0.
    nu : float
        Smoothness parameter ν > 0.
        Common choices: 0.5 (exponential), 1.5, 2.5, ∞ (Gaussian).

    Returns
    -------
    float
        Correlation value in [0, 1].

    Notes
    -----
    The limit at r → 0 is 1, handled explicitly to avoid 0·∞.
    """
    if nu <= 0:
        raise ValueError(f"Smoothness nu must be positive; got {nu}.")
    if r == 0.0:
        return 1.0
    return (2.0 ** (1.0 - nu) / gamma(nu)) * (abs(r) ** nu) * kv(nu, abs(r))


def ch_kernel(r: float, nu: float, alpha: float, beta: float) -> float:
    """
    CH (Cauchy-type / confluent hypergeometric) isotropic correlation kernel:

        CH(r; ν, α, β) = Γ(ν+α) / Γ(ν) · U(α, 1-ν, (r/β)²)

    where U is Tricomi's confluent hypergeometric function
    (`scipy.special.hyperu`).

    Parameters
    ----------
    r : float
        Distance r ≥ 0.
    nu : float
        Smoothness parameter ν > 0.
    alpha : float
        Tail-decay parameter α > 0.  Larger α gives heavier tails.
    beta : float
        Scale (range) parameter β > 0.

    Returns
    -------
    float
        Correlation value.
    """
    if nu <= 0:
        raise ValueError(f"Smoothness nu must be positive; got {nu}.")
    if alpha <= 0:
        raise ValueError(f"Tail-decay alpha must be positive; got {alpha}.")
    if beta <= 0:
        raise ValueError(f"Scale beta must be positive; got {beta}.")
    prefactor = gamma(nu + alpha) / gamma(nu)

    if r == 0.0:
        return 1.0
    return prefactor * hyperu(alpha, 1.0 - nu, (r / beta) ** 2)