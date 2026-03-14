"""
gpbayeskit.kernels._parametrisation
=======================================
Helpers to build the model parameters:

  λ  = λ₀ [cos θ₀, sin θ₀]             (advection vector)
  Λ  = U diag(λ₁, λ₂) Uᵀ              (anisotropy matrix)
  U  = 2-D rotation matrix for angle θ
"""

from __future__ import annotations

import numpy as np

__all__ = ["rotation_matrix", "build_Lambda", "lam_vec_from_polar"]


def rotation_matrix(theta: float) -> np.ndarray:
    """
    2-D rotation matrix for angle *theta* (radians).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    U : ndarray, shape (2, 2)
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def build_Lambda(
    lambda1: float,
    lambda2: float,
    theta: float,
) -> np.ndarray:
    """
    Construct the 2×2 anisotropy matrix:

        Λ = U diag(λ₁, λ₂) Uᵀ

    where U is the 2-D rotation matrix for angle *theta*.

    Parameters
    ----------
    lambda1 : float
        First (larger) eigenvalue of Λ.  Must be positive.
    lambda2 : float
        Second eigenvalue of Λ.  Must be positive.
    theta : float
        Rotation angle θ (radians) that aligns the principal axes of
        anisotropy with the coordinate frame.

    Returns
    -------
    Lambda : ndarray, shape (2, 2)
        Symmetric positive-definite anisotropy matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from gpbayeskit.covariance import build_Lambda
    >>> build_Lambda(0.8, 0.2, theta=np.deg2rad(45))
    array([[0.5, 0.3],
           [0.3, 0.5]])   # approximate
    """
    if lambda1 <= 0 or lambda2 <= 0:
        raise ValueError("Eigenvalues lambda1 and lambda2 must be positive.")
    U = rotation_matrix(theta)
    return U @ np.diag([lambda1, lambda2]) @ U.T


def lam_vec_from_polar(lam0: float, theta0: float) -> np.ndarray:
    """
    Construct the advection vector from its polar representation:

        λ = λ₀ [cos θ₀, sin θ₀]

    Parameters
    ----------
    lam0 : float
        Advection magnitude λ₀.  Must be non-negative.
    theta0 : float
        Advection direction θ₀ (radians), measured counter-clockwise
        from the positive h₁-axis.

    Returns
    -------
    lam_vec : ndarray, shape (2,)

    Examples
    --------
    >>> import numpy as np
    >>> from gpbayeskit.covariance import lam_vec_from_polar
    >>> lam_vec_from_polar(1.0, np.deg2rad(45))
    array([0.707, 0.707])   # approximate
    """
    if lam0 < 0:
        raise ValueError("Advection magnitude lam0 must be non-negative.")
    return lam0 * np.array([np.cos(theta0), np.sin(theta0)])