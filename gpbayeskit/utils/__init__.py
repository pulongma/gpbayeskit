"""
gpbayeskit.utils
================
Shared utilities: input validation, distance functions, scoring metrics.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

__all__ = [
    "euclidean_dist",
    "axis_dists",
    "great_circle_dist",
    "chordal_dist",
    "validate_Xy",
    "validate_H",
    "gp_scores",
]


# ── Input validation ──────────────────────────────────────────────────────────

def validate_Xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} elements."
        )
    return X, y.reshape(-1, 1)


def validate_H(H, n):
    if H is None:
        return np.ones((n, 1), dtype=float)
    H = np.asarray(H, dtype=float)
    if H.ndim == 1:
        H = H.reshape(-1, 1)
    if H.shape[0] != n:
        raise ValueError(
            f"H must have {n} rows (same as X), got {H.shape[0]}."
        )
    return H


# ── Distance functions ────────────────────────────────────────────────────────

def euclidean_dist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distances between rows of X1 and X2.

    Parameters
    ----------
    X1 : (n1, d)
    X2 : (n2, d)

    Returns
    -------
    D : (n1, n2)
    """
    return cdist(X1, X2, metric="euclidean")


def axis_dists(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Coordinate-wise (per-axis) absolute differences.
    Used by ARD and tensor-product kernels.

    Parameters
    ----------
    X1 : (n1, d)
    X2 : (n2, d)

    Returns
    -------
    D : (n1, n2, d)  — D[i, j, k] = |X1[i,k] - X2[j,k]|
    """
    return np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :])


def great_circle_dist(
    X1: np.ndarray,
    X2: np.ndarray,
    radius: float = 6_371.0,
) -> np.ndarray:
    """
    Great-circle (geodesic) distances via the Haversine formula.

    Parameters
    ----------
    X1, X2 : (n, 2)  columns are [latitude, longitude] in **degrees**.
    radius  : float  sphere radius (default 6 371 km for Earth).

    Returns
    -------
    D : (n1, n2)  distances in the same units as *radius*.

    Notes
    -----
    Valid covariance functions on the sphere should be applied to
    *chordal* distances (see ``chordal_dist``), not great-circle arcs,
    to guarantee positive definiteness (Gneiting 2013).
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    _check_latlon(X1)
    _check_latlon(X2)

    lat1 = np.radians(X1[:, 0])
    lon1 = np.radians(X1[:, 1])
    lat2 = np.radians(X2[:, 0])
    lon2 = np.radians(X2[:, 1])

    dlat = lat1[:, None] - lat2[None, :]
    dlon = lon1[:, None] - lon2[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon / 2) ** 2
    )
    return radius * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def chordal_dist(
    X1: np.ndarray,
    X2: np.ndarray,
    radius: float = 6_371.0,
) -> np.ndarray:
    """
    Chordal (straight-line through the sphere) distances.

    Preferred over great-circle arcs when building covariance matrices on
    the sphere because isotropic functions of chordal distance are always
    positive-definite on S^2.

    Parameters
    ----------
    X1, X2 : (n, 2)  columns are [latitude, longitude] in **degrees**.
    radius  : float  sphere radius (default 6 371 km).

    Returns
    -------
    D : (n1, n2)  chord lengths in the same units as *radius*.
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    _check_latlon(X1)
    _check_latlon(X2)

    C1 = _to_cartesian(X1) * radius   # (n1, 3)
    C2 = _to_cartesian(X2) * radius   # (n2, 3)
    return cdist(C1, C2, metric="euclidean")


def _to_cartesian(X: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) in degrees to unit-sphere Cartesian (x, y, z)."""
    lat = np.radians(X[:, 0])
    lon = np.radians(X[:, 1])
    return np.stack(
        [np.cos(lat) * np.cos(lon),
         np.cos(lat) * np.sin(lon),
         np.sin(lat)],
        axis=1,
    )


def _check_latlon(X: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Lat/lon arrays must have shape (n, 2).")
    if not np.all(np.abs(X[:, 0]) <= 90):
        raise ValueError("Latitudes must be in [-90, 90].")
    if not np.all(np.abs(X[:, 1]) <= 180):
        raise ValueError("Longitudes must be in [-180, 180].")


# ── Scoring ───────────────────────────────────────────────────────────────────

def gp_scores(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float]:
    """
    Compute standard evaluation metrics for GP predictions.

    Parameters
    ----------
    y_true   : (n,)  observed values
    mean     : (n,)  predictive mean
    variance : (n,)  predictive variance  (must be >= 0)

    Returns
    -------
    dict with keys:
        rmse         root mean squared error
        mae          mean absolute error
        coverage_95  empirical coverage of the 95 % predictive interval
        ci_length    mean length of the 95 % predictive interval
        crps         mean continuous ranked probability score (lower = better)
        log_score    mean log predictive density (higher = better)
    """
    y_true   = np.asarray(y_true,   dtype=float).ravel()
    mean     = np.asarray(mean,     dtype=float).ravel()
    variance = np.asarray(variance, dtype=float).ravel()

    if np.any(variance < 0):
        warnings.warn("Negative variances clipped to zero.", RuntimeWarning, stacklevel=2)
    variance = np.clip(variance, 0.0, None)
    std = np.sqrt(variance)

    # Point accuracy
    rmse = float(np.sqrt(np.mean((y_true - mean) ** 2)))
    mae  = float(np.mean(np.abs(y_true - mean)))

    # 95 % predictive interval
    z           = 1.96
    lower       = mean - z * std
    upper       = mean + z * std
    coverage_95 = float(np.mean((y_true >= lower) & (y_true <= upper)))
    ci_length   = float(np.mean(upper - lower))

    # CRPS (closed-form for Gaussian)
    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = np.where(std > 0, (y_true - mean) / std, 0.0)
    crps = std * (
        zscore * (2.0 * norm.cdf(zscore) - 1.0)
        + 2.0 * norm.pdf(zscore)
        - 1.0 / np.sqrt(np.pi)
    )
    crps_mean = float(np.mean(crps))

    # Log predictive score
    with np.errstate(divide="ignore"):
        log_score = float(np.mean(norm.logpdf(y_true, mean, std)))

    return {
        "rmse":        rmse,
        "mae":         mae,
        "coverage_95": coverage_95,
        "ci_length":   ci_length,
        "crps":        crps_mean,
        "log_score":   log_score,
    }