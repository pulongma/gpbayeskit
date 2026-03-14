"""
kernels.py
----------
Kernel (correlation) functions for spatial GP models.

``KernelBuilder`` composes a *kernel family* with an *anisotropy form* and
exposes a single ``__call__`` that returns an (n1, n2) correlation matrix.

Kernel families
---------------
  matern    General Matérn with smoothness nu > 0
  exp       Exponential  — Matérn(nu=0.5)
  matern32  Matérn-3/2   — Matérn(nu=1.5), fixed
  matern52  Matérn-5/2   — Matérn(nu=2.5), fixed
  ch        Confluent Hypergeometric function
  gauss     Squared-exponential (Gaussian), nu → ∞

Anisotropy forms
----------------
  isotropic  Single scale phi (scalar).  Distance = ||x-y||₂ / phi.
  ard        One phi_j per dimension.    Distance = ||diag(phi)⁻¹(x-y)||₂.
  tensor     Product kernel.             R = ∏ⱼ k(|xⱼ-yⱼ| / phi_j).

Matérn parameterisation
-----------------------
All Matérn variants use the *range* parameterisation with z = r / phi
(no sqrt(2*nu) pre-scaling).  The general form is

    C(r) = [2^{1-nu} / Gamma(nu)] * (r/phi)^nu * K_nu(r/phi)

and the closed-form special cases are

    nu=0.5 : exp(−r/phi)
    nu=1.5 : (1 + r/phi) * exp(−r/phi)
    nu=2.5 : (1 + r/phi + (r/phi)²/3) * exp(−r/phi)

This is a common alternative to the Rasmussen & Williams convention
(which uses sqrt(2*nu)*r/phi); the two are equivalent up to a
reparameterisation of phi.  The range parameter phi in this code
therefore has units of *distance* and is directly interpretable as the
practical correlation length.
"""

from __future__ import annotations

import numpy as np
import scipy.special as sc

from gpbayeskit.utils import euclidean_dist, axis_dists


# ─────────────────────────────────────────────────────────────────────────────
# Valid options (mirrors SpatialGP for consistent validation)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_KERNEL_TYPES: frozenset[str] = frozenset(
    {"matern", "exp", "matern32", "matern52", "ch", "gauss"}
)
_VALID_KERNEL_FORMS: frozenset[str] = frozenset(
    {"isotropic", "ard", "tensor"}
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure kernel functions  (operate on *already-scaled* scalar distance arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _matern(z: np.ndarray, nu: float) -> np.ndarray:
    """
    Matérn correlation evaluated at z = r / phi  (range parameterisation).

    The output is 1 wherever z == 0 (same-location pairs).

    Parameters
    ----------
    z   : non-negative distance array (any shape), z = r / phi
    nu  : smoothness > 0

    Returns
    -------
    ndarray of same shape as z, values in [0, 1].
    """
    z = np.asarray(z, dtype=float)

    # --- fast closed-form special cases ---
    if nu == 0.5:
        return np.exp(-z)

    if nu == 1.5:
        return (1.0 + z) * np.exp(-z)

    if nu == 2.5:
        return (1.0 + z + z ** 2 / 3.0) * np.exp(-z)

    # --- general nu via modified Bessel function K_nu ---
    out = np.ones_like(z)
    nz  = z != 0.0
    if np.any(nz):
        out[nz] = (
            (2.0 ** (1.0 - nu) / sc.gamma(nu))
            * np.power(z[nz], nu)
            * sc.kv(nu, z[nz])
        )

    return out


def _ch(z: np.ndarray, nu: float, tail: float) -> np.ndarray:
    """
    Confluent hypergeometric correlation from Ma (2025; https://arxiv.org/abs/2511.07959):

        C(r) = Γ(nu+tail)/Γ(nu) · U(tail, 1−nu, (r/phi)²)

    where U is Tricomi's confluent hypergeometric function. This is original developed by 
    Ma and Bhadra (2023; https://doi.org/10.1080/01621459.2022.2027775)

    Parameters
    ----------
    z    : r/phi  
    nu   : smoothness > 0
    tail : tail/shape > 0

    Returns
    -------
    ndarray, same shape as z.
    """
    z    = np.asarray(z, dtype=float)
    const = np.exp(sc.gammaln(nu + tail) - sc.gammaln(nu))
    out   = const * sc.hyperu(tail, 1.0 - nu, z ** 2)
    out   = np.where(z == 0.0, 1.0, out)
    return out


def _gauss(z: np.ndarray) -> np.ndarray:
    """
    Squared-exponential (Gaussian) correlation:  exp(-(r/phi)²).

    Parameters
    ----------
    z : r/phi

    Returns
    -------
    ndarray, same shape as z.
    """
    return np.exp(-(z ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Single-dimension dispatcher  (evaluate one kernel on a pre-scaled distance)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_kernel(
    r_scaled: np.ndarray,
    kernel_type: str,
    nu: float = 2.5,
    tail: float = 0.5,
) -> np.ndarray:
    """
    Evaluate *kernel_type* on a pre-scaled distance array *r_scaled* = r / phi.

    For all kernels the caller passes z = r / phi directly;  

    This is the **single dispatch point** used by iso, ARD, and tensor paths,
    eliminating repeated if/elif chains.

    Parameters
    ----------
    r_scaled    : non-negative distance array (any shape), r_scaled = r / phi
    kernel_type : one of _VALID_KERNEL_TYPES
    nu          : smoothness (Matérn / CH only)
    tail        : tail parameter (CH only)

    Returns
    -------
    Correlation values, same shape as r_scaled.
    """
    if kernel_type == "matern":
        return _matern(r_scaled, nu)
    if kernel_type == "exp":
        return _matern(r_scaled, 0.5)
    if kernel_type == "matern32":
        return _matern(r_scaled, 1.5)
    if kernel_type == "matern52":
        return _matern(r_scaled, 2.5)
    if kernel_type == "ch":
        return _ch(r_scaled, nu, tail)
    if kernel_type == "gauss":
        return _gauss(r_scaled)
    raise ValueError(
        f"Unknown kernel_type '{kernel_type}'. "
        f"Choose from {sorted(_VALID_KERNEL_TYPES)}."
    )



# ─────────────────────────────────────────────────────────────────────────────
# KernelBuilder
# ─────────────────────────────────────────────────────────────────────────────

class KernelBuilder:
    """
    Composable kernel builder: family × anisotropy form → (n1, n2) matrix.

    Parameters
    ----------
    kernel_type : str  Kernel family (see module docstring).
    kernel_form : str  Anisotropy form (see module docstring).

    Examples
    --------
    >>> kb = KernelBuilder("matern52", "ard")
    >>> R  = kb(X1, X2, phi=[0.5, 1.2])           # (n1, n2)
    """

    def __init__(
        self,
        kernel_type: str = "matern",
        kernel_form: str = "isotropic",
    ) -> None:
        kt = kernel_type.lower()
        kf = kernel_form.lower()

        if kt not in _VALID_KERNEL_TYPES:
            raise ValueError(
                f"kernel_type '{kernel_type}' not recognised. "
                f"Choose from {sorted(_VALID_KERNEL_TYPES)}."
            )
        if kf not in _VALID_KERNEL_FORMS:
            raise ValueError(
                f"kernel_form '{kernel_form}' not recognised. "
                f"Choose from {sorted(_VALID_KERNEL_FORMS)}."
            )

        self.kernel_type = kt
        self.kernel_form = kf

    def __repr__(self) -> str:
        return (
            f"KernelBuilder("
            f"kernel_type='{self.kernel_type}', "
            f"kernel_form='{self.kernel_form}')"
        )

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _distance(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise distances appropriate for the kernel form.

        isotropic → (n1, n2)    Euclidean distances
        ard/tensor→ (n1, n2, d) per-axis absolute differences
        """
        if self.kernel_form == "isotropic":
            return euclidean_dist(X1, X2)
        return axis_dists(X1, X2)    # (n1, n2, d)  for ard and tensor

    # ------------------------------------------------------------------
    # Per-form kernel evaluation
    # ------------------------------------------------------------------

    def _iso_kernel(
        self,
        D: np.ndarray,
        phi: float,
        nu: float,
        tail: float,
    ) -> np.ndarray:
        """
        Isotropic kernel:  R_ij = k(||xi - xj||₂ / phi).

        D : (n1, n2)  Euclidean distances.
        """
        return _eval_kernel(D / phi, self.kernel_type, nu=nu, tail=tail)

    def _ard_kernel(
        self,
        D: np.ndarray,
        phi: np.ndarray,
        nu: float,
        tail: float,
    ) -> np.ndarray:
        """
        ARD kernel:  R_ij = k(||diag(phi)⁻¹ (xi−xj)||₂).

        D   : (n1, n2, d)  per-axis absolute differences.
        phi : scalar or (d,) — broadcast to d if scalar.
        """
        d   = D.shape[2]                              # inferred from data
        phi = np.asarray(phi, dtype=float).ravel()
        if phi.size == 1:
            phi = np.full(d, phi.item())              # broadcast scalar → (d,)
        if phi.size != d:
            raise ValueError(
                f"ARD kernel expects phi of length {d} (one per dimension), "
                f"got {phi.size}."
            )
        # Weighted Euclidean: sqrt(sum_j (D_j / phi_j)^2)
        z_sq = np.sum((D / phi.reshape(1, 1, -1)) ** 2, axis=-1)  # (n1, n2)
        z    = np.sqrt(z_sq)
        return _eval_kernel(z, self.kernel_type, nu=nu, tail=tail)

    def _tensor_kernel(
        self,
        D: np.ndarray,
        phi: np.ndarray,
        nu: float | np.ndarray,
        tail: float | np.ndarray,
    ) -> np.ndarray:
        """
        Tensor-product kernel:  R_ij = ∏_j k_j(|xi_j − xj_j| / phi_j).

        Each dimension gets its own 1-D kernel evaluation; the results are
        multiplied across dimensions.

        D    : (n1, n2, d)  per-axis absolute differences.
        phi  : scalar or (d,) — broadcast to d if scalar.
        nu   : scalar or (d,) — smoothness per dimension.
        tail : scalar or (d,) — tail parameter per dimension (CH only).
        """
        d   = D.shape[2]                              # inferred from data
        phi = np.asarray(phi, dtype=float).ravel()
        if phi.size == 1:
            phi = np.full(d, phi.item())              # broadcast scalar → (d,)
        if phi.size != d:
            raise ValueError(
                f"Tensor kernel expects phi of length {d} (one per dimension), "
                f"got {phi.size}."
            )

        nu_arr   = np.broadcast_to(np.asarray(nu,   dtype=float), (d,)).copy()
        tail_arr = np.broadcast_to(np.asarray(tail, dtype=float), (d,)).copy()

        R = np.ones(D.shape[:2])   # (n1, n2)
        for j in range(d):
            dj  = D[:, :, j]                         # (n1, n2)
            R  *= _eval_kernel(
                dj / phi[j],
                self.kernel_type,
                nu=nu_arr[j],
                tail=tail_arr[j],
            )
        return R

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        phi: float | np.ndarray = 1.0,
        nu: float | np.ndarray = 2.5,
        tail: float | np.ndarray = 0.5,
        # `dim` accepted but ignored — inferred from X1.shape[1]
        dim: int | None = None,
    ) -> np.ndarray:
        """
        Build the (n1, n2) correlation matrix between X1 and X2.

        Parameters
        ----------
        X1, X2 : (n1, d), (n2, d)  input locations
        phi    : range / scale — scalar for isotropic, (d,) for ARD/tensor
        nu     : smoothness (Matérn / CH); scalar or (d,) for tensor
        tail   : tail parameter (CH only);  scalar or (d,) for tensor
        dim    : *deprecated* — ignored, retained for backward compatibility

        Returns
        -------
        R : (n1, n2)  correlation matrix with values in [0, 1]
        """
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("X1 and X2 must be 2-D arrays.")
        if X1.shape[1] != X2.shape[1]:
            raise ValueError(
                f"X1 has {X1.shape[1]} columns but X2 has {X2.shape[1]}."
            )

        D = self._distance(X1, X2)

        if self.kernel_form == "isotropic":
            phi_scalar = float(np.asarray(phi).ravel()[0])
            return self._iso_kernel(D, phi=phi_scalar, nu=float(nu), tail=float(tail))

        if self.kernel_form == "ard":
            return self._ard_kernel(D, phi=phi, nu=float(nu), tail=float(tail))

        if self.kernel_form == "tensor":
            return self._tensor_kernel(D, phi=phi, nu=nu, tail=tail)

        raise ValueError(f"Unknown kernel_form '{self.kernel_form}'.")