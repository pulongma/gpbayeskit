"""
gpbayeskit.models.spatial
==========================
Spatial GP model — six kernel families × three anisotropy forms.

This class is now a thin layer: it declares its parameters with
``Parameter``, sets up phi's vector size for ARD/tensor kernels in
``_setup_params()``, and implements ``covariance()``.  All fitting,
prediction, and summary logic lives in ``BaseGP``.

Kernel families
---------------
  matern    general Matérn (nu free)
  exp       Matérn(nu=0.5) — nu fixed
  matern32  Matérn(nu=1.5) — nu fixed
  matern52  Matérn(nu=2.5) — nu fixed
  ch        confluent hypergeometric (adds tail parameter)
  gauss     squared-exponential (nu=∞, fixed)

Kernel forms
------------
  isotropic  single phi
  ard        one phi per spatial dimension
  tensor     product kernel, one phi per dimension
"""

from __future__ import annotations

import numpy as np

from gpbayeskit.models.base import BaseGP
from gpbayeskit.parameters import Parameter
from gpbayeskit.kernels._spatial import KernelBuilder


_FIXED_NU = {"exp": 0.5, "matern32": 1.5, "matern52": 2.5, "gauss": np.inf}
_DEFAULT_NU = {"matern": 1.5, "ch": 1.5, **_FIXED_NU}

_VALID_KERNEL_TYPES = frozenset({"matern", "exp", "matern32", "matern52", "ch", "gauss"})
_VALID_KERNEL_FORMS = frozenset({"isotropic", "ard", "tensor"})


class SpatialGP(BaseGP):
    """
    Spatial GP with composable kernel family and anisotropy form.

    Parameters declared
    -------------------
    phi    : range / scale  (scalar for isotropic, d-vector for ARD/tensor)
    nu     : smoothness  (auto-fixed for exp, matern32, matern52, gauss)
    tail   : tail shape  (only free for CH kernel, otherwise fixed)
    nugget : noise variance ratio

    Examples
    --------
    >>> gp = SpatialGP(X_train, y_train, kernel_type="matern52", kernel_form="ard")
    >>> gp.fit(phi_init=0.3)          # nu auto-fixed to 2.5
    >>> mean, var = gp.predict(X_test)

    >>> gp_ch = SpatialGP(X_train, y_train, kernel_type="ch")
    >>> gp_ch.fit(fix_nu=1.5)         # tail is free by default
    """

    # ── Parameter declarations ────────────────────────────────────────────────
    # phi size is 1 here; _setup_params() resizes it for ARD/tensor.
    phi    = Parameter(0.5,  lower=1e-8,       doc="Range / scale")
    nu     = Parameter(1.5,  lower=1e-8,       doc="Matérn smoothness")
    tail   = Parameter(0.5,  lower=1e-8,       doc="CH tail shape (α)")
    nugget = Parameter(1e-6, lower=0.0,        doc="Nugget ratio")

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kernel_type: str = "matern",
        kernel_form: str = "isotropic",
        H: np.ndarray | None = None,
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
        self.kernel_type    = kt
        self.kernel_form    = kf
        self.kernel_builder = KernelBuilder(kernel_type=kt, kernel_form=kf)
        super().__init__(X, y, H=H)

    def __repr__(self) -> str:
        status = "fitted" if self.params_ is not None else "unfitted"
        phi_str = ""
        if self.params_ is not None:
            phi = self.params_["phi"]
            phi_str = (
                f", phi={phi:.4g}"
                if np.isscalar(phi)
                else f", phi=[{', '.join(f'{v:.4g}' for v in phi)}]"
            )
        return (
            f"SpatialGP(n={self.n}, d={self.d}, "
            f"kernel='{self.kernel_type}/{self.kernel_form}'"
            f"{phi_str}, status={status})"
        )

    # ── _setup_params: resize phi for ARD/tensor; fix nu and tail ────────────

    def _setup_params(self) -> None:
        # Expand phi to a d-vector for ARD and tensor kernels
        if self.kernel_form in ("ard", "tensor"):
            self._resize_param("phi", self.d)

        # Fix nu for families where it is not a free parameter
        if self.kernel_type in _FIXED_NU:
            self._param_store["nu"].fix(_FIXED_NU[self.kernel_type])
        else:
            # Set a sensible default initial value
            self._param_store["nu"].set_value(_DEFAULT_NU.get(self.kernel_type, 1.5))

        # Fix tail for all non-CH kernels
        if self.kernel_type != "ch":
            self._param_store["tail"].fix(0.5)

    # ── covariance ────────────────────────────────────────────────────────────

    def covariance(self, X1, X2, **params) -> np.ndarray:
        """(n1, n2) correlation matrix via KernelBuilder."""
        return self.kernel_builder(
            np.asarray(X1, dtype=float),
            np.asarray(X2, dtype=float),
            phi=params["phi"],
            nu=float(params.get("nu", 1.5)),
            tail=float(params.get("tail", 0.5)),
        )