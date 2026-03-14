"""
gpbayeskit.models.spatiotemporal
=================================
Spatio-temporal GP model using Lagrangian and frozen-field covariance
functions.

Training data
-------------
``X`` is an (n, d_space + 1) array where the **last column is time** and
the first ``d_space`` columns are spatial coordinates.

Models
------
  frozen_matern      σ² M( |h − uλ| / φ ; ν )
  frozen_ch          Same with CH kernel.
  lagrangian_matern  σ² |I + ρ²Λ/2|^{−½} M( h_u / φ ; ν )
  lagrangian_ch      Same with CH kernel.

Parameters declared
-------------------
Standard:
    phi    : spatial range
    nu     : smoothness
    tail   : CH tail shape  (fixed for non-CH models)
    nugget : noise variance ratio

Advection (all models):
    lam0   : advection magnitude  λ₀ ≥ 0
    theta0 : advection direction  θ₀ ∈ (−π, π]

Lagrangian only (fixed to defaults for frozen models):
    rho          : temporal range  ρ > 0
    lambda1      : larger eigenvalue of Λ
    lambda2      : smaller eigenvalue of Λ
    theta_Lambda : rotation angle of Λ ∈ (−π, π]

All pack/unpack/fit logic is inherited from BaseGP — this class only
declares parameters and implements covariance().
"""

from __future__ import annotations

import numpy as np

from gpbayeskit.models.base import BaseGP
from gpbayeskit.parameters import Parameter
from gpbayeskit.kernels._lagrangian import LagrangianKernelBuilder


_VALID_MODELS   = frozenset({"frozen_matern", "frozen_ch", "lagrangian_matern", "lagrangian_ch"})
_FROZEN_MODELS  = frozenset({"frozen_matern", "frozen_ch"})
_CH_MODELS      = frozenset({"frozen_ch", "lagrangian_ch"})


class SpatioTemporalGP(BaseGP):
    """
    Spatio-temporal GP with Lagrangian or frozen-field covariance.

    Parameters
    ----------
    X       : (n, d_space + 1) array_like
              Spatial columns first, time in the last column.
    y       : (n,) array_like
    model   : str  — one of ``"frozen_matern"``, ``"frozen_ch"``,
              ``"lagrangian_matern"``, ``"lagrangian_ch"``.
    d_space : int  — number of spatial dimensions (default 2).
    H       : (n, p) or None.

    Examples
    --------
    >>> gp = SpatioTemporalGP(X_train, y_train, model="lagrangian_matern")
    >>> gp.fit(fix_nu=1.5, fix_nugget=1e-2)
    >>> mean, var = gp.predict(X_test)
    >>> gp.summary()
    """

    # ── Standard parameters ───────────────────────────────────────────────────
    phi    = Parameter(1.0,  lower=1e-8,             doc="Spatial range φ")
    nu     = Parameter(1.5,  lower=1e-8,             doc="Smoothness ν")
    tail   = Parameter(0.5,  lower=1e-8,             doc="CH tail shape α")
    nugget = Parameter(1e-6, lower=0.0,              doc="Nugget ratio")

    # ── Advection parameters (all models) ─────────────────────────────────────
    lam0   = Parameter(0.5,  lower=0.0,              doc="Advection magnitude λ₀")
    theta0 = Parameter(0.0,  lower=-np.pi, upper=np.pi, doc="Advection direction θ₀ (rad)")

    # ── Lagrangian parameters (frozen for frozen models) ──────────────────────
    rho          = Parameter(1.0,  lower=1e-8,             doc="Temporal range ρ")
    lambda1      = Parameter(1.0,  lower=1e-8,             doc="Λ eigenvalue 1")
    lambda2      = Parameter(0.5,  lower=1e-8,             doc="Λ eigenvalue 2")
    theta_Lambda = Parameter(0.0,  lower=-np.pi, upper=np.pi, doc="Λ rotation θ_Λ (rad)")

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: str = "lagrangian_matern",
        d_space: int = 2,
        H: np.ndarray | None = None,
    ) -> None:
        m = model.lower()
        if m not in _VALID_MODELS:
            raise ValueError(
                f"model '{model}' not recognised. "
                f"Choose from {sorted(_VALID_MODELS)}."
            )
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != d_space + 1:
            raise ValueError(
                f"X must have shape (n, {d_space + 1}) "
                f"(d_space={d_space} spatial + 1 time column), got {X.shape}."
            )
        self.st_model = m
        self.d_space  = d_space
        self._kb      = LagrangianKernelBuilder(model=m)
        super().__init__(X, y, H=H)

    def __repr__(self) -> str:
        status = "fitted" if self.params_ is not None else "unfitted"
        return (
            f"SpatioTemporalGP(n={self.n}, d_space={self.d_space}, "
            f"model='{self.st_model}', status={status})"
        )

    # ── _setup_params: fix structurally absent parameters ────────────────────

    def _setup_params(self) -> None:
        # Tail only free for CH models
        if self.st_model not in _CH_MODELS:
            self._param_store["tail"].fix(0.5)

        # rho, lambda1/2, theta_Lambda are unused by frozen models
        if self.st_model in _FROZEN_MODELS:
            self._param_store["rho"].fix(1.0)
            self._param_store["lambda1"].fix(1.0)
            self._param_store["lambda2"].fix(1.0)
            self._param_store["theta_Lambda"].fix(0.0)

    # ── covariance ────────────────────────────────────────────────────────────

    def covariance(self, X1, X2, **params) -> np.ndarray:
        """(n1, n2) spatio-temporal correlation matrix."""
        S1, T1 = X1[:, : self.d_space], X1[:, self.d_space]
        S2, T2 = X2[:, : self.d_space], X2[:, self.d_space]
        return self._kb(
            S1, S2, T1, T2,
            phi          = float(params["phi"]),
            nu           = float(params.get("nu", 1.5)),
            tail         = float(params.get("tail", 0.5)),
            lam0         = float(params.get("lam0", 0.0)),
            theta0       = float(params.get("theta0", 0.0)),
            rho          = float(params.get("rho", 1.0)),
            lambda1      = float(params.get("lambda1", 1.0)),
            lambda2      = float(params.get("lambda2", 1.0)),
            theta_Lambda = float(params.get("theta_Lambda", 0.0)),
        )

    # ── Summary: skip fixed Lagrangian params for frozen models ──────────────

    def summary(self, print_result: bool = True) -> dict:
        if self.params_ is None:
            print(f"{self.__class__.__name__} — not fitted yet.")
            return {}

        info = super().summary(print_result=False)

        if print_result:
            frozen_tag = " [frozen field]" if self.st_model in _FROZEN_MODELS else ""
            title = f"SpatioTemporalGP ({self.st_model}{frozen_tag}) — fit summary"
            print(title)
            print("─" * max(len(title), 50))

            _lagrangian_only = {"rho", "lambda1", "lambda2", "theta_Lambda"}
            skip = _lagrangian_only if self.st_model in _FROZEN_MODELS else set()

            from gpbayeskit.parameters import Parameter as _P
            param_docs: dict[str, str] = {}
            for klass in type(self).__mro__:
                for name, attr in vars(klass).items():
                    if isinstance(attr, _P) and name not in param_docs:
                        param_docs[name] = attr.doc

            for name, val in self.params_.items():
                if name in skip:
                    continue
                doc   = param_docs.get(name, "")
                label = f"  {name:<22}"
                vstr  = f"{val:.6g}" if not isinstance(val, np.ndarray) else (
                    "[" + ", ".join(f"{v:.6g}" for v in val) + "]"
                )
                suffix = f"  # {doc}" if doc else ""
                print(f"{label}: {vstr}{suffix}")

            print(f"  {'sigma2_hat':<22}: {self.sigma2_hat_:.6g}")
            print(f"  {'loglik':<22}: {self.loglik_:.6g}")

        return info