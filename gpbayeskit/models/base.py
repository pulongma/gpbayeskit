"""
gpbayeskit.models.base
=======================
Abstract base class for all gpbayeskit GP models.

Design
------
- GLS mean structure:   E[y] = H β,  β estimated by REML/MLE.
- Covariance structure: Cov[y] = σ² (R(θ) + nugget · I).
- Parameters are **self-describing**: subclasses declare them as
  ``Parameter`` class attributes; ``BaseGP`` discovers, packs, and
  unpacks them automatically.  Subclasses only need to implement
  ``covariance()``.

Parameter discovery
-------------------
``BaseGP.__init__`` walks ``type(self).__mro__`` from most-derived to
least-derived, collecting every ``Parameter`` class attribute into a
per-instance ``_param_store`` (``dict[str, _ParamEntry]``).  Duplicate
names are skipped, so the most-derived definition always wins.

Fit interface
-------------
``fit(**kwargs)`` accepts two kinds of keyword arguments, discovered
dynamically from the parameter store:

    fix_<n>=value   → fix parameter <n> to *value*
    <n>_init=value  → set initial value before optimisation

Any combination of these may be passed; parameters with neither keyword
are optimised starting from their declared ``default``.

Adding a new model
------------------
The *only* two things a subclass must provide::

    class MyModel(BaseGP):
        phi    = Parameter(0.5, lower=1e-8, doc="Range")
        nu     = Parameter(1.5, lower=1e-8, doc="Smoothness")
        nugget = Parameter(1e-6, lower=0.0, doc="Nugget")

        def covariance(self, X1, X2, **params):
            ...   # use params['phi'], params['nu'], etc.

Everything else (fit, predict, summary, scoring) is inherited.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from gpbayeskit.utils import validate_Xy, validate_H, gp_scores
from gpbayeskit.parameters import Parameter, _ParamEntry


class BaseGP:
    """
    Abstract base Gaussian process model with GLS mean and self-describing
    parameters.

    Subclasses **must** implement
    --------------------------------
    covariance(X1, X2, **params) -> ndarray (n1, n2)

    Subclasses **may** override
    ---------------------------
    _setup_params()
        Post-construction hook to resize vector parameters or fix
        structurally absent parameters.
    """

    nugget = Parameter(1e-6, lower=0.0, doc="Nugget (noise variance ratio)")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, X, y, H=None):
        self.X, self.y = validate_Xy(X, y)
        self.n, self.d = self.X.shape
        self.H         = validate_H(H, self.n)

        self._build_param_store()
        self._setup_params()

        self.params_:       dict | None       = None
        self.loglik_:       float | None      = None
        self.beta_hat_:     np.ndarray | None = None
        self.sigma2_hat_:   float | None      = None
        self.R_:            np.ndarray | None = None
        self.L_:            np.ndarray | None = None
        self.HRH_inv_:      np.ndarray | None = None
        self.optim_result_: object | None     = None

    def __repr__(self):
        status = "fitted" if self.params_ is not None else "unfitted"
        return f"{self.__class__.__name__}(n={self.n}, d={self.d}, status={status})"

    # ------------------------------------------------------------------
    # Parameter store
    # ------------------------------------------------------------------

    def _build_param_store(self):
        """Collect Parameter class attrs in MRO order into _ParamEntry objects."""
        store: dict[str, _ParamEntry] = {}
        seen: set[str] = set()
        for klass in type(self).__mro__:
            for name, attr in vars(klass).items():
                if isinstance(attr, Parameter) and name not in seen:
                    seen.add(name)
                    size = self.d if attr.size == 'd' else int(attr.size)
                    store[name] = _ParamEntry(
                        default=attr.default,
                        lower=attr.lower,
                        upper=attr.upper,
                        size=size,
                    )
        self._param_store: dict[str, _ParamEntry] = store

    def _setup_params(self):
        """Post-construction hook. Override in subclasses."""

    def _resize_param(self, name: str, new_size: int):
        """Replace a scalar _ParamEntry with a vector of length new_size."""
        old = self._param_store[name]
        self._param_store[name] = _ParamEntry(
            default=old.value,
            lower=old.lower,
            upper=old.upper,
            size=new_size,
        )

    def _params_dict(self) -> dict:
        return {name: entry.current for name, entry in self._param_store.items()}

    def _pack(self):
        values, bounds = [], []
        for entry in self._param_store.values():
            if entry.is_free:
                v = entry.value
                if entry.is_vector:
                    values.extend(v.tolist())
                else:
                    values.append(float(v))
                bounds.extend(entry.bounds_list())
        return np.array(values, dtype=float), bounds

    def _unpack(self, theta: np.ndarray):
        idx = 0
        for entry in self._param_store.values():
            if entry.is_free:
                n = entry.n_elements
                entry.set_value(theta[idx : idx + n])
                idx += n

    # ------------------------------------------------------------------
    # Abstract covariance
    # ------------------------------------------------------------------

    def covariance(self, X1, X2, **params):
        """Return (n1, n2) correlation matrix. Must be overridden."""
        raise NotImplementedError("Subclasses must implement covariance().")

    # ------------------------------------------------------------------
    # Linear algebra helpers
    # ------------------------------------------------------------------

    def _solve_R(self, B):
        if self.L_ is None:
            raise RuntimeError("Call fit() before _solve_R().")
        z = solve_triangular(self.L_, B, lower=True)
        return solve_triangular(self.L_.T, z, lower=False)

    @staticmethod
    def _safe_cholesky(M):
        try:
            return cholesky(M, lower=True)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "Correlation matrix is not positive definite."
            ) from exc

    @staticmethod
    def _robust_cholesky(M):
        """Cholesky with exponential jitter retry for borderline-PD matrices."""
        n = M.shape[0]
        jitter = 0.0
        for _ in range(12):
            try:
                return cholesky(M + jitter * np.eye(n), lower=True)
            except np.linalg.LinAlgError:
                jitter = max(jitter, 1e-10) * 10.0
        raise np.linalg.LinAlgError(
            f"Could not obtain positive-definite Cholesky with jitter up to {jitter:.1e}."
        )

    # ------------------------------------------------------------------
    # Log marginal likelihood (REML-style, generic)
    # ------------------------------------------------------------------

    def log_marginal_likelihood(self, **params) -> float:
        """
        Integrated REML log marginal likelihood.

            ℓ = -½ log|R|  -  ½ log|HᵀR⁻¹H|  -  (n-p)/2 · log(S²)

        ``nugget`` is extracted from *params* to form R + nugget·I; the
        full *params* dict (including nugget) is forwarded to covariance()
        so subclasses can ignore what they don't need.
        """
        H, y = self.H, self.y
        n, p = H.shape
        if n <= p:
            return -np.inf

        nugget = float(params.get("nugget", 0.0))

        try:
            R = self.covariance(self.X, self.X, **params)
            R = 0.5 * (R + R.T)
            R = R + nugget * np.eye(n)
            L = self._robust_cholesky(R)
        except np.linalg.LinAlgError:
            return -np.inf

        logdet_R   = 2.0 * np.sum(np.log(np.diag(L)))
        LH         = solve_triangular(L, H, lower=True)
        HRH        = LH.T @ LH

        try:
            LH2        = cholesky(HRH, lower=True)
            logdet_HRH = 2.0 * np.sum(np.log(np.diag(LH2)))
            HRH_inv    = np.linalg.solve(HRH, np.eye(p))
        except np.linalg.LinAlgError:
            return -np.inf

        Ly      = solve_triangular(L, y, lower=True)
        Rinv_y  = solve_triangular(L.T, Ly, lower=False)
        HRinv_y = H.T @ Rinv_y
        S2      = float((y.T @ Rinv_y).squeeze()) - float(
            (HRinv_y.T @ HRH_inv @ HRinv_y).squeeze()
        )

        if not np.isfinite(S2) or S2 <= 0:
            return -np.inf

        return float(-0.5 * logdet_R - 0.5 * logdet_HRH - 0.5 * (n - p) * np.log(S2))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, **kwargs) -> "BaseGP":
        """
        Fit by maximising the log marginal likelihood.

        Keyword arguments
        -----------------
        fix_<name>=value    Fix parameter to value (excluded from optim).
        <name>_init=value   Set initial value of free parameter.
        method=str          scipy optimiser (default 'L-BFGS-B').
        """
        method      = kwargs.pop("method", "L-BFGS-B")
        param_names = set(self._param_store.keys())

        for key, val in kwargs.items():
            if key.startswith("fix_"):
                name = key[4:]
                if name in param_names:
                    self._param_store[name].fix(val)
            elif key.endswith("_init"):
                name = key[:-5]
                if name in param_names and self._param_store[name].is_free:
                    self._param_store[name].set_value(val)

        theta0, bounds = self._pack()

        def _obj(theta):
            self._unpack(theta)
            val = self.log_marginal_likelihood(**self._params_dict())
            return -val if np.isfinite(val) else 1e20

        opt = minimize(_obj, theta0, method=method, bounds=bounds)
        self._unpack(opt.x)
        self.params_       = self._params_dict()
        self.loglik_       = self.log_marginal_likelihood(**self.params_)
        self.optim_result_ = opt
        self._post_fit_cache()
        return self

    # ------------------------------------------------------------------
    # Post-fit caching
    # ------------------------------------------------------------------

    def _post_fit_cache(self):
        if self.params_ is None:
            raise RuntimeError("Call fit() before _post_fit_cache().")

        H, y   = self.H, self.y
        n, p   = H.shape
        params = self.params_
        nugget = float(params.get("nugget", 0.0))

        R = self.covariance(self.X, self.X, **params)
        R = 0.5 * (R + R.T)
        R = R + nugget * np.eye(n)
        L = self._robust_cholesky(R)

        Ly       = solve_triangular(L, y, lower=True)
        LH       = solve_triangular(L, H, lower=True)
        HtH      = LH.T @ LH
        HtH_inv  = np.linalg.solve(HtH, np.eye(p))
        beta_hat = HtH_inv @ (LH.T @ Ly)

        resid_white = Ly - LH @ beta_hat
        S2          = float((resid_white.T @ resid_white).item())

        self.R_          = R
        self.L_          = L
        self.beta_hat_   = beta_hat
        self.sigma2_hat_ = S2 / (n - p)
        self.HRH_inv_    = HtH_inv

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, Xnew, Hnew=None, return_var=True):
        """
        Kriging prediction at new locations.

        Parameters
        ----------
        Xnew       : (m, d)
        Hnew       : (m, p) or None
        return_var : bool

        Returns
        -------
        mean : (m, 1)
        var  : (m, 1)  — only when return_var=True
        """
        if self.params_ is None:
            raise RuntimeError("Call fit() before predict().")

        Xnew = np.asarray(Xnew, dtype=float)
        if Xnew.ndim != 2 or Xnew.shape[1] != self.d:
            raise ValueError(f"Xnew must have shape (m, {self.d}).")

        m = Xnew.shape[0]
        if Hnew is None:
            if self.H.shape[1] == 1:
                Hnew = np.ones((m, 1))
            else:
                raise ValueError("Hnew required when H is not intercept-only.")
        else:
            Hnew = np.asarray(Hnew, dtype=float)

        params  = self.params_
        R0      = self.covariance(self.X, Xnew, **params)          # (n, m)
        resid   = self.y - self.H @ self.beta_hat_
        mean    = Hnew @ self.beta_hat_ + R0.T @ self._solve_R(resid)

        if not return_var:
            return mean

        Rinv_R0 = self._solve_R(R0)
        R00     = self.covariance(Xnew, Xnew, **params)
        cov     = R00 - R0.T @ Rinv_R0
        temp    = Hnew - R0.T @ self._solve_R(self.H)
        cov    += temp @ self.HRH_inv_ @ temp.T
        cov     = self.sigma2_hat_ * cov
        var     = np.clip(np.diag(cov), 0.0, None).reshape(-1, 1)

        return mean, var

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, Xnew, y_true, Hnew=None):
        """Predict at Xnew and compute evaluation metrics against y_true."""
        mean, var = self.predict(Xnew, Hnew=Hnew, return_var=True)
        return gp_scores(y_true, mean.ravel(), var.ravel())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, print_result=True) -> dict:
        """Print and return all fitted parameters and diagnostics."""
        if self.params_ is None:
            print(f"{self.__class__.__name__} — not fitted yet.")
            return {}

        param_docs: dict[str, str] = {}
        for klass in type(self).__mro__:
            for name, attr in vars(klass).items():
                if isinstance(attr, Parameter) and name not in param_docs:
                    param_docs[name] = attr.doc

        info = dict(self.params_)
        info["sigma2_hat"] = self.sigma2_hat_
        info["loglik"]     = self.loglik_

        if print_result:
            title = f"{self.__class__.__name__} — fit summary"
            print(title)
            print("─" * max(len(title), 44))
            for name, val in self.params_.items():
                doc   = param_docs.get(name, "")
                label = f"  {name:<20}"
                if isinstance(val, np.ndarray):
                    vstr = "[" + ", ".join(f"{v:.6g}" for v in val) + "]"
                else:
                    vstr = f"{val:.6g}"
                suffix = f"  # {doc}" if doc else ""
                print(f"{label}: {vstr}{suffix}")
            print(f"  {'sigma2_hat':<20}: {self.sigma2_hat_:.6g}")
            print(f"  {'loglik':<20}: {self.loglik_:.6g}")

        return info
