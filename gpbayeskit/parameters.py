"""
gpbayeskit.parameters
======================
Self-describing parameter system for GP models.

``Parameter``
    Class-level descriptor that declares a scalar or vector parameter:
    its default value, optimisation bounds, and documentation string.
    Subclasses of ``BaseGP`` declare their parameters as class attributes;
    ``BaseGP.__init__`` discovers them via MRO introspection and creates
    per-instance ``_ParamEntry`` objects that hold the mutable runtime state.

``_ParamEntry``
    Per-instance mutable runtime state for one parameter.  Holds the
    current value (updated during optimisation), whether the parameter is
    fixed, and bound information required by ``scipy.optimize.minimize``.

Design rules
------------
* Parameters are discovered in MRO order (most-derived class first) so
  subclass parameters appear before base-class parameters in the packed
  optimisation vector.  Duplicate names (same parameter redeclared in a
  parent) are skipped — the most-derived definition wins.
* Scalar parameters are stored as ``float``; vector parameters as
  ``np.ndarray`` of shape ``(size,)``.
* ``size`` may be the special string ``'d'`` to mean "resolve to the
  model's spatial dimension ``self.d`` at construction time".  This is
  used by ``SpatialGP`` for ARD/tensor phi vectors.
* Fixing a parameter (``_ParamEntry.fixed_to is not None``) excludes it
  from the optimisation vector but keeps it in ``_params_dict()`` so it
  is still forwarded to ``covariance()``.

Examples
--------
Declare a new covariance model with a novel parameter set::

    from gpbayeskit.models.base import BaseGP
    from gpbayeskit.parameters import Parameter

    class PeriodicMatern(BaseGP):
        phi    = Parameter(0.5,  lower=1e-8, doc="Spatial range")
        nu     = Parameter(1.5,  lower=1e-8, doc="Smoothness")
        period = Parameter(1.0,  lower=1e-8, doc="Period")
        nugget = Parameter(1e-6, lower=0.0,  doc="Nugget ratio")

        def covariance(self, X1, X2, **p):
            ...

    # fit automatically optimises phi, nu, period, nugget
    gp = PeriodicMatern(X, y).fit(fix_nu=1.5)
"""

from __future__ import annotations

import numpy as np


# ── Parameter declaration (class-level) ───────────────────────────────────────

class Parameter:
    """
    Declare one optimisable (or fixable) model parameter.

    Parameters
    ----------
    default : float or array_like
        Initial value used before fitting.  For vector parameters, a scalar
        is broadcast to the resolved size.
    lower   : float
        Lower bound for optimisation (default 1e-8, i.e. strictly positive).
    upper   : float or None
        Upper bound for optimisation (default None = unbounded).
    size    : int or ``'d'``
        ``1`` → scalar; ``k`` → vector of length k;
        ``'d'`` → vector of length ``self.d`` (resolved at model init).
    doc     : str
        Human-readable description shown in ``summary()``.
    """

    def __init__(
        self,
        default: float,
        *,
        lower: float = 1e-8,
        upper: float | None = None,
        size: int | str = 1,
        doc: str = "",
    ) -> None:
        self.default = default
        self.lower   = lower
        self.upper   = upper
        self.size    = size     # 1, int, or 'd'
        self.doc     = doc

    def __repr__(self) -> str:
        return (
            f"Parameter(default={self.default}, lower={self.lower}, "
            f"upper={self.upper}, size={self.size!r})"
        )


# ── Per-instance runtime state ─────────────────────────────────────────────────

class _ParamEntry:
    """
    Mutable runtime state for one parameter on a specific model instance.

    Attributes
    ----------
    lower, upper : float / None
        Optimisation bounds.
    value        : float or ndarray
        Current value; updated in-place during optimisation.
    fixed_to     : float / ndarray / None
        If not None, the parameter is fixed and excluded from optimisation.
    is_vector    : bool
        True for vector parameters (size > 1).
    """

    __slots__ = ("lower", "upper", "value", "fixed_to", "is_vector")

    def __init__(
        self,
        default: float,
        lower: float,
        upper: float | None,
        size: int,              # already resolved (no 'd' strings here)
    ) -> None:
        self.lower    = lower
        self.upper    = upper
        self.fixed_to = None

        if size == 1:
            self.is_vector = False
            self.value     = float(np.asarray(default).ravel()[0])
        else:
            self.is_vector = True
            dv = np.asarray(default, dtype=float).ravel()
            self.value = (
                np.full(size, dv[0])    # broadcast scalar
                if dv.size == 1
                else dv[:size].copy()
            )

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def is_free(self) -> bool:
        """True if this parameter participates in optimisation."""
        return self.fixed_to is None

    @property
    def current(self) -> float | np.ndarray:
        """The effective value: fixed_to if fixed, else the mutable value."""
        return self.fixed_to if self.fixed_to is not None else self.value

    @property
    def n_elements(self) -> int:
        """Number of scalar slots this parameter occupies in the theta vector."""
        return len(self.value) if self.is_vector else 1

    def bounds_list(self) -> list[tuple]:
        """Return ``[(lower, upper)] * n_elements`` for scipy.optimize."""
        return [(self.lower, self.upper)] * self.n_elements

    # ── Mutation helpers (used by BaseGP) ─────────────────────────────────────

    def set_value(self, val: float | np.ndarray) -> None:
        """Set the mutable value from a scalar or array."""
        if self.is_vector:
            v = np.asarray(val, dtype=float).ravel()
            self.value[:] = v if len(v) == len(self.value) else np.full(len(self.value), v[0])
        else:
            self.value = float(np.asarray(val).ravel()[0])

    def fix(self, val: float | np.ndarray) -> None:
        """Fix this parameter to *val* (exclude from optimisation)."""
        if self.is_vector:
            v = np.asarray(val, dtype=float).ravel()
            self.fixed_to = (
                np.full(len(self.value), v[0]) if v.size == 1 else v.copy()
            )
        else:
            self.fixed_to = float(np.asarray(val).ravel()[0])

    def unfix(self) -> None:
        """Release a previously fixed parameter back to free."""
        self.fixed_to = None

    def __repr__(self) -> str:
        status = f"fixed={self.fixed_to}" if not self.is_free else f"value={self.current}"
        return f"_ParamEntry({status}, bounds=({self.lower}, {self.upper}))"
