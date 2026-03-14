"""
gpbayeskit.models
=================
GP model classes.

Spatial
-------
  BaseGP          Abstract base class (GLS mean, REML fit, kriging predict).
  SpatialGP       Spatial GP with composable kernel family and anisotropy form.

Spatio-temporal
---------------
  SpatioTemporalGP  Lagrangian / frozen-field spatio-temporal GP.
"""

from gpbayeskit.models.base import BaseGP
from gpbayeskit.models.spatial import SpatialGP
from gpbayeskit.models.spatiotemporal import SpatioTemporalGP

__all__ = ["BaseGP", "SpatialGP", "SpatioTemporalGP"]