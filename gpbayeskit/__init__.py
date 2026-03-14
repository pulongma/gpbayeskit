"""
gpbayeskit
==========
Spatio-temporal modelling with Gaussian processes.
        
Sub-packages
------------
kernels
    KernelBuilder (spatial) and LagrangianKernelBuilder (spatio-temporal),
    plus scalar kernel functions and parametrisation helpers.
models
    BaseGP, SpatialGP, SpatioTemporalGP.
plotting
    Likelihood profile plots (plot_loglik, plot_loglik_panel) and
    space-time covariance contours (plot_cov_st_1d, plot_cov_st_2d).
utils
    Input validation, distance helpers (euclidean, great-circle, chordal),
    and GP scoring metrics.
"""
 
__version__ = "0.1.0"
 
from gpbayeskit.models.spatial        import SpatialGP
from gpbayeskit.models.spatiotemporal import SpatioTemporalGP
from gpbayeskit.kernels               import KernelBuilder, LagrangianKernelBuilder
from gpbayeskit.simulation import simulate, SimulationResult
from gpbayeskit.utils                 import (
    gp_scores,
    euclidean_dist,
    great_circle_dist,
    chordal_dist,
)
 
__all__ = [
    # Models
    "SpatialGP",
    "SpatioTemporalGP",
    # Kernel builders
    "KernelBuilder",
    "LagrangianKernelBuilder",
    # Simulation
    "simulate",
    "SimulationResult",
    # Utilities
    "gp_scores",
    "euclidean_dist",
    "great_circle_dist",
    "chordal_dist",
]
 

