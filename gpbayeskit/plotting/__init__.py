"""
gpbayeskit.plotting
=========================
Plotting utilities for spatial and spatio-temporal GP models.

Functions
---------
Likelihood profiles  (spatial GP)          — gpbayeskit.plotting._likelihood
    plot_loglik        1-D profile of the integrated log-likelihood vs. one parameter.
    plot_loglik_panel  Grid of profile plots from a list of specs.

Space-time covariance surfaces             — gpbayeskit.plotting._contours
    plot_cov_st_1d     Filled contour of C(h, u) for a 1-D spatial domain.
                       x-axis = spatial lag h, y-axis = temporal lag u.
    plot_cov_st_2d     Panel of C(h₁, h₂; u) contours at specified temporal lags.
                       One subplot per u value; white arrow shows advection u·λ.

Space-time GP realisations                — gpbayeskit.plotting._realizations
    plot_st_realizations  Panel of spatial heatmaps across time with advection tracer.

All functions return (Figure, Axes) and never call plt.show().

Requirements
------------
matplotlib is an optional dependency.  Install with::

    pip install gpbayeskit[plot]
"""

from gpbayeskit.plotting._likelihood import plot_loglik, plot_loglik_panel
from gpbayeskit.plotting._contours   import plot_cov_st_1d, plot_cov_st_2d
from gpbayeskit.plotting._realizations import plot_st_realizations

__all__ = [
    "plot_loglik",
    "plot_loglik_panel",
    "plot_cov_st_1d",
    "plot_cov_st_2d",
    "plot_st_realizations",
]