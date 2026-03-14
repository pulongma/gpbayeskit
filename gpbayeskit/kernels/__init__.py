"""
gpbayeskit.kernels
==================
Kernel builders and scalar correlation functions.

Builders
--------
  KernelBuilder           Spatial GP kernels (6 families × 3 anisotropy forms).
  LagrangianKernelBuilder Spatio-temporal Lagrangian kernels (4 models).

Scalar kernel functions
-----------------------
  matern_kernel   M(r; nu)
  ch_kernel       CH(r; nu, alpha, beta)

Parametrisation helpers
-----------------------
  lam_vec_from_polar   Build advection vector lambda from polar form.
  build_Lambda         Build anisotropy matrix Lambda.
  rotation_matrix      2-D rotation matrix.

Low-level Lagrangian functions
------------------------------
  frozen_matern, frozen_ch
  lagrangian_matern, lagrangian_ch
  frozen_matern_matrix, frozen_ch_matrix
  lagrangian_matern_matrix, lagrangian_ch_matrix
"""

from gpbayeskit.kernels._spatial import KernelBuilder
from gpbayeskit.kernels._lagrangian import LagrangianKernelBuilder

from gpbayeskit.kernels._functions import matern_kernel, ch_kernel
from gpbayeskit.kernels._parametrisation import (
    lam_vec_from_polar,
    build_Lambda,
    rotation_matrix,
)
from gpbayeskit.kernels._lagrangian_fns import (
    frozen_matern, frozen_ch,
    lagrangian_matern, lagrangian_ch,
    frozen_matern_matrix, frozen_ch_matrix,
    lagrangian_matern_matrix, lagrangian_ch_matrix,
)

__all__ = [
    "KernelBuilder", "LagrangianKernelBuilder",
    "matern_kernel", "ch_kernel",
    "lam_vec_from_polar", "build_Lambda", "rotation_matrix",
    "frozen_matern", "frozen_ch", "lagrangian_matern", "lagrangian_ch",
    "frozen_matern_matrix", "frozen_ch_matrix",
    "lagrangian_matern_matrix", "lagrangian_ch_matrix",
]