"""
gpbayeskit — end-to-end demo
=============================

Covers:
  1. Simulation         — draw a GP realisation to use as synthetic data
  2. Model construction — spatial and spatio-temporal GP objects
  3. Model fitting      — REML optimisation with parameter fixing / init
  4. Prediction         — kriging mean and variance at new locations
  5. Scoring            — evaluate against held-out data
  6. Plotting           — likelihood profiles and space-time covariance contours
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; remove if running interactively
import matplotlib.pyplot as plt

import gpbayeskit as gp
from gpbayeskit.plotting import (
    plot_loglik,
    plot_loglik_panel,
    plot_cov_st_1d,
    plot_cov_st_2d,
)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Reproducible RNG
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1 — Simulation")
print("=" * 60)

# 1a. Simulate from a spatial GP on a 2-D domain
X_all = rng.uniform(0, 1, (120, 2))

sim = gp.simulate(
    X_all,
    kernel_type  = "matern52",     # Matérn ν = 2.5
    kernel_form  = "isotropic",
    phi          = 0.25,           # spatial range
    sigma2       = 1.5,            # marginal variance
    nugget       = 0.05,           # 5 % noise
    return_latent= True,           # also store the noise-free draw
    seed         = 42,
)

print(f"SimulationResult.y      shape : {sim.y.shape}")
print(f"SimulationResult.latent shape : {sim.latent.shape}")
print(f"SimulationResult.cov    shape : {sim.cov.shape}")

# Train / test split
idx_train = rng.choice(120, 80, replace=False)
idx_test  = np.setdiff1d(np.arange(120), idx_train)

X_train, y_train = X_all[idx_train], sim.y[idx_train]
X_test,  y_test  = X_all[idx_test],  sim.y[idx_test]

print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test  size : {X_test.shape[0]}")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — MODEL CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2 — Model construction")
print("=" * 60)

# 2a. Isotropic Matérn GP (unfitted)
gp_iso = gp.SpatialGP(X_train, y_train,
                      kernel_type="matern52",
                      kernel_form="isotropic")
print("\nUnfitted spatial GP:")
print(" ", gp_iso)

# 2b. ARD Matérn (one length-scale per spatial dimension)
gp_ard = gp.SpatialGP(X_train, y_train,
                      kernel_type="matern",
                      kernel_form="ard")
print("\nUnfitted ARD GP:")
print(" ", gp_ard)

# 2c. CH (confluent hypergeometric) kernel — has a free tail parameter
gp_ch = gp.SpatialGP(X_train, y_train,
                     kernel_type="ch")
print("\nUnfitted CH GP:")
print(" ", gp_ch)

# 2d. Spatio-temporal GP (Lagrangian Matérn)
# X must have shape (n, d_space + 1): last column is time
t_all = rng.uniform(0, 2, (120, 1))
Xst_all = np.hstack([X_all, t_all])
Xst_train, y_st_train = Xst_all[idx_train], y_train
Xst_test,  y_st_test  = Xst_all[idx_test],  y_test

gp_st = gp.SpatioTemporalGP(Xst_train, y_st_train,
                              model="lagrangian_matern",
                              d_space=2)
print("\nUnfitted spatio-temporal GP:")
print(" ", gp_st)


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — MODEL FITTING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3 — Model fitting")
print("=" * 60)

# 3a. Simple fit — all free parameters optimised from defaults
gp_iso.fit()
print("\nFitted isotropic GP:")
print(" ", gp_iso)
gp_iso.summary()

# 3b. Fix smoothness ν, set a custom starting value for phi
gp_ard.fit(fix_nu=1.5, phi_init=0.3)
print("\nFitted ARD GP (ν fixed to 1.5):")
print(" ", gp_ard)
gp_ard.summary()

# 3c. Fix nugget to a small constant; let tail be free
gp_ch.fit(fix_nugget=0.01, fix_nu=1.5)
print("\nFitted CH GP (nugget fixed, tail free):")
print(" ", gp_ch)
gp_ch.summary()

# 3d. Spatio-temporal fit — fix ν and nugget to reduce runtime
gp_st.fit(fix_nu=1.5, fix_nugget=0.01)
print("\nFitted spatio-temporal GP:")
print(" ", gp_st)
gp_st.summary()


# ═════════════════════════════════════════════════════════════════════════════
# PART 4 — PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4 — Prediction")
print("=" * 60)

# 4a. Kriging mean + variance at test locations
mean_iso, var_iso = gp_iso.predict(X_test)
print(f"\nSpatialGP predictions:  mean shape={mean_iso.shape}, var shape={var_iso.shape}")

# 4b. Prediction without variance (faster)
mean_only = gp_iso.predict(X_test, return_var=False)
print(f"Mean-only prediction:   shape={mean_only.shape}")

# 4c. 95 % credible intervals
sd_iso = np.sqrt(var_iso)
ci_lo  = mean_iso - 1.96 * sd_iso
ci_hi  = mean_iso + 1.96 * sd_iso
coverage = np.mean((y_test >= ci_lo) & (y_test <= ci_hi))
print(f"Empirical 95 % CI coverage: {coverage:.1%}")

# 4d. Spatio-temporal prediction
mean_st, var_st = gp_st.predict(Xst_test)
print(f"\nSpatioTemporalGP predictions: mean shape={mean_st.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# PART 5 — SCORING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 5 — Scoring")
print("=" * 60)

scores = gp_iso.score(X_test, y_test)
print("\nIsotropic GP test scores:")
for k, v in scores.items():
    print(f"  {k:<15} {v:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# PART 6 — PLOTTING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 6 — Plotting")
print("=" * 60)

# ── 6a. Single likelihood profile ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
plot_loglik(
    X_train, y_train,
    param        = "phi",
    param_range  = (0.05, 1.0),
    fixed_params = {"nu": 2.5, "nugget": 0.05},
    kernel_type  = "matern52",
    log_x        = True,
    ax           = ax,
)
ax.set_title(r"Log-likelihood profile: $\phi$  (Matérn 5/2)")
fig.tight_layout()
fig.savefig("demo_loglik_phi.png", dpi=120)
print("\nSaved: demo_loglik_phi.png")

# ── 6b. Panel of likelihood profiles ─────────────────────────────────────────
specs = [
    {
        "param"       : "phi",
        "param_range" : (0.05, 1.0),
        "fixed_params": {"nu": 2.5, "nugget": 0.05},
        "log_x"       : True,
    },
    {
        "param"       : "nugget",
        "param_range" : (1e-4, 0.3),
        "fixed_params": {"phi": 0.25, "nu": 2.5},
        "log_x"       : True,
    },
    {
        "param"       : "nu",
        "param_range" : (0.5, 4.0),
        "fixed_params": {"phi": 0.25, "nugget": 0.05},
    },
]
fig, axes = plot_loglik_panel(
    X_train, y_train,
    specs       = specs,
    kernel_type = "matern",
    ncols       = 3,
)
fig.suptitle("Integrated log-likelihood profiles", fontsize=13)
fig.savefig("demo_loglik_panel.png", dpi=120)
print("Saved: demo_loglik_panel.png")

# ── 6c. Space-time covariance — 1-D spatial domain ───────────────────────────
params_1d = dict(
    phi          = 0.4,
    nu           = 1.5,
    lam0         = 0.6,    # advection magnitude
    theta0       = 0.0,    # advection direction (radians, along h₁)
    rho          = 0.8,
    lambda1      = 0.8,
    lambda2      = 0.3,
    theta_Lambda = 0.4,
)
fig, ax = plot_cov_st_1d(
    params_1d,
    h_range = (-3, 3),
    u_range = (0, 3),
    model   = "lagrangian_matern",
)
fig.savefig("demo_cov_st_1d.png", dpi=120)
print("Saved: demo_cov_st_1d.png")

# ── 6d. Space-time covariance — 2-D spatial domain panel ─────────────────────
fig, axes = plot_cov_st_2d(
    params_1d,
    h_range  = (-2, 2),
    u_values = [0, 0.5, 1.0, 2.0],
    model    = "lagrangian_matern",
    ncols    = 4,
    n_h      = 60,
)
fig.savefig("demo_cov_st_2d.png", dpi=120)
print("Saved: demo_cov_st_2d.png")

# ── 6e. Covariance panels from a *fitted* SpatioTemporalGP ───────────────────
fig, axes = plot_cov_st_2d(
    gp_st,              # pass the fitted model directly
    h_range  = (-1.5, 1.5),
    u_values = [0, 0.5, 1.0],
    ncols    = 3,
    n_h      = 60,
)
fig.savefig("demo_cov_st_2d_fitted.png", dpi=120)
print("Saved: demo_cov_st_2d_fitted.png")

plt.close("all")
print("\nDemo complete.")