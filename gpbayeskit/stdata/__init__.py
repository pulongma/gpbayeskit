"""
gpbayeskit.stdata
=================
Space-time data containers for real-world spatio-temporal data sets.

Inspired by R's *spacetime* package (Pebesma 2012), four classes cover the
most common data structures encountered in practice:

STF — Space-Time Full grid
    All n_loc × n_t combinations are present.  The data tensor has shape
    (n_t, n_loc, n_var).  Typical use: gridded model output, reanalysis,
    simulation results.

    Key methods:
        to_array()         → (n_t, n_loc, n_var) NumPy array
        to_time_wide()     → (n_loc × n_t) wide DataFrame (loc rows, time cols)
        to_space_wide()    → (n_t × n_loc) wide DataFrame (time rows, loc cols)
        at_time(t)         → spatial snapshot at one time
        at_location(i)     → time series at one location
        subset_time(...)   → new STF for a time window
        subset_locations() → new STF for a subset of sites
        to_long()          → tidy long DataFrame
        to_sti()           → convert to STI
        to_sts(mask)       → convert to STS (with optional missing mask)
        from_simulation()  → build from STSimulationResult

STS — Space-Time Sparse grid
    Same declared lattice as STF, but only a subset of (location, time)
    pairs is observed (rest are missing).  Typical use: monitoring networks
    with gaps, irregular satellite revisit times.

    Key methods:
        missing_mask       → (n_t, n_loc) bool array (True = missing)
        observed_fraction  → scalar
        to_masked_array()  → NumPy masked array (n_t, n_loc)
        to_time_wide()     → wide DataFrame with NaN for missing
        to_stf(fill)       → upgrade to full grid
        to_sti()           → drop lattice structure
        subset_time(...)   → new STS for a time window
        from_long()        → build from long DataFrame

STI — Space-Time Irregular
    Each observation has its own arbitrary (s, t).  No lattice assumed.
    The most flexible format and the natural input for SpatioTemporalGP.

    Key methods:
        to_model_arrays()      → (X, y) for SpatioTemporalGP
        to_stf(grid_n, t_bins) → aggregate onto a regular grid
        subset_time(lo, hi)    → time window
        subset_bbox(bbox)      → spatial bounding box
        train_test_split()     → random train/test split
        sort_by_time()         → sorted copy
        add_variable(name, v)  → add a column
        apply(func)            → apply a function to data
        to_stt(ids)            → assign trajectory IDs
        from_long()            → build from long DataFrame
        from_simulation()      → build from STSimulationResult

STT — Space-Time Trajectories
    Ordered (s, t) paths partitioned into trajectories by an ID column.
    Typical use: animal movement, vehicle GPS, Lagrangian floats.

    Key methods:
        get_trajectory(id)     → one trajectory as STI
        iter_trajectories()    → yield (id, STI) pairs
        subset_trajectories()  → keep named trajectories
        bounding_boxes()       → spatial/temporal bounds per trajectory
        arc_lengths()          → total path length per trajectory
        to_sti()               → flatten to STI
        from_long()            → build from long DataFrame with id column
        from_sti_list()        → concatenate a list of STI objects

Shared interface (all classes)
-------------------------------
    to_long()     → tidy long DataFrame  [coord_names..., time_name, vars...]
    summary()     → print + return summary dict
    n_obs         → number of observations
    d_space       → number of spatial dimensions
    n_var         → number of data variables
    t_min, t_max  → temporal extent
"""

from gpbayeskit.stdata._stf import STF
from gpbayeskit.stdata._sts import STS
from gpbayeskit.stdata._sti import STI
from gpbayeskit.stdata._stt import STT

__all__ = ["STF", "STS", "STI", "STT"]
