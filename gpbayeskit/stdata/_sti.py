"""
gpbayeskit.stdata._sti
========================
STI — Space-Time Irregular.

Each observation has its own arbitrary (s, t) pair; there is no implied
lattice.  This is the most general format and the natural representation
for opportunistic sensor readings, occurrence records, or model output
that doesn't align to a grid.

It is the common "input" format for fitting SpatioTemporalGP:
the ``.to_model_arrays()`` method returns the (X, y) pair expected by the
model constructor.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ._base import _STBase


class STI(_STBase):
    """
    Space-Time Irregular: fully arbitrary (location, time) pairs.

    Parameters
    ----------
    coords    : (n_obs, d_space)  Spatial coordinates per observation.
    times     : (n_obs,)          Time stamps per observation.
    data      : array, dict, or DataFrame, shape (n_obs, n_var)
    coord_names, time_name, var_names : as in other ST classes.

    Examples
    --------
    >>> rng    = np.random.default_rng(0)
    >>> coords = rng.uniform(0, 1, (80, 2))
    >>> times  = rng.uniform(0, 5, 80)
    >>> y      = rng.standard_normal(80)
    >>> sti    = STI(coords, times, data=y, var_names=["y"])
    >>> X, y   = sti.to_model_arrays()   # ready for SpatioTemporalGP(X, y)
    >>> X.shape
    (80, 3)
    """

    _type_name = "STI"

    def _validate(self):
        pass   # No structural invariant — any (s, t) pairs are valid.

    # ── Subsetting ────────────────────────────────────────────────────────────

    def subset_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> "STI":
        """Return a new STI keeping only observations in [t_min, t_max]."""
        lo = t_min if t_min is not None else self.t_min
        hi = t_max if t_max is not None else self.t_max
        mask = (self.times >= lo) & (self.times <= hi)
        return STI(
            coords=self.coords[mask],
            times=self.times[mask],
            data=self._data[mask] if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names or None,
        )

    def subset_bbox(
        self,
        bbox: Sequence[tuple[float, float]],
    ) -> "STI":
        """
        Return a new STI keeping observations within a spatial bounding box.

        Parameters
        ----------
        bbox : list of (lo, hi) pairs, one per spatial dimension.
               e.g. [(0, 0.5), (0, 0.5)] for a 2-D box.
        """
        mask = np.ones(self.n_obs, dtype=bool)
        for dim, (lo, hi) in enumerate(bbox):
            mask &= (self.coords[:, dim] >= lo) & (self.coords[:, dim] <= hi)
        return STI(
            coords=self.coords[mask],
            times=self.times[mask],
            data=self._data[mask] if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names or None,
        )

    def subset_mask(self, mask: np.ndarray) -> "STI":
        """Return a new STI keeping rows where *mask* is True (or integer indices)."""
        mask = np.asarray(mask)
        if mask.dtype.kind in ('i', 'u'):       # integer index array → bool mask
            bool_mask = np.zeros(self.n_obs, dtype=bool)
            bool_mask[mask] = True
            mask = bool_mask
        else:
            mask = mask.astype(bool)
        return STI(
            coords=self.coords[mask],
            times=self.times[mask],
            data=self._data[mask] if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names or None,
        )

    # ── Sorting ───────────────────────────────────────────────────────────────

    def sort_by_time(self) -> "STI":
        """Return a new STI sorted by time (ascending)."""
        idx = np.argsort(self.times)
        return self.subset_mask(idx)

    def sort_by_space(self, dim: int = 0) -> "STI":
        """Return a new STI sorted by coordinate *dim* (ascending)."""
        idx = np.argsort(self.coords[:, dim])
        return self.subset_mask(idx)

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply(self, func: Callable, new_var_name: str = "result") -> "STI":
        """
        Apply *func* to the data DataFrame and return a new STI.

        *func* receives the data DataFrame and must return an array-like of
        shape (n_obs,) or (n_obs, k).
        """
        result = func(self._data)
        result = np.asarray(result)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        n_v    = result.shape[1]
        vnames = [new_var_name] if n_v == 1 else [f"{new_var_name}{i}" for i in range(n_v)]
        return STI(
            coords=self.coords, times=self.times,
            data=result,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=vnames,
        )

    def add_variable(self, name: str, values) -> "STI":
        """Return a new STI with an extra variable column."""
        values = np.asarray(values).ravel()
        if len(values) != self.n_obs:
            raise ValueError(f"values length {len(values)} != n_obs {self.n_obs}.")
        new_data = self._data.copy()
        new_data[name] = values
        return STI(
            coords=self.coords, times=self.times,
            data=new_data,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=list(new_data.columns),
        )

    # ── Binning to grid ───────────────────────────────────────────────────────

    def to_stf(
        self,
        grid_n: int | tuple[int, int] = 10,
        t_bins: int = 5,
        agg_func: str | Callable = "mean",
        var: str | int = 0,
    ) -> "STF":  # noqa: F821
        """
        Aggregate irregular observations onto a regular space-time grid.

        Parameters
        ----------
        grid_n  : int or (n_h, n_w)   Spatial grid resolution.
        t_bins  : int                  Number of time bins.
        agg_func: str or callable      Aggregation function ("mean", "median",
                                       "sum", "count", or any numpy ufunc).
        var     : str or int           Variable to aggregate.
        """
        from ._stf import STF

        col = self.var_names[var] if isinstance(var, int) else var
        d   = self.d_space

        if self.n_obs == 0:
            raise ValueError("Cannot aggregate an empty STI.")

        # Build spatial and temporal bin edges
        if np.isscalar(grid_n):
            ns = (int(grid_n),) * min(d, 2)
        else:
            ns = (int(grid_n[0]), int(grid_n[1]))

        t_edges = np.linspace(self.t_min, self.t_max, t_bins + 1)
        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

        s_edges   = [
            np.linspace(self.coords[:, i].min(), self.coords[:, i].max(), ns[i] + 1)
            for i in range(min(d, 2))
        ]
        s_centers = [0.5 * (e[:-1] + e[1:]) for e in s_edges]

        # Assign each obs to a bin
        t_bin_idx = np.digitize(self.times, t_edges[1:-1])  # 0 to t_bins-1
        s_bin_idx = [
            np.digitize(self.coords[:, i], s_edges[i][1:-1])
            for i in range(len(s_centers))
        ]

        # Build output array
        if d == 1:
            out_shape = (t_bins, ns[0])
            keys = zip(t_bin_idx, s_bin_idx[0])
        else:
            out_shape = (t_bins, ns[0], ns[1])
            keys = zip(t_bin_idx, s_bin_idx[0], s_bin_idx[1])

        from collections import defaultdict
        bins: dict = defaultdict(list)
        vals = self._data[col].values
        for key, v in zip(keys, vals):
            bins[key].append(v)

        # Apply aggregation
        _agg = {"mean": np.nanmean, "median": np.nanmedian,
                 "sum": np.nansum,  "count": len}.get(agg_func, agg_func)
        arr = np.full(out_shape, np.nan)
        for key, vlist in bins.items():
            arr[key] = _agg(vlist) if callable(_agg) else np.nan

        # Build grid locations
        if d == 1:
            locs = s_centers[0].reshape(-1, 1)
            field = arr.reshape(t_bins, ns[0])
        else:
            S1m, S2m = np.meshgrid(s_centers[0], s_centers[1])
            locs     = np.column_stack([S1m.ravel(), S2m.ravel()])
            n_loc    = ns[0] * ns[1]
            field    = arr.reshape(t_bins, n_loc)

        return STF(
            locations=locs,
            times=t_centers,
            data=field.ravel().reshape(-1, 1),
            coord_names=self.coord_names[:2] if d >= 2 else self.coord_names[:1],
            time_name=self.time_name,
            var_names=[col],
        )

    # ── Model interface ───────────────────────────────────────────────────────

    def to_model_arrays(
        self, var: str | int = 0, d_space: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(X, y)`` ready for ``SpatioTemporalGP(X, y)``.

        ``X`` is ``(n_obs, d_space + 1)`` with spatial columns first and time
        in the last column.  ``y`` is ``(n_obs, 1)``.

        Parameters
        ----------
        var     : variable to use as the response.
        d_space : number of spatial columns to use (default: all).
        """
        if not self.var_names:
            raise ValueError("No data variables stored in this STI.")
        col  = self.var_names[var] if isinstance(var, int) else var
        d    = d_space if d_space is not None else self.d_space
        S    = self.coords[:, :d]
        T    = self.times.reshape(-1, 1)
        X    = np.hstack([S, T])
        y    = self._data[col].values.reshape(-1, 1)
        return X, y

    # ── Split ─────────────────────────────────────────────────────────────────

    def train_test_split(
        self,
        test_fraction: float = 0.2,
        seed: int | None = None,
        stratify_by_time: bool = False,
    ) -> tuple["STI", "STI"]:
        """
        Randomly split into train and test STI objects.

        Parameters
        ----------
        test_fraction   : float  Fraction of observations for the test set.
        seed            : int or None  Random seed.
        stratify_by_time: bool   If True, sample test points uniformly across
                                 time quantiles to avoid temporal clustering.
        """
        rng = np.random.default_rng(seed)
        n   = self.n_obs

        if stratify_by_time:
            n_test  = max(1, int(n * test_fraction))
            order   = np.argsort(self.times)
            n_bins  = max(1, n_test)
            splits  = np.array_split(order, n_bins)
            test_idx = np.array([rng.choice(sp) for sp in splits[:n_test]])
        else:
            n_test   = max(1, int(n * test_fraction))
            test_idx = rng.choice(n, size=n_test, replace=False)

        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False

        return self.subset_mask(train_mask), self.subset_mask(~train_mask)


    # ── Wide formats (via pandas pivot) ──────────────────────────────────────

    def to_time_wide(
        self,
        var: str | int = 0,
        loc_id: str | None = None,
        agg_func: str = "mean",
        include_coords: bool = True,
    ) -> pd.DataFrame:
        """
        Time-wide format: rows = locations, columns = time points.

        Because STI is irregular, duplicate (location, time) pairs are
        aggregated with *agg_func* (default "mean").  Each unique spatial
        location becomes a row; each unique time stamp becomes a column.

        Parameters
        ----------
        var            : variable to pivot
        loc_id         : optional column in data that serves as row label;
                         if None, locations are identified by rounded
                         coordinate strings.
        agg_func       : aggregation for duplicate (loc, time) pairs.
        include_coords : when loc_id is None, prepend coordinate columns
                         so rows are self-describing (default True).
        """
        col    = self.var_names[var] if isinstance(var, int) else var
        long   = self.to_long()

        if loc_id is not None:
            row_key = loc_id
        else:
            # Build a string key from rounded coordinates
            coord_str = long[self.coord_names].round(8).astype(str).agg(",".join, axis=1)
            long      = long.copy()
            long["_loc_key"] = coord_str
            row_key   = "_loc_key"

        pivot = long.pivot_table(
            index=row_key, columns=self.time_name, values=col,
            aggfunc=agg_func,
        )
        pivot.columns = [f"{self.time_name}={c:.4g}" for c in pivot.columns]
        pivot = pivot.reset_index(drop=(loc_id is None))

        if include_coords and loc_id is None:
            # Attach first-occurrence coordinates for each location key
            long2 = long.drop_duplicates("_loc_key")
            coord_df = long2.set_index("_loc_key")[self.coord_names]
            pivot = pd.concat(
                [coord_df.reset_index(drop=True), pivot.reset_index(drop=True)],
                axis=1,
            )
        return pivot

    def to_space_wide(
        self,
        var: str | int = 0,
        loc_id: str | None = None,
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Space-wide format: rows = time points, columns = locations.

        Because STI is irregular, duplicate (location, time) pairs are
        aggregated with *agg_func*.  Each unique time stamp becomes a row;
        each unique spatial location becomes a column.

        Parameters
        ----------
        var      : variable to pivot
        loc_id   : optional column in data identifying locations; if None,
                   a coordinate string is used as the column name.
        agg_func : aggregation for duplicate (loc, time) pairs.
        """
        col  = self.var_names[var] if isinstance(var, int) else var
        long = self.to_long()

        if loc_id is not None:
            col_key = loc_id
        else:
            coord_str = long[self.coord_names].round(8).astype(str).agg(",".join, axis=1)
            long      = long.copy()
            long["_loc_key"] = coord_str
            col_key   = "_loc_key"

        pivot = long.pivot_table(
            index=self.time_name, columns=col_key, values=col,
            aggfunc=agg_func,
        )
        pivot.index = [f"{self.time_name}={t:.4g}" for t in pivot.index]
        pivot.columns.name = None
        return pivot

    # ── Format conversion ─────────────────────────────────────────────────────

    def to_stt(
        self,
        id_col: str = "id",
        ids: np.ndarray | None = None,
    ) -> "STT":  # noqa: F821
        """
        Convert to STT by assigning a trajectory ID to each observation.

        Parameters
        ----------
        id_col : str           Name of the ID variable in STT.
        ids    : (n_obs,) or None
                 Integer trajectory ID per observation.  If None, each
                 observation becomes its own single-point trajectory.
        """
        from ._stt import STT
        if ids is None:
            ids = np.arange(self.n_obs)
        data_with_id = self._data.copy()
        data_with_id[id_col] = ids
        return STT(
            coords=self.coords, times=self.times,
            data=data_with_id,
            coord_names=self.coord_names,
            time_name=self.time_name,
            id_col=id_col,
        )

    # ── Class method constructors ─────────────────────────────────────────────

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        coord_names: Sequence[str],
        time_name: str = "t",
        var_names: Sequence[str] | None = None,
    ) -> "STI":
        """Build STI directly from a long-format DataFrame."""
        coords = df[list(coord_names)].values
        times  = df[time_name].values
        if var_names is None:
            var_names = [c for c in df.columns
                         if c not in list(coord_names) + [time_name]]
        data = df[list(var_names)].values if var_names else None
        return cls(
            coords=coords, times=times, data=data,
            coord_names=list(coord_names),
            time_name=time_name,
            var_names=list(var_names) if var_names else None,
        )

    @classmethod
    def from_simulation(cls, result) -> "STI":
        """Build STI from an ``STSimulationResult`` (converts to irregular)."""
        n_t, n_h, n_w = result.field.shape
        S1_flat = np.tile(result.S1.ravel(), n_t)
        S2_flat = np.tile(result.S2.ravel(), n_t)
        t_rep   = np.repeat(result.t, n_h * n_w)
        y_flat  = result.field.reshape(-1)
        return cls(
            coords=np.column_stack([S1_flat, S2_flat]),
            times=t_rep,
            data=y_flat,
            coord_names=["s1", "s2"],
            time_name="t",
            var_names=["y"],
        )


__all__ = ["STI"]
