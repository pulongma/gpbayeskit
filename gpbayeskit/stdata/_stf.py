"""
gpbayeskit.stdata._stf
========================
STF — Space-Time Full grid.

All n_loc × n_t combinations are present.  The data array can be thought
of as a 3-D tensor:  (n_t, n_loc, n_var).

Internal long-form ordering is **time-major**:  the first n_loc rows are
all locations at t[0], the next n_loc at t[1], etc.  This matches the
ordering used for covariance matrices in SpatioTemporalGP.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ._base import _STBase, _coerce_coords, _coerce_times


class STF(_STBase):
    """
    Space-Time Full grid: all n_loc × n_t combinations present.

    Parameters
    ----------
    locations : (n_loc, d_space) array_like
        Spatial locations — one per unique site.
    times     : (n_t,) array_like
        Unique time points.
    data      : array_like, dict, or DataFrame, shape (n_loc * n_t, n_var)
                Observations in **time-major** order (all locations at t₀,
                then all at t₁, …).  May also be an (n_t, n_loc, n_var)
                array — the constructor reshapes it automatically.
                ``None`` → attribute-free container.
    coord_names : list[str] or None
    time_name   : str
    var_names   : list[str] or None

    Attributes
    ----------
    locations : (n_loc, d_space)  Unique spatial locations.
    unique_times : (n_t,)         Unique time points.
    n_loc, n_t   : int            Grid dimensions.

    Examples
    --------
    >>> locs = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    >>> t    = np.array([0.0, 0.5, 1.0])
    >>> y    = np.random.randn(4 * 3)           # time-major
    >>> stf  = STF(locs, t, data=y, var_names=["y"])
    >>> stf.to_array().shape                    # (n_t, n_loc, n_var)
    (3, 4, 1)
    >>> stf.to_time_wide().shape                # (n_loc, n_t)
    (4, 3)
    """

    _type_name = "STF"

    def __init__(
        self,
        locations,
        times,
        data=None,
        coord_names: Sequence[str] | None = None,
        time_name: str = "t",
        var_names: Sequence[str] | None = None,
    ):
        locs       = _coerce_coords(locations)
        unique_t   = _coerce_times(times)
        n_loc, n_t = locs.shape[0], len(unique_t)

        # Allow (n_t, n_loc) or (n_t, n_loc, n_var) array input
        if data is not None and isinstance(data, np.ndarray):
            if data.ndim == 2 and data.shape == (n_t, n_loc):
                data = data.T.reshape(n_loc * n_t, 1)          # → time-major col
            elif data.ndim == 3 and data.shape[:2] == (n_t, n_loc):
                n_var_in = data.shape[2]
                data = data.reshape(n_t * n_loc, n_var_in)      # time-major rows

        # Build expanded coords/times for long storage (time-major)
        coords_exp = np.tile(locs, (n_t, 1))                    # (n_t * n_loc, d)
        times_exp  = np.repeat(unique_t, n_loc)                 # (n_t * n_loc,)

        # Store unique grid
        self._locations   = locs
        self._unique_times = unique_t
        self._n_loc        = n_loc
        self._n_t          = n_t

        super().__init__(
            coords=coords_exp, times=times_exp, data=data,
            coord_names=coord_names, time_name=time_name, var_names=var_names,
        )

    def _validate(self):
        expected = self._n_loc * self._n_t
        if self.n_obs != expected:
            raise ValueError(
                f"STF requires n_obs = n_loc * n_t = {self._n_loc} × {self._n_t} "
                f"= {expected}, got {self.n_obs}."
            )

    # ── Grid properties ───────────────────────────────────────────────────────

    @property
    def locations(self) -> np.ndarray:
        """(n_loc, d_space) unique spatial locations."""
        return self._locations.copy()

    @property
    def unique_times(self) -> np.ndarray:
        """(n_t,) unique time points."""
        return self._unique_times.copy()

    @property
    def n_loc(self) -> int:
        return self._n_loc

    @property
    def n_t(self) -> int:
        return self._n_t

    def __repr__(self) -> str:
        vstr = ", ".join(self.var_names) if self.var_names else "(no vars)"
        return (
            f"STF(n_loc={self.n_loc}, n_t={self.n_t}, d_space={self.d_space}, "
            f"t=[{self.t_min:.4g}, {self.t_max:.4g}], vars=[{vstr}])"
        )

    def summary(self, print_result: bool = True) -> dict:
        info = super().summary(print_result=False)
        info.update({"n_loc": self.n_loc, "n_t": self.n_t})
        if print_result:
            print("STF — Space-Time Full grid")
            print("─" * 40)
            print(f"  n_loc    : {self.n_loc}")
            print(f"  n_t      : {self.n_t}")
            print(f"  n_var    : {self.n_var}  {self.var_names}")
            print(f"  t range  : [{self.t_min:.4g}, {self.t_max:.4g}]")
            for cn, (lo, hi) in info["coord_range"].items():
                print(f"  {cn} range : [{lo:.4g}, {hi:.4g}]")
        return info

    # ── Conversion ────────────────────────────────────────────────────────────

    def to_array(self, var: str | int | None = None) -> np.ndarray:
        """
        Return a 3-D array of shape (n_t, n_loc, n_var) or (n_t, n_loc)
        when *var* selects a single variable.

        Parameters
        ----------
        var : str, int, or None
            Variable name or column index.  Returns all variables if None.
        """
        if not self.var_names:
            raise ValueError("No data variables stored in this STF.")
        if var is None:
            arr = self._data.values.reshape(self._n_t, self._n_loc, self.n_var)
            return arr
        col = self.var_names[var] if isinstance(var, int) else var
        return self._data[col].values.reshape(self._n_t, self._n_loc)

    def to_time_wide(
        self, var: str | int = 0, include_coords: bool = True
    ) -> pd.DataFrame:
        """
        Time-wide format: rows = locations, columns = time points.

        Each column holds the spatial field at one time step.  This is the
        natural format for plotting multiple time slices side-by-side or for
        temporal differencing.

        Parameters
        ----------
        var            : variable to extract
        include_coords : if True (default) prepend coordinate columns so
                         each row is self-describing; False returns the pure
                         (n_loc, n_t) value matrix.
        """
        col = self.var_names[var] if isinstance(var, int) else var
        arr = self._data[col].values.reshape(self._n_t, self._n_loc).T  # (n_loc, n_t)
        df  = pd.DataFrame(
            arr,
            columns=[f"{self.time_name}={t:.4g}" for t in self._unique_times],
        )
        if not include_coords:
            return df
        loc_df = pd.DataFrame(self._locations, columns=self.coord_names)
        return pd.concat([loc_df.reset_index(drop=True), df], axis=1)

    def to_space_wide(self, var: str | int = 0) -> pd.DataFrame:
        """
        Space-wide format: rows = time points, columns = locations.

        Each column is the time series at one location.  This is the natural
        format for spatial correlation analysis across time.
        """
        col = self.var_names[var] if isinstance(var, int) else var
        arr = self._data[col].values.reshape(self._n_t, self._n_loc)   # (n_t, n_loc)
        df  = pd.DataFrame(
            arr,
            index=[f"{self.time_name}={t:.4g}" for t in self._unique_times],
            columns=[f"loc{i}" for i in range(self._n_loc)],
        )
        return df

    # ── Subsetting ────────────────────────────────────────────────────────────

    def subset_time(
        self,
        t_values: Sequence[float] | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> "STF":
        """
        Return a new STF restricted to a subset of time points.

        Pass either an explicit list of time values or a (t_min, t_max) range.
        """
        if t_values is not None:
            mask_t = np.isin(self._unique_times, t_values)
        else:
            lo = t_min if t_min is not None else self.t_min
            hi = t_max if t_max is not None else self.t_max
            mask_t = (self._unique_times >= lo) & (self._unique_times <= hi)

        sel_t  = self._unique_times[mask_t]
        t_idx  = np.where(mask_t)[0]
        # Rows in time-major storage for selected time steps
        row_idx = np.concatenate([
            np.arange(k * self._n_loc, (k + 1) * self._n_loc) for k in t_idx
        ])
        return STF(
            locations=self._locations,
            times=sel_t,
            data=self._data.iloc[row_idx].values if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names or None,
        )

    def subset_locations(self, loc_indices: Sequence[int]) -> "STF":
        """
        Return a new STF restricted to the given location indices.
        """
        idx   = np.asarray(loc_indices, dtype=int)
        locs  = self._locations[idx]
        # In time-major storage, rows for location k are: k, k+n_loc, k+2*n_loc, …
        row_idx = np.concatenate([
            np.arange(k * self._n_loc, (k + 1) * self._n_loc)[idx]
            for k in range(self._n_t)
        ])
        return STF(
            locations=locs,
            times=self._unique_times,
            data=self._data.iloc[row_idx].values if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names or None,
        )

    def at_time(self, t: float, tol: float = 1e-12) -> pd.DataFrame:
        """
        Return a DataFrame of all locations at time *t*.

        The returned DataFrame has columns [coord_names..., var_names...].
        """
        k = np.where(np.abs(self._unique_times - t) <= tol)[0]
        if len(k) == 0:
            raise KeyError(f"Time {t} not found in unique_times.")
        k = int(k[0])
        rows = slice(k * self._n_loc, (k + 1) * self._n_loc)
        loc_df = pd.DataFrame(self._locations, columns=self.coord_names)
        return pd.concat([loc_df, self._data.iloc[rows].reset_index(drop=True)], axis=1)

    def at_location(self, loc_index: int) -> pd.DataFrame:
        """
        Return a time series DataFrame for a single location.

        The returned DataFrame has columns [time_name, var_names...].
        """
        row_idx = loc_index + self._n_loc * np.arange(self._n_t)
        df = pd.DataFrame({self.time_name: self._unique_times})
        return pd.concat(
            [df, self._data.iloc[row_idx].reset_index(drop=True)],
            axis=1,
        )

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply(self, func: Callable, new_var_name: str = "result") -> "STF":
        """
        Apply *func* to the data array and return a new STF with the result.

        *func* receives an (n_t, n_loc, n_var) array and must return an
        (n_t, n_loc) or (n_t, n_loc, k) array.
        """
        arr    = self.to_array()
        result = func(arr)
        if result.ndim == 2:
            result = result.reshape(self._n_t * self._n_loc, 1)
        else:
            result = result.reshape(self._n_t * self._n_loc, result.shape[2])
        n_v    = result.shape[1]
        vnames = [new_var_name] if n_v == 1 else [f"{new_var_name}{i}" for i in range(n_v)]
        return STF(
            locations=self._locations,
            times=self._unique_times,
            data=result,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=vnames,
        )

    # ── Format conversion ─────────────────────────────────────────────────────

    def to_sti(self) -> "STI":  # noqa: F821 — forward ref resolved at import time
        """Convert to STI (drops the full-grid invariant)."""
        from ._sti import STI
        return STI(
            coords=self.coords,
            times=self.times,
            data=self._data if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names or None,
        )

    def to_sts(self, missing_mask: np.ndarray | None = None) -> "STS":  # noqa: F821
        """
        Convert to STS.

        Parameters
        ----------
        missing_mask : (n_t, n_loc) bool array or None
            True where data is missing/excluded.  If None, all points kept.
        """
        from ._sts import STS
        if missing_mask is None:
            return STS(
                locations=self._locations,
                times=self._unique_times,
                data=self._data if self.var_names else None,
                coord_names=self.coord_names,
                time_name=self.time_name,
                var_names=self.var_names or None,
            )
        mask_flat = missing_mask.ravel()    # time-major
        keep      = ~mask_flat
        locs_rep  = np.tile(self._locations, (self._n_t, 1))[keep]
        times_rep = np.repeat(self._unique_times, self._n_loc)[keep]
        data_sub  = self._data[keep] if self.var_names else None
        return STS(
            locations=self._locations,
            times=self._unique_times,
            obs_coords=locs_rep,
            obs_times=times_rep,
            data=data_sub,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names or None,
        )

    # ── Class method constructors ─────────────────────────────────────────────

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        coord_names: Sequence[str],
        time_name: str = "t",
        var_names: Sequence[str] | None = None,
    ) -> "STF":
        """
        Build an STF from a tidy long-format DataFrame.

        The DataFrame must have one row per (location, time) observation in
        time-major order.  ``coord_names`` and ``time_name`` identify the
        relevant columns; remaining columns become data variables.
        """
        coords = df[list(coord_names)].values
        times  = df[time_name].values
        if var_names is None:
            var_names = [c for c in df.columns
                         if c not in list(coord_names) + [time_name]]
        data = df[list(var_names)].values if var_names else None

        # Infer unique locations and times
        unique_t   = np.unique(times)
        _, loc_idx = np.unique(coords, axis=0, return_index=True)
        unique_locs = coords[np.sort(loc_idx)]

        return cls(
            locations=unique_locs,
            times=unique_t,
            data=data,
            coord_names=list(coord_names),
            time_name=time_name,
            var_names=list(var_names) if var_names else None,
        )

    @classmethod
    def from_simulation(cls, result) -> "STF":
        """
        Build an STF directly from an ``STSimulationResult``.

        Parameters
        ----------
        result : STSimulationResult
            Output of ``gpbayeskit.simulation.simulate_st``.
        """
        n_t, n_h, n_w = result.field.shape
        S1_flat = result.S1.ravel()
        S2_flat = result.S2.ravel()
        locs    = np.column_stack([S1_flat, S2_flat])   # (n_loc, 2)
        n_var   = 1
        # field is (n_t, n_h, n_w) → reshape to (n_t, n_loc)
        field   = result.field.reshape(n_t, -1)          # (n_t, n_loc)
        # time-major long: (n_t * n_loc,)
        data    = field.ravel().reshape(-1, n_var)

        return cls(
            locations=locs,
            times=result.t,
            data=data,
            coord_names=["s1", "s2"],
            time_name="t",
            var_names=["y"],
        )


__all__ = ["STF"]
