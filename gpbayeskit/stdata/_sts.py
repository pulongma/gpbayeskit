"""
gpbayeskit.stdata._sts
========================
STS — Space-Time Sparse grid.

Like STF, the locations and time points come from a fixed lattice, but only
a subset of the n_loc × n_t combinations is observed.  Missing combinations
are tracked explicitly so the lattice structure is preserved.

Invariant
---------
Every observed (location, time) pair must be a node on the declared lattice
(i.e. location ∈ ``locations`` and time ∈ ``unique_times``).  The number of
observations n_obs ≤ n_loc × n_t.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._base import _STBase, _coerce_coords, _coerce_times


class STS(_STBase):
    """
    Space-Time Sparse grid: a subset of an implied space-time lattice.

    Parameters
    ----------
    locations   : (n_loc, d_space)   All declared spatial locations.
    times       : (n_t,)             All declared time points.
    obs_coords  : (n_obs, d_space)   Spatial coordinates of observed points.
    obs_times   : (n_obs,)           Time stamps of observed points.
    data        : array, dict, or DataFrame of shape (n_obs, n_var)
    coord_names, time_name, var_names : as in STF.

    Attributes
    ----------
    locations    : (n_loc, d_space)  Full declared location set.
    unique_times : (n_t,)            Full declared time set.
    n_loc, n_t   : int               Declared lattice dimensions.
    n_obs        : int               Number of observed (non-missing) points.
    missing_mask : (n_t, n_loc) bool  True where observations are absent.

    Examples
    --------
    >>> locs = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    >>> t    = np.array([0.0, 0.5, 1.0])
    >>> # Only observe 2 of the 4 locations at each time step
    >>> obs_locs  = np.tile(locs[:2], (3, 1))
    >>> obs_times = np.repeat(t, 2)
    >>> y = np.random.randn(6)
    >>> sts = STS(locs, t, obs_locs, obs_times, data=y, var_names=["y"])
    >>> sts.missing_mask.shape
    (3, 4)
    """

    _type_name = "STS"

    def __init__(
        self,
        locations,
        times,
        obs_coords=None,
        obs_times=None,
        data=None,
        coord_names: Sequence[str] | None = None,
        time_name: str = "t",
        var_names: Sequence[str] | None = None,
    ):
        self._locations    = _coerce_coords(locations)
        self._unique_times = _coerce_times(times)
        self._n_loc        = self._locations.shape[0]
        self._n_t          = len(self._unique_times)

        # If obs_coords/obs_times are omitted assume full grid (same as STF)
        if obs_coords is None:
            obs_coords = np.tile(self._locations, (self._n_t, 1))
            obs_times  = np.repeat(self._unique_times, self._n_loc)

        super().__init__(
            coords=obs_coords, times=obs_times, data=data,
            coord_names=coord_names, time_name=time_name, var_names=var_names,
        )

    def _validate(self):
        """Each observed point must be on the declared lattice."""
        tol = 1e-12
        # Check times
        for t in self.times:
            if not np.any(np.abs(self._unique_times - t) <= tol):
                raise ValueError(
                    f"Observed time {t} is not in the declared unique_times lattice."
                )
        # Check locations (round-trip via nearest-index)
        for s in self.coords:
            dists = np.linalg.norm(self._locations - s, axis=1)
            if dists.min() > tol:
                raise ValueError(
                    f"Observed location {s} is not in the declared locations lattice."
                )

    # ── Grid properties ───────────────────────────────────────────────────────

    @property
    def locations(self) -> np.ndarray:
        return self._locations.copy()

    @property
    def unique_times(self) -> np.ndarray:
        return self._unique_times.copy()

    @property
    def n_loc(self) -> int:
        return self._n_loc

    @property
    def n_t(self) -> int:
        return self._n_t

    @property
    def missing_mask(self) -> np.ndarray:
        """
        (n_t, n_loc) boolean array — True where observation is absent.

        Rows = time steps, columns = locations (matching declared order).
        """
        mask = np.ones((self._n_t, self._n_loc), dtype=bool)
        tol  = 1e-12
        for obs_s, obs_t in zip(self.coords, self.times):
            t_idx = int(np.argmin(np.abs(self._unique_times - obs_t)))
            l_idx = int(np.argmin(np.linalg.norm(self._locations - obs_s, axis=1)))
            mask[t_idx, l_idx] = False
        return mask

    @property
    def observed_fraction(self) -> float:
        """Fraction of the full lattice that is observed."""
        return 1.0 - self.missing_mask.mean()

    def __repr__(self) -> str:
        vstr = ", ".join(self.var_names) if self.var_names else "(no vars)"
        return (
            f"STS(n_obs={self.n_obs}/{self.n_loc*self.n_t}, "
            f"n_loc={self.n_loc}, n_t={self.n_t}, "
            f"obs_frac={self.observed_fraction:.2f}, "
            f"t=[{self.t_min:.4g}, {self.t_max:.4g}], vars=[{vstr}])"
        )

    def summary(self, print_result: bool = True) -> dict:
        info = super().summary(print_result=False)
        info.update({
            "n_loc": self.n_loc, "n_t": self.n_t,
            "observed_fraction": self.observed_fraction,
        })
        if print_result:
            print("STS — Space-Time Sparse grid")
            print("─" * 40)
            print(f"  n_obs    : {self.n_obs} / {self.n_loc * self.n_t} "
                  f"({self.observed_fraction:.1%} observed)")
            print(f"  n_loc    : {self.n_loc}")
            print(f"  n_t      : {self.n_t}")
            print(f"  n_var    : {self.n_var}  {self.var_names}")
            print(f"  t range  : [{self.t_min:.4g}, {self.t_max:.4g}]")
        return info

    # ── Conversion ────────────────────────────────────────────────────────────

    def to_masked_array(self, var: str | int = 0) -> np.ma.MaskedArray:
        """
        (n_t, n_loc) masked array; masked entries are unobserved.

        Parameters
        ----------
        var : str or int  Variable to extract.
        """
        col  = self.var_names[var] if isinstance(var, int) else var
        arr  = np.ma.masked_all((self._n_t, self._n_loc))
        tol  = 1e-12
        vals = self._data[col].values
        for row_i, (obs_s, obs_t) in enumerate(zip(self.coords, self.times)):
            t_idx = int(np.argmin(np.abs(self._unique_times - obs_t)))
            l_idx = int(np.argmin(np.linalg.norm(self._locations - obs_s, axis=1)))
            arr[t_idx, l_idx] = vals[row_i]
        return arr

    def to_time_wide(
        self, var: str | int = 0, include_coords: bool = True
    ) -> pd.DataFrame:
        """
        Time-wide format: rows = locations, columns = time points.
        Missing lattice entries appear as NaN.

        Parameters
        ----------
        var            : variable to extract
        include_coords : prepend coordinate columns when True (default)
        """
        ma  = self.to_masked_array(var)              # (n_t, n_loc)
        arr = np.where(ma.mask, np.nan, ma.data).T   # (n_loc, n_t)
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
        Missing lattice entries appear as NaN.
        """
        ma  = self.to_masked_array(var)
        arr = np.where(ma.mask, np.nan, ma.data)   # (n_t, n_loc)
        return pd.DataFrame(
            arr,
            index=[f"{self.time_name}={t:.4g}" for t in self._unique_times],
            columns=[f"loc{i}" for i in range(self._n_loc)],
        )

    def to_stf(self, fill_value: float = np.nan) -> "STF":  # noqa: F821
        """
        Upgrade to STF by filling missing entries with *fill_value*.
        Only valid for single-variable STS objects.
        """
        from ._stf import STF
        if self.n_var == 0:
            return STF(self._locations, self._unique_times,
                       coord_names=self.coord_names, time_name=self.time_name)
        full_arr = np.full((self._n_t, self._n_loc, self.n_var), fill_value)
        tol = 1e-12
        for row_i, (obs_s, obs_t) in enumerate(zip(self.coords, self.times)):
            t_idx = int(np.argmin(np.abs(self._unique_times - obs_t)))
            l_idx = int(np.argmin(np.linalg.norm(self._locations - obs_s, axis=1)))
            full_arr[t_idx, l_idx, :] = self._data.iloc[row_i].values
        return STF(
            locations=self._locations,
            times=self._unique_times,
            data=full_arr.reshape(self._n_t * self._n_loc, self.n_var),
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names,
        )

    def to_sti(self) -> "STI":  # noqa: F821
        """Drop the lattice structure and return an STI."""
        from ._sti import STI
        return STI(
            coords=self.coords, times=self.times,
            data=self._data if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=self.var_names or None,
        )

    # ── Subsetting ────────────────────────────────────────────────────────────

    def subset_time(
        self,
        t_values: Sequence[float] | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> "STS":
        if t_values is not None:
            keep_t = np.isin(self._unique_times, t_values)
        else:
            lo = t_min if t_min is not None else self.t_min
            hi = t_max if t_max is not None else self.t_max
            keep_t = (self._unique_times >= lo) & (self._unique_times <= hi)
        sel_unique_t = self._unique_times[keep_t]
        # Keep only observations whose time is in selected set
        obs_mask = np.isin(self.times, sel_unique_t)
        return STS(
            locations=self._locations,
            times=sel_unique_t,
            obs_coords=self.coords[obs_mask],
            obs_times=self.times[obs_mask],
            data=self._data[obs_mask] if self.var_names else None,
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
        all_locations: np.ndarray | None = None,
        all_times: np.ndarray | None = None,
    ) -> "STS":
        """Build STS from a long-format DataFrame (may have missing rows)."""
        obs_coords = df[list(coord_names)].values
        obs_times  = df[time_name].values
        if var_names is None:
            var_names = [c for c in df.columns
                         if c not in list(coord_names) + [time_name]]
        data = df[list(var_names)].values if var_names else None

        locs  = all_locations if all_locations is not None else np.unique(obs_coords, axis=0)
        times = all_times     if all_times     is not None else np.unique(obs_times)

        return cls(
            locations=locs, times=times,
            obs_coords=obs_coords, obs_times=obs_times,
            data=data,
            coord_names=list(coord_names),
            time_name=time_name,
            var_names=list(var_names) if var_names else None,
        )


__all__ = ["STS"]
