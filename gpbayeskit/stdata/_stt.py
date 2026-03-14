"""
gpbayeskit.stdata._stt
========================
STT — Space-Time Trajectories.

An ordered sequence of (s, t) pairs partitioned into named trajectories
by an ID column.  Each trajectory is an STI internally; the STT is the
collection.  This is the natural format for tracking data (animals,
vehicles, weather systems, lagrangian floats).

Invariant
---------
Within each trajectory the time stamps must be strictly increasing
(sorted ascending).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._base import _STBase
from ._sti  import STI


class STT(_STBase):
    """
    Space-Time Trajectories: ordered (s, t) paths with trajectory IDs.

    Parameters
    ----------
    coords   : (n_obs, d_space)   Spatial coordinates (all trajectories).
    times    : (n_obs,)           Time stamps (all trajectories).
    data     : array, dict, or DataFrame, shape (n_obs, n_var)
               Must contain *at least* the column named by ``id_col``.
    id_col   : str   Column name that identifies trajectory membership.
    coord_names, time_name, var_names : as in other ST classes.

    Attributes
    ----------
    trajectory_ids : np.ndarray   Sorted array of unique trajectory IDs.
    n_traj         : int          Number of distinct trajectories.

    Examples
    --------
    >>> rng    = np.random.default_rng(0)
    >>> n_each = 20
    >>> t      = np.linspace(0, 1, n_each)
    >>> coords = np.column_stack([np.cumsum(rng.normal(0, 0.05, n_each)),
    ...                           np.cumsum(rng.normal(0, 0.05, n_each))])
    >>> ids    = np.zeros(n_each, dtype=int)
    >>> stt    = STT(coords, t, data={"id": ids, "z": rng.standard_normal(n_each)},
    ...             id_col="id")
    >>> stt.n_traj
    1
    >>> traj = stt.get_trajectory(0)
    >>> isinstance(traj, STI)
    True
    """

    _type_name = "STT"

    def __init__(
        self,
        coords,
        times,
        data=None,
        coord_names: Sequence[str] | None = None,
        time_name: str = "t",
        id_col: str = "id",
        var_names: Sequence[str] | None = None,
    ):
        self._id_col = id_col
        super().__init__(
            coords=coords, times=times, data=data,
            coord_names=coord_names, time_name=time_name, var_names=var_names,
        )

    def _validate(self):
        if self._id_col not in self._data.columns:
            raise ValueError(
                f"id_col '{self._id_col}' not found in data columns "
                f"{list(self._data.columns)}."
            )
        # Within each trajectory, times must be non-decreasing
        ids = self._data[self._id_col].values
        for uid in np.unique(ids):
            mask = ids == uid
            t_traj = self.times[mask]
            if not np.all(np.diff(t_traj) >= 0):
                raise ValueError(
                    f"Trajectory id={uid} has non-monotone time stamps. "
                    "Sort observations by time before constructing STT."
                )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def id_col(self) -> str:
        return self._id_col

    @property
    def trajectory_ids(self) -> np.ndarray:
        """Sorted array of unique trajectory IDs."""
        return np.unique(self._data[self._id_col].values)

    @property
    def n_traj(self) -> int:
        """Number of distinct trajectories."""
        return len(self.trajectory_ids)

    @property
    def traj_lengths(self) -> dict:
        """Dict mapping trajectory ID → number of observations."""
        ids = self._data[self._id_col].values
        return {uid: int(np.sum(ids == uid)) for uid in self.trajectory_ids}

    def __repr__(self) -> str:
        vstr = ", ".join(v for v in self.var_names if v != self._id_col) or "(no vars)"
        return (
            f"STT(n_traj={self.n_traj}, n_obs={self.n_obs}, "
            f"d_space={self.d_space}, "
            f"t=[{self.t_min:.4g}, {self.t_max:.4g}], vars=[{vstr}])"
        )

    def summary(self, print_result: bool = True) -> dict:
        info = super().summary(print_result=False)
        lens = list(self.traj_lengths.values())
        info.update({
            "n_traj": self.n_traj,
            "traj_length_mean": float(np.mean(lens)),
            "traj_length_min":  int(np.min(lens)),
            "traj_length_max":  int(np.max(lens)),
        })
        if print_result:
            print("STT — Space-Time Trajectories")
            print("─" * 40)
            print(f"  n_traj   : {self.n_traj}")
            print(f"  n_obs    : {self.n_obs}")
            print(f"  length   : mean={np.mean(lens):.1f}, "
                  f"min={np.min(lens)}, max={np.max(lens)}")
            print(f"  d_space  : {self.d_space}")
            print(f"  t range  : [{self.t_min:.4g}, {self.t_max:.4g}]")
            vars_shown = [v for v in self.var_names if v != self._id_col]
            print(f"  vars     : {vars_shown}")
        return info

    # ── Trajectory access ─────────────────────────────────────────────────────

    def get_trajectory(self, traj_id) -> STI:
        """
        Return a single trajectory as an STI.

        Parameters
        ----------
        traj_id : trajectory ID value (as stored in the id_col).
        """
        mask   = self._data[self._id_col].values == traj_id
        vnames = [v for v in self.var_names if v != self._id_col]
        return STI(
            coords=self.coords[mask],
            times=self.times[mask],
            data=self._data[vnames].iloc[mask] if vnames else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=vnames or None,
        )

    def iter_trajectories(self):
        """Yield (traj_id, STI) pairs for each trajectory."""
        for uid in self.trajectory_ids:
            yield uid, self.get_trajectory(uid)

    # ── Subsetting ────────────────────────────────────────────────────────────

    def subset_trajectories(self, traj_ids) -> "STT":
        """Return a new STT keeping only the specified trajectory IDs."""
        ids  = self._data[self._id_col].values
        mask = np.isin(ids, traj_ids)
        return STT(
            coords=self.coords[mask],
            times=self.times[mask],
            data=self._data[mask] if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            id_col=self._id_col,
            var_names=self.var_names or None,
        )

    def subset_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> "STT":
        """Clip all trajectories to [t_min, t_max]."""
        lo   = t_min if t_min is not None else self.t_min
        hi   = t_max if t_max is not None else self.t_max
        mask = (self.times >= lo) & (self.times <= hi)
        return STT(
            coords=self.coords[mask],
            times=self.times[mask],
            data=self._data[mask] if self.var_names else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            id_col=self._id_col,
            var_names=self.var_names or None,
        )


    # ── Wide formats ──────────────────────────────────────────────────────────

    def to_time_wide(
        self, var: str | int = 0, agg_func: str = "mean",
        include_coords: bool = True,
    ) -> pd.DataFrame:
        """
        Time-wide format across all trajectories: rows = locations,
        columns = time points.  Delegates to STI after flattening.

        Trajectory ID is added as a column when available.  Duplicate
        (location, time) entries are aggregated with *agg_func*.
        """
        return self.to_sti(include_id=True).to_time_wide(
            var=var, agg_func=agg_func, include_coords=include_coords,
        )

    def to_space_wide(
        self, var: str | int = 0, agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Space-wide format across all trajectories: rows = time points,
        columns = locations.  Delegates to STI after flattening.
        """
        return self.to_sti().to_space_wide(var=var, agg_func=agg_func)

    def to_long_per_traj(self) -> pd.DataFrame:
        """Long format with trajectory ID column included."""
        return self.to_long()

    # ── Statistics ────────────────────────────────────────────────────────────

    def bounding_boxes(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per trajectory containing the
        bounding box of each spatial coordinate.
        """
        rows = []
        for uid, traj in self.iter_trajectories():
            row = {"id": uid}
            for i, cn in enumerate(self.coord_names):
                row[f"{cn}_min"] = float(traj.coords[:, i].min())
                row[f"{cn}_max"] = float(traj.coords[:, i].max())
            row["t_min"] = float(traj.t_min)
            row["t_max"] = float(traj.t_max)
            rows.append(row)
        return pd.DataFrame(rows)

    def arc_lengths(self) -> pd.Series:
        """
        Return a Series of cumulative arc length per trajectory
        (sum of Euclidean step lengths in space).
        """
        lengths = {}
        for uid, traj in self.iter_trajectories():
            if len(traj.coords) < 2:
                lengths[uid] = 0.0
            else:
                diffs = np.diff(traj.coords, axis=0)
                lengths[uid] = float(np.linalg.norm(diffs, axis=1).sum())
        return pd.Series(lengths, name="arc_length")

    # ── Format conversion ─────────────────────────────────────────────────────

    def to_sti(self, include_id: bool = False) -> STI:
        """
        Flatten all trajectories into a single STI.

        Parameters
        ----------
        include_id : bool  If True, keep the ID column as a data variable.
        """
        data_cols = self.var_names if include_id else [
            v for v in self.var_names if v != self._id_col
        ]
        return STI(
            coords=self.coords, times=self.times,
            data=self._data[data_cols] if data_cols else None,
            coord_names=self.coord_names,
            time_name=self.time_name,
            var_names=data_cols or None,
        )

    # ── Class method constructors ─────────────────────────────────────────────

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        coord_names: Sequence[str],
        time_name: str = "t",
        id_col: str = "id",
        var_names: Sequence[str] | None = None,
    ) -> "STT":
        """Build STT from a long-format DataFrame with an ID column."""
        coords = df[list(coord_names)].values
        times  = df[time_name].values
        if var_names is None:
            var_names = [c for c in df.columns
                         if c not in list(coord_names) + [time_name]]
        data = df[list(var_names)] if var_names else None
        return cls(
            coords=coords, times=times, data=data,
            coord_names=list(coord_names),
            time_name=time_name,
            id_col=id_col,
            var_names=list(var_names) if var_names else None,
        )

    @classmethod
    def from_sti_list(
        cls,
        sti_list: list[STI],
        id_col: str = "id",
        ids=None,
    ) -> "STT":
        """
        Concatenate a list of STI objects into a single STT.

        Parameters
        ----------
        sti_list : list[STI]  Each element becomes one trajectory.
        id_col   : str        Name of the trajectory ID column.
        ids      : list or None
                   ID values for each STI.  Defaults to 0, 1, 2, …
        """
        if ids is None:
            ids = list(range(len(sti_list)))
        all_coords, all_times, all_data = [], [], []
        for uid, sti in zip(ids, sti_list):
            all_coords.append(sti.coords)
            all_times.append(sti.times)
            df = sti._data.copy()
            df[id_col] = uid
            all_data.append(df)
        coords = np.vstack(all_coords)
        times  = np.concatenate(all_times)
        data   = pd.concat(all_data, ignore_index=True)
        vnames = list(data.columns)
        return cls(
            coords=coords, times=times, data=data,
            coord_names=sti_list[0].coord_names,
            time_name=sti_list[0].time_name,
            id_col=id_col,
            var_names=vnames,
        )


__all__ = ["STT"]
