"""Recording container returned by :meth:`wetware.Culture.record`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class StimLog:
    """Stimulation pulses delivered during a recording (one row per pulse)."""

    times_ms: FloatArray = field(default_factory=lambda: np.empty(0))
    electrodes: list[tuple[int, ...]] = field(default_factory=list)
    charge_nc: FloatArray = field(default_factory=lambda: np.empty(0))
    peak_amplitude_ua: FloatArray = field(default_factory=lambda: np.empty(0))

    def __len__(self) -> int:
        return int(self.times_ms.size)


@dataclass
class SpikeTrains:
    """Spikes detected on each electrode, plus optional raw data.

    All times are absolute simulation times in ms.

    Attributes:
        t_start_ms: Start of the recording window.
        duration_ms: Length of the recording window.
        times: Detected spike times, one sorted array per electrode.
        electrode_positions_um: ``(n_electrodes, 2)`` electrode coordinates.
        grid_shape: ``(rows, cols)`` of the MEA.
        dead_electrodes: Electrodes that were not working.
        stim: Stimulation pulses delivered inside the window.
        voltage_uv: Optional ``(n_electrodes, n_samples)`` synthesized trace.
        sample_rate_hz: Sampling rate of ``voltage_uv``.
        gt_neurons, gt_times_ms: Optional ground-truth spikes of every neuron.
        metadata: Culture state (seed, age, model, ...).
    """

    t_start_ms: float
    duration_ms: float
    times: list[FloatArray]
    electrode_positions_um: FloatArray
    grid_shape: tuple[int, int]
    dead_electrodes: list[int] = field(default_factory=list)
    stim: StimLog = field(default_factory=StimLog)
    voltage_uv: NDArray[np.float32] | None = None
    sample_rate_hz: float = 10_000.0
    gt_neurons: NDArray[np.int64] | None = None
    gt_times_ms: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- basics
    @property
    def n_electrodes(self) -> int:
        return len(self.times)

    @property
    def t_stop_ms(self) -> float:
        return self.t_start_ms + self.duration_ms

    @property
    def counts(self) -> NDArray[np.int64]:
        """Number of detected spikes per electrode."""
        return np.array([t.size for t in self.times], dtype=np.int64)

    @property
    def firing_rates_hz(self) -> FloatArray:
        """Mean detected spike rate per electrode (Hz)."""
        return self.counts / (self.duration_ms / 1000.0)

    @property
    def active_electrodes(self) -> list[int]:
        """Electrodes with at least 0.1 Hz of detected activity."""
        return [int(e) for e in np.flatnonzero(self.firing_rates_hz >= 0.1)]

    def spike_times(self, electrode: int) -> FloatArray:
        """Detected spike times on one electrode (ms)."""
        return self.times[electrode]

    def all_spikes(self) -> tuple[FloatArray, NDArray[np.int64]]:
        """All detected spikes as ``(times_ms, electrode_ids)`` sorted by time."""
        if not self.times:
            return np.empty(0), np.empty(0, dtype=np.int64)
        t = np.concatenate(self.times)
        e = np.concatenate([np.full(x.size, i, dtype=np.int64) for i, x in enumerate(self.times)])
        order = np.argsort(t, kind="stable")
        return t[order], e[order]

    def binned(self, bin_ms: float = 10.0) -> FloatArray:
        """Spike counts in ``(n_electrodes, n_bins)`` bins of ``bin_ms``."""
        n_bins = max(1, int(np.ceil(self.duration_ms / bin_ms - 1e-9)))
        edges = self.t_start_ms + np.arange(n_bins + 1) * bin_ms
        return np.stack([np.histogram(t, edges)[0] for t in self.times]).astype(float)

    def mua(self, bin_ms: float = 10.0) -> FloatArray:
        """Multi-unit activity: per-electrode spike rate in Hz per bin."""
        return self.binned(bin_ms) / (bin_ms / 1000.0)

    def population_rate(self, bin_ms: float = 10.0) -> FloatArray:
        """Array-wide spike rate (Hz) per bin, summed over electrodes."""
        return self.mua(bin_ms).sum(axis=0)

    def window(self, start_ms: float, stop_ms: float) -> SpikeTrains:
        """A copy restricted to ``[start_ms, stop_ms)`` (raw data is sliced too)."""
        start_ms = max(start_ms, self.t_start_ms)
        stop_ms = min(stop_ms, self.t_stop_ms)
        times = [t[(t >= start_ms) & (t < stop_ms)] for t in self.times]
        s = self.stim
        m = (s.times_ms >= start_ms) & (s.times_ms < stop_ms)
        stim = StimLog(
            s.times_ms[m],
            [el for el, keep in zip(s.electrodes, m, strict=True) if keep],
            s.charge_nc[m],
            s.peak_amplitude_ua[m],
        )
        v = None
        if self.voltage_uv is not None:
            fs = self.sample_rate_hz / 1000.0
            a = int(round((start_ms - self.t_start_ms) * fs))
            b = int(round((stop_ms - self.t_start_ms) * fs))
            v = self.voltage_uv[:, a:b]
        gn = gt = None
        if self.gt_times_ms is not None and self.gt_neurons is not None:
            gm = (self.gt_times_ms >= start_ms) & (self.gt_times_ms < stop_ms)
            gn, gt = self.gt_neurons[gm], self.gt_times_ms[gm]
        return SpikeTrains(
            start_ms,
            stop_ms - start_ms,
            times,
            self.electrode_positions_um,
            self.grid_shape,
            list(self.dead_electrodes),
            stim,
            v,
            self.sample_rate_hz,
            gn,
            gt,
            dict(self.metadata),
        )

    @staticmethod
    def concatenate(parts: list[SpikeTrains]) -> SpikeTrains:
        """Join consecutive recordings into one."""
        if not parts:
            raise ValueError("nothing to concatenate")
        first, last = parts[0], parts[-1]
        n_el = first.n_electrodes
        times = [np.concatenate([p.times[e] for p in parts]) for e in range(n_el)]
        stim = StimLog(
            np.concatenate([p.stim.times_ms for p in parts]),
            [el for p in parts for el in p.stim.electrodes],
            np.concatenate([p.stim.charge_nc for p in parts]),
            np.concatenate([p.stim.peak_amplitude_ua for p in parts]),
        )
        v = None
        if all(p.voltage_uv is not None for p in parts):
            v = np.concatenate([p.voltage_uv for p in parts if p.voltage_uv is not None], axis=1)
        gn = gt = None
        if all(p.gt_times_ms is not None for p in parts):
            gn = np.concatenate([p.gt_neurons for p in parts if p.gt_neurons is not None])
            gt = np.concatenate([p.gt_times_ms for p in parts if p.gt_times_ms is not None])
        return SpikeTrains(
            first.t_start_ms,
            last.t_stop_ms - first.t_start_ms,
            times,
            first.electrode_positions_um,
            first.grid_shape,
            sorted(set().union(*[p.dead_electrodes for p in parts])),
            stim,
            v,
            first.sample_rate_hz,
            gn,
            gt,
            dict(last.metadata),
        )

    # ---------------------------------------------------------------- I/O
    def to_hdf5(self, path: str | Path, **kwargs: Any) -> Path:
        """Write to an NWB-like HDF5 file. See :func:`wetware.io.save_hdf5`."""
        from .io import save_hdf5

        return save_hdf5(self, path, **kwargs)

    @classmethod
    def from_hdf5(cls, path: str | Path) -> SpikeTrains:
        """Load a file written by :meth:`to_hdf5`."""
        from .io import load_hdf5

        return load_hdf5(path)

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        """JSON-friendly representation (used by the HTTP server)."""
        d: dict[str, Any] = {
            "t_start_ms": self.t_start_ms,
            "duration_ms": self.duration_ms,
            "times": [t.tolist() for t in self.times],
            "electrode_positions_um": self.electrode_positions_um.tolist(),
            "grid_shape": list(self.grid_shape),
            "dead_electrodes": list(self.dead_electrodes),
            "stim": {
                "times_ms": self.stim.times_ms.tolist(),
                "electrodes": [list(e) for e in self.stim.electrodes],
                "charge_nc": self.stim.charge_nc.tolist(),
                "peak_amplitude_ua": self.stim.peak_amplitude_ua.tolist(),
            },
            "sample_rate_hz": self.sample_rate_hz,
            "metadata": self.metadata,
        }
        if include_raw and self.voltage_uv is not None:
            d["voltage_uv"] = self.voltage_uv.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpikeTrains:
        s = d.get("stim", {})
        v = d.get("voltage_uv")
        return cls(
            float(d["t_start_ms"]),
            float(d["duration_ms"]),
            [np.asarray(t, dtype=float) for t in d["times"]],
            np.asarray(d["electrode_positions_um"], dtype=float),
            (int(d["grid_shape"][0]), int(d["grid_shape"][1])),
            [int(x) for x in d.get("dead_electrodes", [])],
            StimLog(
                np.asarray(s.get("times_ms", []), dtype=float),
                [tuple(int(x) for x in e) for e in s.get("electrodes", [])],
                np.asarray(s.get("charge_nc", []), dtype=float),
                np.asarray(s.get("peak_amplitude_ua", []), dtype=float),
            ),
            None if v is None else np.asarray(v, dtype=np.float32),
            float(d.get("sample_rate_hz", 10_000.0)),
            metadata=dict(d.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            f"SpikeTrains({self.t_start_ms:.1f}-{self.t_stop_ms:.1f} ms, "
            f"{self.n_electrodes} electrodes, {int(self.counts.sum())} spikes, "
            f"{len(self.stim)} stim pulses)"
        )
