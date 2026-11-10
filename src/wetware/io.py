"""NWB-like HDF5 export and import.

The layout borrows the group names and conventions of the Neurodata Without
Borders (NWB 2.x) schema, so the files feel familiar, but it is plain HDF5
written with h5py and is **not** a validated NWB file. Times are stored in
seconds, as in NWB, and converted back to ms on load.

::

    /                                   attrs: nwb_like_version, identifier,
                                               session_description, session_start_time,
                                               file_create_date, simulator
    /general/culture                    attrs: metadata_json (seed, model, age, ...)
    /general/devices/mea                attrs: rows, cols, pitch_um, electrode_diameter_um,
                                               sample_rate_hz
    /general/extracellular_ephys/electrodes
        id            (n_el,)  int
        x, y          (n_el,)  float   µm
        dead          (n_el,)  bool
    /acquisition/ElectricalSeries       (only if raw voltage was recorded)
        data          (n_samples, n_el) float32, attrs unit="volts", conversion=1e-6
        attrs: rate (Hz), starting_time (s)
    /processing/ecephys/detected_spikes
        spike_times        (n_spikes,) float64  s, grouped by electrode
        spike_times_index  (n_el,)     int64    end offset of each electrode's block
        electrodes         (n_el,)     int64    electrode id of each block
        attrs: t_start (s), duration (s)
    /stimulus/presentation/stim_pulses
        timestamps         (n_pulses,) float64  s
        electrodes         (n_pulses, max_el) int64, padded with -1
        charge_nc          (n_pulses,) float64
        peak_amplitude_ua  (n_pulses,) float64
    /analysis/ground_truth              (only if ground truth was recorded)
        neuron_ids         (n,) int64
        spike_times        (n,) float64 s

``spike_times`` / ``spike_times_index`` follow NWB's ragged-array
(VectorData / VectorIndex) convention: electrode ``i``'s spikes are
``spike_times[index[i-1]:index[i]]``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .recording import SpikeTrains, StimLog

FORMAT_VERSION = "0.1"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"cannot serialise {type(o)}")


def save_hdf5(
    rec: SpikeTrains,
    path: str | Path,
    session_description: str = "wetware simulated MEA recording",
    compression: str | None = "gzip",
) -> Path:
    """Write a recording to ``path`` using the layout above.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    meta = rec.metadata
    mea = meta.get("mea", {})
    with h5py.File(path, "w") as f:
        f.attrs["nwb_like_version"] = FORMAT_VERSION
        f.attrs["identifier"] = str(uuid.uuid4())
        f.attrs["session_description"] = session_description
        f.attrs["session_start_time"] = now
        f.attrs["file_create_date"] = now
        f.attrs["simulator"] = str(meta.get("simulator", "wetware"))

        g = f.create_group("general")
        g.create_group("culture").attrs["metadata_json"] = json.dumps(meta, default=_json_default)
        dev = g.create_group("devices/mea")
        dev.attrs["rows"] = rec.grid_shape[0]
        dev.attrs["cols"] = rec.grid_shape[1]
        for k in ("pitch_um", "electrode_diameter_um"):
            if k in mea:
                dev.attrs[k] = mea[k]
        dev.attrs["sample_rate_hz"] = rec.sample_rate_hz
        el = g.create_group("extracellular_ephys/electrodes")
        el["id"] = np.arange(rec.n_electrodes)
        el["x"] = rec.electrode_positions_um[:, 0]
        el["y"] = rec.electrode_positions_um[:, 1]
        dead = np.zeros(rec.n_electrodes, dtype=bool)
        dead[list(rec.dead_electrodes)] = True
        el["dead"] = dead

        if rec.voltage_uv is not None:
            es = f.create_group("acquisition/ElectricalSeries")
            ds = es.create_dataset(
                "data", data=rec.voltage_uv.T.astype(np.float32), compression=compression
            )
            ds.attrs["unit"] = "volts"
            ds.attrs["conversion"] = 1e-6
            es.attrs["rate"] = rec.sample_rate_hz
            es.attrs["starting_time"] = rec.t_start_ms / 1000.0

        sp = f.create_group("processing/ecephys/detected_spikes")
        sp.attrs["t_start"] = rec.t_start_ms / 1000.0
        sp.attrs["duration"] = rec.duration_ms / 1000.0
        all_t = np.concatenate(rec.times) if rec.times else np.empty(0)
        sp.create_dataset("spike_times", data=all_t / 1000.0, compression=compression)
        sp["spike_times_index"] = np.cumsum([t.size for t in rec.times]).astype(np.int64)
        sp["electrodes"] = np.arange(rec.n_electrodes, dtype=np.int64)

        st = f.create_group("stimulus/presentation/stim_pulses")
        s = rec.stim
        st["timestamps"] = s.times_ms / 1000.0
        width = max((len(e) for e in s.electrodes), default=1)
        els = np.full((len(s), width), -1, dtype=np.int64)
        for i, e in enumerate(s.electrodes):
            els[i, : len(e)] = e
        st["electrodes"] = els
        st["charge_nc"] = s.charge_nc
        st["peak_amplitude_ua"] = s.peak_amplitude_ua

        if rec.gt_times_ms is not None and rec.gt_neurons is not None:
            gt = f.create_group("analysis/ground_truth")
            gt.create_dataset("neuron_ids", data=rec.gt_neurons, compression=compression)
            gt.create_dataset("spike_times", data=rec.gt_times_ms / 1000.0, compression=compression)
    return path


def load_hdf5(path: str | Path) -> SpikeTrains:
    """Read a file written by :func:`save_hdf5` back into a :class:`SpikeTrains`."""
    with h5py.File(path, "r") as f:
        meta = json.loads(f["general/culture"].attrs["metadata_json"])
        dev = f["general/devices/mea"]
        grid = (int(dev.attrs["rows"]), int(dev.attrs["cols"]))
        fs = float(dev.attrs["sample_rate_hz"])
        el = f["general/extracellular_ephys/electrodes"]
        pos = np.column_stack([el["x"][()], el["y"][()]])
        dead = [int(i) for i in np.flatnonzero(el["dead"][()])]

        sp = f["processing/ecephys/detected_spikes"]
        t_start = float(sp.attrs["t_start"]) * 1000.0
        duration = float(sp.attrs["duration"]) * 1000.0
        all_t = sp["spike_times"][()] * 1000.0
        idx = sp["spike_times_index"][()]
        starts = np.concatenate([[0], idx[:-1]])
        times = [all_t[a:b] for a, b in zip(starts, idx, strict=True)]

        st = f["stimulus/presentation/stim_pulses"]
        els = st["electrodes"][()]
        stim = StimLog(
            st["timestamps"][()] * 1000.0,
            [tuple(int(x) for x in row if x >= 0) for row in els],
            st["charge_nc"][()],
            st["peak_amplitude_ua"][()],
        )
        v = None
        if "acquisition/ElectricalSeries" in f:
            v = f["acquisition/ElectricalSeries/data"][()].T.astype(np.float32)
        gn = gt = None
        if "analysis/ground_truth" in f:
            gn = f["analysis/ground_truth/neuron_ids"][()]
            gt = f["analysis/ground_truth/spike_times"][()] * 1000.0
    return SpikeTrains(t_start, duration, times, pos, grid, dead, stim, v, fs, gn, gt, meta)
