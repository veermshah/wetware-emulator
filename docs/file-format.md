# HDF5 file format

`SpikeTrains.to_hdf5(path)` (or `wetware.save_hdf5`) writes one recording per
file. The layout borrows group names and conventions from Neurodata Without
Borders (NWB 2.x) so it feels familiar, but it is plain HDF5 written with
h5py and is **not** a validated NWB file. Times in the file are in seconds,
as in NWB; the Python API uses milliseconds and converts on load.

```text
/                                   attrs: nwb_like_version, identifier,
                                           session_description, session_start_time,
                                           file_create_date, simulator
/general/culture                    attrs: metadata_json (seed, model, age, health,
                                           protocol and epochs when run from YAML)
/general/devices/mea                attrs: rows, cols, pitch_um,
                                           electrode_diameter_um, sample_rate_hz
/general/extracellular_ephys/electrodes
    id            (n_el,)  int
    x, y          (n_el,)  float    µm, array centred on the origin
    dead          (n_el,)  bool
/acquisition/ElectricalSeries       only when recorded with raw=True
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
    charge_nc          (n_pulses,) float64  effective charge per pulse
    peak_amplitude_ua  (n_pulses,) float64
/analysis/ground_truth              only when recorded with ground_truth=True
    neuron_ids         (n,) int64
    spike_times        (n,) float64 s
```

`spike_times` and `spike_times_index` follow NWB's ragged-array convention:
the spikes of electrode `i` are `spike_times[index[i-1]:index[i]]`.

## Reading a file without wetware

```python
import h5py

with h5py.File("recording.h5") as f:
    g = f["processing/ecephys/detected_spikes"]
    t, idx = g["spike_times"][()], g["spike_times_index"][()]
    electrode_5 = t[idx[4] : idx[5]]  # seconds
    volts = f["acquisition/ElectricalSeries/data"][:, 5] * 1e-6
```

## Reading it back

```python
from wetware import load_hdf5

rec = load_hdf5("recording.h5")  # a SpikeTrains, times in ms again
```
