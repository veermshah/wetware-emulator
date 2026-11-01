import h5py
import numpy as np

from wetware import MEA, Culture, SpikeTrains, biphasic_pulse, load_hdf5


def _rec(raw: bool, gt: bool) -> SpikeTrains:
    c = Culture(n_neurons=300, seed=9, mea=MEA(rows=4, cols=4))
    c.mea.mark_dead(2)
    c.stimulate([1, 6], biphasic_pulse(10.0))
    return c.record(500, raw=raw, ground_truth=gt)


def _same(a: SpikeTrains, b: SpikeTrains) -> None:
    assert a.t_start_ms == b.t_start_ms
    assert np.isclose(a.duration_ms, b.duration_ms)
    assert a.grid_shape == b.grid_shape
    assert a.dead_electrodes == b.dead_electrodes
    for x, y in zip(a.times, b.times, strict=True):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-9)
    np.testing.assert_allclose(a.stim.times_ms, b.stim.times_ms)
    assert a.stim.electrodes == b.stim.electrodes
    np.testing.assert_allclose(a.stim.charge_nc, b.stim.charge_nc)
    np.testing.assert_allclose(a.electrode_positions_um, b.electrode_positions_um)
    assert a.metadata["seed"] == b.metadata["seed"]


def test_roundtrip_spikes_only(tmp_path):
    rec = _rec(raw=False, gt=False)
    p = rec.to_hdf5(tmp_path / "rec.h5")
    back = load_hdf5(p)
    _same(rec, back)
    assert back.voltage_uv is None and back.gt_times_ms is None


def test_roundtrip_with_raw_and_ground_truth(tmp_path):
    rec = _rec(raw=True, gt=True)
    back = SpikeTrains.from_hdf5(rec.to_hdf5(tmp_path / "full.h5"))
    _same(rec, back)
    np.testing.assert_array_equal(rec.voltage_uv, back.voltage_uv)
    np.testing.assert_array_equal(rec.gt_neurons, back.gt_neurons)
    np.testing.assert_allclose(rec.gt_times_ms, back.gt_times_ms)


def test_layout_is_nwb_like(tmp_path):
    rec = _rec(raw=True, gt=False)
    p = rec.to_hdf5(tmp_path / "layout.h5")
    with h5py.File(p) as f:
        assert "nwb_like_version" in f.attrs
        assert f["acquisition/ElectricalSeries/data"].shape == rec.voltage_uv.T.shape
        assert f["acquisition/ElectricalSeries/data"].attrs["unit"] == "volts"
        idx = f["processing/ecephys/detected_spikes/spike_times_index"][()]
        assert idx[-1] == rec.counts.sum()
        # Times are stored in seconds.
        ts = f["stimulus/presentation/stim_pulses/timestamps"][()]
        np.testing.assert_allclose(ts * 1000.0, rec.stim.times_ms)


def test_dict_roundtrip():
    rec = _rec(raw=False, gt=False)
    back = SpikeTrains.from_dict(rec.to_dict())
    _same(rec, back)
