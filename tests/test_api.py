import numpy as np
import pytest

from wetware import (
    MEA,
    Culture,
    SpikeTrains,
    available_neuron_models,
    biphasic_pulse,
    pulse_train,
)


def test_mea_geometry():
    mea = MEA(rows=8, cols=8, pitch_um=200)
    assert mea.n_electrodes == 64
    pos = mea.positions
    assert pos.shape == (64, 2)
    assert np.allclose(pos.mean(axis=0), 0.0)
    assert mea.index(2, 3) == 19
    assert mea.grid_position(19) == (2, 3)
    assert np.isclose(np.diff(np.unique(pos[:, 0])).min(), 200.0)
    with pytest.raises(IndexError):
        mea.index(8, 0)


def test_record_returns_spiketrains(small_culture):
    rec = small_culture.record(500)
    assert isinstance(rec, SpikeTrains)
    assert rec.n_electrodes == 16
    assert rec.t_start_ms == 0.0
    assert rec.duration_ms == pytest.approx(500.0)
    assert small_culture.time_ms == pytest.approx(500.0)
    for t in rec.times:
        assert np.all(np.diff(t) >= 0)
        assert np.all((t >= 0) & (t < 500))
    assert rec.mua(10).shape == (16, 50)
    assert rec.voltage_uv is None


def test_raw_voltage_shape_and_noise(small_culture):
    rec = small_culture.record(200, raw=True)
    fs = small_culture.mea.sample_rate_hz
    assert rec.voltage_uv is not None
    assert rec.voltage_uv.shape == (16, int(200 * fs / 1000))
    assert rec.voltage_uv.dtype == np.float32
    # Noise floor is around the configured RMS.
    mad = np.median(np.abs(rec.voltage_uv), axis=1) / 0.6745
    assert 2.0 < np.median(mad) < 12.0


def test_consecutive_recordings_continue_time(small_culture):
    a = small_culture.record(100)
    b = small_culture.record(100)
    assert b.t_start_ms == pytest.approx(a.t_stop_ms)
    joined = SpikeTrains.concatenate([a, b])
    assert joined.duration_ms == pytest.approx(200.0)
    assert joined.counts.sum() == a.counts.sum() + b.counts.sum()


def test_stimulation_is_logged_and_evokes_activity():
    c = Culture(n_neurons=400, seed=3, mea=MEA(rows=4, cols=4))
    c.advance(300)
    quiet = c.record(1000, ground_truth=True)
    c.stimulate([5], pulse_train(20.0, n_pulses=10, frequency_hz=10, phase_us=200))
    stim = c.record(1000, ground_truth=True)
    assert len(stim.stim) == 10
    assert stim.stim.electrodes[0] == (5,)
    assert stim.stim.times_ms[0] == pytest.approx(quiet.t_stop_ms)
    # Ground-truth spikes within 10 ms of each pulse beat the unstimulated baseline.
    def early(rec, events):
        rel = rec.gt_times_ms[None, :] - events[:, None]
        return int(((rel >= 0) & (rel < 10)).sum())

    base_events = quiet.t_start_ms + np.arange(10) * 100.0
    assert early(stim, stim.stim.times_ms) > 2 * early(quiet, base_events) + 5


def test_pending_and_cancel(small_culture):
    small_culture.stimulate(0, biphasic_pulse(5.0), delay_ms=50)
    assert len(small_culture.pending_stimuli) == 1
    assert small_culture.cancel_stimulation() == 1
    rec = small_culture.record(100)
    assert len(rec.stim) == 0


def test_window_slices(small_culture):
    rec = small_culture.record(400, raw=True)
    w = rec.window(100, 200)
    assert w.duration_ms == pytest.approx(100)
    assert w.voltage_uv.shape[1] == int(100 * rec.sample_rate_hz / 1000)
    for t in w.times:
        assert np.all((t >= 100) & (t < 200))


def test_neuron_models_available():
    assert {"izhikevich", "lif"} <= set(available_neuron_models())


def test_lif_culture_runs():
    c = Culture(n_neurons=200, seed=1, neuron_model="lif", mea=MEA(rows=4, cols=4))
    rec = c.record(300, ground_truth=True)
    assert rec.metadata["neuron_model"] == "lif"
    assert rec.gt_times_ms.size > 0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        Culture(n_neurons=5)
    with pytest.raises(ValueError):
        Culture(n_neurons=100, dt_ms=2.0)
    c = Culture(n_neurons=100, seed=0, mea=MEA(rows=2, cols=2))
    with pytest.raises(ValueError):
        c.record(0)
    with pytest.raises(ValueError):
        Culture(n_neurons=100, neuron_model="nope")
