import numpy as np
import pytest

from wetware import SpikeTrains, StimLog
from wetware import analysis as A


def _synthetic() -> SpikeTrains:
    """Hand-made recording: 4 electrodes, two bursts, one stim pulse."""
    rng = np.random.default_rng(0)
    times = []
    for e in range(4):
        bg = np.sort(rng.uniform(0, 2000, 4))
        b1 = 500 + rng.uniform(0, 40, 20)
        b2 = 1500 + rng.uniform(0, 40, 20)
        times.append(np.sort(np.concatenate([bg, b1, b2])))
    pos = np.array([[0, 0], [200, 0], [0, 200], [200, 200]], float)
    stim = StimLog(np.array([1000.0]), [(0,)], np.array([2.0]), np.array([10.0]))
    return SpikeTrains(0.0, 2000.0, times, pos, (2, 2), [3], stim)


def test_detect_bursts_finds_both():
    rec = _synthetic()
    bursts = A.detect_bursts(rec, min_electrodes=3)
    assert len(bursts) == 2
    assert bursts[0].start_ms == pytest.approx(500, abs=10)
    assert bursts[1].start_ms == pytest.approx(1500, abs=10)
    stats = A.burst_statistics(bursts, rec.duration_ms)
    assert stats["rate_hz"] == pytest.approx(1.0)
    assert stats["mean_ibi_ms"] == pytest.approx(1000, abs=20)


def test_rate_map_marks_dead():
    m = A.firing_rate_map(_synthetic())
    assert m.shape == (2, 2)
    assert np.isnan(m[1, 1])
    assert np.all(m[~np.isnan(m)] > 0)


def test_psth_peaks_after_event():
    rec = _synthetic()
    rec.times[0] = np.sort(np.concatenate([rec.times[0], 1000 + np.arange(5, 25, 2.0)]))
    c, r = A.psth(rec, window_ms=(-50, 100), bin_ms=10, electrodes=[0])
    assert c[np.argmax(r)] > 0
    resp = A.evoked_response(rec, window_ms=(0, 30), electrodes=[0])
    assert resp[0] >= 10


def test_cross_correlogram_lag():
    a = np.arange(0, 1000, 50.0)
    b = a + 3.0
    c, h = A.cross_correlogram(a, b, max_lag_ms=10, bin_ms=1)
    assert c[np.argmax(h)] == pytest.approx(3.5)


def test_functional_connectivity_is_symmetric():
    fc = A.functional_connectivity(_synthetic(), bin_ms=10, min_spikes=5)
    assert fc.shape == (4, 4)
    ok = ~np.isnan(fc)
    np.testing.assert_allclose(fc[ok], fc.T[ok])
    assert np.nanmin(fc) >= -1.0 - 1e-9 and np.nanmax(fc) <= 1.0 + 1e-9
    assert fc[0, 1] > 0.5  # shared bursts -> strong correlation


def test_summarize_keys():
    s = A.summarize(_synthetic())
    assert {"mean_rate_hz", "burst_rate_hz", "active_electrodes", "burst_fraction"} <= set(s)


def test_plotting_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from wetware import plotting

    fig = plotting.summary_figure(_synthetic(), title="t")
    fig.savefig(tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 1000
