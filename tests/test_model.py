import numpy as np

from wetware import MEA, Culture
from wetware.analysis import detect_bursts


def test_spontaneous_network_bursts():
    c = Culture(n_neurons=1000, seed=1)
    rec = c.record(10_000)
    bursts = detect_bursts(rec)
    assert 1 <= len(bursts) <= 30
    assert max(b.n_electrodes for b in bursts) >= 16
    # Bursts carry a large share of all spikes, as in mature cortical cultures.
    assert sum(b.n_spikes for b in bursts) > 0.2 * rec.counts.sum()


def test_detected_spikes_track_ground_truth():
    c = Culture(n_neurons=600, seed=2, mea=MEA(rows=4, cols=4))
    rec = c.record(2000, ground_truth=True)
    hits = total = 0
    for e in range(rec.n_electrodes):
        vis = c.recorder.visible_neurons(e, min_uv=60.0)
        gt = np.sort(rec.gt_times_ms[np.isin(rec.gt_neurons, vis)])
        if gt.size == 0:
            continue
        det = rec.times[e]
        idx = np.clip(np.searchsorted(det, gt), 1, max(det.size - 1, 1))
        if det.size == 0:
            total += gt.size
            continue
        near = np.minimum(np.abs(det[idx] - gt), np.abs(det[idx - 1] - gt))
        hits += int((near < 1.0).sum())
        total += gt.size
    assert total > 0
    assert hits / total > 0.7


def test_blanking_hides_artifact():
    from wetware import biphasic_pulse

    c = Culture(n_neurons=200, seed=0, mea=MEA(rows=4, cols=4))
    c.stimulate([5], biphasic_pulse(20.0))
    rec = c.record(50, raw=True)
    blank = int(c.recorder_params.blank_ms * c.mea.sample_rate_hz / 1000)
    assert np.all(rec.voltage_uv[:, :blank] == 0.0)
    for t in rec.times:
        assert np.all(t >= c.recorder_params.blank_ms)


def test_stdp_sign():
    """Pre-before-post potentiates, post-before-pre depresses."""
    from wetware.network import Network, NetworkParams
    from wetware.neurons import Izhikevich

    def pair(delta_ms: float) -> float:
        rng = np.random.default_rng(0)
        pos = rng.uniform(-50, 50, (20, 2))
        net = Network(pos, Izhikevich(), NetworkParams(p_connect_max=1.0), rng, rng)
        exc = np.flatnonzero(net.is_exc)
        pre, post = exc[0], exc[1]
        net.conn[pre, post] = net.exc_conn[pre, post] = True
        net.w[pre, post] = w0 = 0.5 * net.w_max
        first, second = (pre, post) if delta_ms > 0 else (post, pre)
        net._stdp(np.array([first]))
        decay = np.exp(-abs(delta_ms) / net.params.stdp_tau_plus_ms)
        net.trace_pre *= decay
        net.trace_post *= decay
        net._stdp(np.array([second]))
        return net.w[pre, post] - w0

    assert pair(+10.0) > 0
    assert pair(-10.0) < 0
