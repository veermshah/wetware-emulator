import numpy as np

from wetware import MEA, Culture, biphasic_pulse


def _run(seed: int):
    c = Culture(n_neurons=300, seed=seed, mea=MEA(rows=4, cols=4))
    c.stimulate([3, 9], biphasic_pulse(12.0))
    rec = c.record(600, raw=True, ground_truth=True)
    c.age(10)
    rec2 = c.record(300)
    return rec, rec2, c


def test_same_seed_identical():
    a, a2, ca = _run(11)
    b, b2, cb = _run(11)
    for x, y in zip(a.times, b.times, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.voltage_uv, b.voltage_uv)
    np.testing.assert_array_equal(a.gt_times_ms, b.gt_times_ms)
    for x, y in zip(a2.times, b2.times, strict=True):
        np.testing.assert_array_equal(x, y)
    assert ca.mea.dead_electrodes == cb.mea.dead_electrodes


def test_different_seed_differs():
    a, _, _ = _run(1)
    b, _, _ = _run(2)
    assert not np.array_equal(a.gt_times_ms, b.gt_times_ms)


def test_raw_flag_does_not_change_spikes():
    c1 = Culture(n_neurons=300, seed=5, mea=MEA(rows=4, cols=4))
    c2 = Culture(n_neurons=300, seed=5, mea=MEA(rows=4, cols=4))
    r1 = c1.record(400, raw=True)
    r2 = c2.record(400, raw=False)
    for x, y in zip(r1.times, r2.times, strict=True):
        np.testing.assert_array_equal(x, y)
