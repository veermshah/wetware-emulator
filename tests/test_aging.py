import numpy as np
import pytest

from wetware import MEA, Culture


def test_age_advances_div_and_drops_electrodes():
    c = Culture(n_neurons=300, seed=4, mea=MEA(rows=8, cols=8), div=21)
    h0 = c.health
    h = c.age(120)
    assert h.div == pytest.approx(141)
    assert h.working_electrodes < h0.working_electrodes
    assert h.alive_fraction < 1.0
    assert h.drive_scale < h0.drive_scale


def test_old_culture_fires_less():
    young = Culture(n_neurons=400, seed=2, mea=MEA(rows=4, cols=4), div=28)
    old = Culture(n_neurons=400, seed=2, mea=MEA(rows=4, cols=4), div=28)
    old.age(150)
    r_young = young.record(3000, ground_truth=True).gt_times_ms.size
    r_old = old.record(3000, ground_truth=True).gt_times_ms.size
    assert r_old < 0.7 * r_young


def test_immature_culture_is_weakly_connected():
    c = Culture(n_neurons=200, seed=0, mea=MEA(rows=2, cols=2), div=7)
    assert c.network.weight_scale < 0.1
    c.age(14)
    assert c.network.weight_scale > 0.5


def test_dead_electrodes_record_nothing():
    c = Culture(n_neurons=300, seed=1, mea=MEA(rows=4, cols=4))
    c.mea.mark_dead([0, 5])
    rec = c.record(1000, raw=True)
    assert rec.counts[0] == 0 and rec.counts[5] == 0
    assert rec.dead_electrodes == [0, 5]
    assert np.abs(rec.voltage_uv[0]).max() < 5.0


def test_negative_age_rejected():
    c = Culture(n_neurons=100, seed=0, mea=MEA(rows=2, cols=2))
    with pytest.raises(ValueError):
        c.age(-1)
