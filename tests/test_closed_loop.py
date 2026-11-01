import numpy as np
import pytest

from wetware import MEA, Culture, biphasic_pulse


def test_callback_runs_on_sim_clock(small_culture):
    seen = []

    def cb(frame):
        seen.append((frame.cycle, frame.t_ms, frame.spikes.duration_ms))

    with small_culture.closed_loop(cb, hz=40) as loop:
        loop.run(duration_ms=1000)
    assert len(seen) == 40
    times = np.array([t for _, t, _ in seen])
    np.testing.assert_allclose(times, 25.0 * np.arange(1, 41), atol=small_culture.dt_ms / 2)
    assert loop.timing_error_ms <= small_culture.dt_ms / 2
    assert loop.recording is not None
    assert loop.recording.duration_ms == pytest.approx(1000.0)


def test_non_integer_period_stays_on_schedule():
    c = Culture(n_neurons=100, seed=0, mea=MEA(rows=2, cols=2))
    with c.closed_loop(lambda f: None, hz=30) as loop:
        loop.run(cycles=90)
    # 1000/30 ms is not a multiple of dt, but the error never accumulates.
    assert loop.timing_error_ms <= c.dt_ms / 2
    assert c.time_ms == pytest.approx(3000.0, abs=c.dt_ms)


def test_stimulation_from_callback_respects_latency(small_culture):
    def cb(frame):
        if frame.cycle == 2:
            frame.stimulate([1], biphasic_pulse(10.0))

    with small_culture.closed_loop(cb, hz=100, latency_ms=3.0) as loop:
        loop.run(cycles=6)
    stim_t = loop.recording.stim.times_ms
    assert stim_t.size == 1
    assert stim_t[0] == pytest.approx(30.0 + 3.0)


def test_stop_from_callback(small_culture):
    def cb(frame):
        if frame.cycle == 4:
            frame.stop()

    with small_culture.closed_loop(cb, hz=50) as loop:
        loop.run(cycles=100)
    assert loop.cycle == 5


def test_realtime_pacing_counts_overruns():
    c = Culture(n_neurons=100, seed=0, mea=MEA(rows=2, cols=2))
    with c.closed_loop(lambda f: None, hz=100, realtime=True, speed=50.0) as loop:
        loop.run(cycles=10)
    assert len(loop.call_times_ms) == 10
    assert loop.overruns >= 0


def test_invalid_loop_args(small_culture):
    with pytest.raises(ValueError):
        with small_culture.closed_loop(lambda f: None, hz=0):
            pass
    with pytest.raises(ValueError):
        with small_culture.closed_loop(lambda f: None, hz=10, latency_ms=-1):
            pass
