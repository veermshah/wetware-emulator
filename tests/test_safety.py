import pytest

from wetware import (
    MEA,
    Culture,
    Phase,
    PulseTrain,
    SafetyLimits,
    StimulationSafetyError,
    Waveform,
    asymmetric_biphasic_pulse,
    biphasic_pulse,
    pulse_train,
)


@pytest.fixture
def culture():
    return Culture(n_neurons=100, seed=0, mea=MEA(rows=4, cols=4))


def test_biphasic_is_balanced():
    w = biphasic_pulse(10.0, phase_us=200, interphase_us=50)
    assert w.net_charge_nc == pytest.approx(0.0)
    assert w.max_phase_charge_nc == pytest.approx(2.0)
    assert w.phases[0].amplitude_ua < 0  # cathodic first
    assert asymmetric_biphasic_pulse(8.0).net_charge_nc == pytest.approx(0.0)


def test_unbalanced_waveform_raises(culture):
    mono = Waveform((Phase(-10.0, 200.0),))
    with pytest.raises(StimulationSafetyError, match="charge-balanced"):
        culture.stimulate(0, mono)
    lopsided = Waveform((Phase(-10.0, 200.0), Phase(10.0, 150.0)))
    with pytest.raises(StimulationSafetyError):
        culture.stimulate(0, lopsided)


def test_amplitude_cap_raises(culture):
    with pytest.raises(StimulationSafetyError, match="amplitude"):
        culture.stimulate(0, biphasic_pulse(80.0, phase_us=50))


def test_charge_cap_raises(culture):
    with pytest.raises(StimulationSafetyError, match="charge"):
        culture.stimulate(0, biphasic_pulse(40.0, phase_us=400))


def test_phase_length_cap_raises(culture):
    with pytest.raises(StimulationSafetyError, match="phase"):
        culture.stimulate(0, biphasic_pulse(1.0, phase_us=2000))


def test_rate_limit_on_train_and_across_calls(culture):
    with pytest.raises(StimulationSafetyError, match="Hz"):
        culture.stimulate(0, pulse_train(5.0, n_pulses=5, frequency_hz=500))
    culture.stimulate(1, biphasic_pulse(5.0))
    with pytest.raises(StimulationSafetyError, match="closer"):
        culture.stimulate(1, biphasic_pulse(5.0), delay_ms=1.0)
    # A different electrode is fine.
    culture.stimulate(2, biphasic_pulse(5.0), delay_ms=1.0)


def test_rate_limit_against_delivered_pulse(culture):
    culture.stimulate(3, biphasic_pulse(5.0))
    culture.record(2.0)
    with pytest.raises(StimulationSafetyError):
        culture.stimulate(3, biphasic_pulse(5.0))
    culture.record(10.0)
    culture.stimulate(3, biphasic_pulse(5.0))


def test_dead_and_out_of_range_electrodes(culture):
    culture.mea.mark_dead(4)
    with pytest.raises(StimulationSafetyError, match="dead"):
        culture.stimulate([4, 5], biphasic_pulse(5.0))
    with pytest.raises(IndexError):
        culture.stimulate(99, biphasic_pulse(5.0))


def test_too_many_electrodes():
    c = Culture(
        n_neurons=100, seed=0, mea=MEA(rows=4, cols=4), safety=SafetyLimits(max_electrodes_per_pulse=2)
    )
    with pytest.raises(StimulationSafetyError):
        c.stimulate([0, 1, 2], biphasic_pulse(5.0))


def test_custom_limits_allow_more():
    c = Culture(
        n_neurons=100,
        seed=0,
        mea=MEA(rows=4, cols=4),
        safety=SafetyLimits(max_amplitude_ua=100.0, max_charge_per_phase_nc=50.0),
    )
    c.stimulate(0, biphasic_pulse(80.0, phase_us=100))


def test_nothing_scheduled_after_violation(culture):
    with pytest.raises(StimulationSafetyError):
        culture.stimulate(0, PulseTrain(biphasic_pulse(5.0), 3, 1000.0))
    assert culture.pending_stimuli == []


def test_waveform_validation():
    with pytest.raises(ValueError):
        Waveform(())
    with pytest.raises(ValueError):
        biphasic_pulse(-5.0)
    with pytest.raises(ValueError):
        PulseTrain(biphasic_pulse(5.0), 0, 10.0)
