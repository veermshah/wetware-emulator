"""Stimulation waveforms, pulse trains and safety checks.

All stimulation is current-controlled. Amplitudes are in µA, durations in
µs, and charges in nC (1 µA x 1 µs = 1 pC = 1e-3 nC). Negative amplitude is
cathodic.

Every waveform passes through :class:`SafetyLimits` before it reaches the
culture. A violation raises :class:`StimulationSafetyError` instead of being
clipped quietly, which is the behaviour you want from real hardware too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class StimulationSafetyError(ValueError):
    """Raised when a stimulation request violates the safety limits."""


@dataclass(frozen=True)
class Phase:
    """One constant-current phase of a pulse.

    Args:
        amplitude_ua: Current in µA. Negative is cathodic, zero is a gap.
        duration_us: Phase length in µs.
    """

    amplitude_ua: float
    duration_us: float

    @property
    def charge_nc(self) -> float:
        """Signed charge delivered by this phase in nC."""
        return self.amplitude_ua * self.duration_us * 1e-3


@dataclass(frozen=True)
class Waveform:
    """A single stimulation pulse made of consecutive phases."""

    phases: tuple[Phase, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("a waveform needs at least one phase")
        for p in self.phases:
            if p.duration_us <= 0:
                raise ValueError("phase durations must be positive")

    @property
    def duration_us(self) -> float:
        """Total pulse length in µs."""
        return float(sum(p.duration_us for p in self.phases))

    @property
    def peak_amplitude_ua(self) -> float:
        """Largest absolute phase amplitude in µA."""
        return float(max(abs(p.amplitude_ua) for p in self.phases))

    @property
    def net_charge_nc(self) -> float:
        """Signed sum of the charge over all phases (zero when balanced)."""
        return float(sum(p.charge_nc for p in self.phases))

    @property
    def max_phase_charge_nc(self) -> float:
        """Largest absolute charge in any single phase."""
        return float(max(abs(p.charge_nc) for p in self.phases))

    @property
    def cathodic_charge_nc(self) -> float:
        """Total cathodic (negative) charge magnitude, which drives activation."""
        return float(sum(-p.charge_nc for p in self.phases if p.amplitude_ua < 0))

    @property
    def effective_charge_nc(self) -> float:
        """Charge used by the activation model: the cathodic charge, or the
        largest phase charge for purely anodic pulses."""
        return self.cathodic_charge_nc or self.max_phase_charge_nc

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {"phases": [[p.amplitude_ua, p.duration_us] for p in self.phases]}

    @classmethod
    def from_dict(cls, d: dict[str, list[list[float]]]) -> Waveform:
        return cls(tuple(Phase(float(a), float(t)) for a, t in d["phases"]))


@dataclass(frozen=True)
class PulseTrain:
    """``n_pulses`` copies of ``pulse`` delivered at ``frequency_hz``."""

    pulse: Waveform
    n_pulses: int
    frequency_hz: float

    def __post_init__(self) -> None:
        if self.n_pulses < 1:
            raise ValueError("n_pulses must be at least 1")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")

    @property
    def onsets_ms(self) -> np.ndarray:
        """Pulse onset times relative to the start of the train (ms)."""
        return np.arange(self.n_pulses) * (1000.0 / self.frequency_hz)

    @property
    def duration_ms(self) -> float:
        return float(self.onsets_ms[-1] + self.pulse.duration_us * 1e-3)


def biphasic_pulse(
    amplitude_ua: float,
    phase_us: float = 200.0,
    interphase_us: float = 0.0,
    cathodic_first: bool = True,
) -> Waveform:
    """Symmetric, charge-balanced biphasic pulse.

    Args:
        amplitude_ua: Magnitude of each phase in µA.
        phase_us: Duration of each phase in µs.
        interphase_us: Optional gap between the phases.
        cathodic_first: Lead with the cathodic (negative) phase.
    """
    if amplitude_ua < 0:
        raise ValueError("give amplitude_ua as a magnitude; use cathodic_first for polarity")
    sign = -1.0 if cathodic_first else 1.0
    phases = [Phase(sign * amplitude_ua, phase_us)]
    if interphase_us > 0:
        phases.append(Phase(0.0, interphase_us))
    phases.append(Phase(-sign * amplitude_ua, phase_us))
    return Waveform(tuple(phases))


def asymmetric_biphasic_pulse(
    amplitude_ua: float, phase_us: float = 200.0, ratio: float = 4.0
) -> Waveform:
    """Charge-balanced pulse with a short strong cathodic phase and a long
    weak anodic recovery phase (``ratio`` times longer, ``ratio`` times weaker)."""
    return Waveform(
        (
            Phase(-amplitude_ua, phase_us),
            Phase(amplitude_ua / ratio, phase_us * ratio),
        )
    )


def pulse_train(
    amplitude_ua: float,
    n_pulses: int,
    frequency_hz: float,
    phase_us: float = 200.0,
    interphase_us: float = 0.0,
) -> PulseTrain:
    """Train of identical biphasic pulses."""
    return PulseTrain(biphasic_pulse(amplitude_ua, phase_us, interphase_us), n_pulses, frequency_hz)


@dataclass(frozen=True)
class SafetyLimits:
    """Hard limits checked on every stimulation request.

    The defaults are deliberately conservative and simple. They are not a
    statement about any particular electrode material; set them to whatever
    your real setup allows.

    Args:
        max_amplitude_ua: Largest allowed phase amplitude.
        max_phase_us: Longest allowed single phase.
        max_charge_per_phase_nc: Largest allowed charge in one phase.
        max_charge_density_uc_cm2: Largest allowed charge density per phase
            (µC/cm², using the MEA's geometric electrode area).
        balance_tolerance: Allowed ``|net charge| / max phase charge``.
        max_pulse_rate_hz: Highest allowed pulse rate on one electrode.
        max_electrodes_per_pulse: Most electrodes stimulated at once.
    """

    max_amplitude_ua: float = 50.0
    max_phase_us: float = 1000.0
    max_charge_per_phase_nc: float = 10.0
    max_charge_density_uc_cm2: float = 2000.0
    balance_tolerance: float = 0.01
    max_pulse_rate_hz: float = 200.0
    max_electrodes_per_pulse: int = 16

    def check_waveform(self, waveform: Waveform, electrode_area_cm2: float | None = None) -> None:
        """Raise :class:`StimulationSafetyError` if the pulse is unsafe."""
        if waveform.peak_amplitude_ua > self.max_amplitude_ua:
            raise StimulationSafetyError(
                f"amplitude {waveform.peak_amplitude_ua:g} µA exceeds the "
                f"{self.max_amplitude_ua:g} µA limit"
            )
        longest = max(p.duration_us for p in waveform.phases if p.amplitude_ua != 0) if any(
            p.amplitude_ua != 0 for p in waveform.phases
        ) else 0.0
        if longest > self.max_phase_us:
            raise StimulationSafetyError(
                f"phase of {longest:g} µs exceeds the {self.max_phase_us:g} µs limit"
            )
        q = waveform.max_phase_charge_nc
        if q > self.max_charge_per_phase_nc:
            raise StimulationSafetyError(
                f"phase charge {q:g} nC exceeds the {self.max_charge_per_phase_nc:g} nC limit"
            )
        if electrode_area_cm2:
            density = q * 1e-3 / electrode_area_cm2  # µC/cm²
            if density > self.max_charge_density_uc_cm2:
                raise StimulationSafetyError(
                    f"charge density {density:.0f} µC/cm² exceeds the "
                    f"{self.max_charge_density_uc_cm2:g} µC/cm² limit"
                )
        if q > 0 and abs(waveform.net_charge_nc) > self.balance_tolerance * q:
            raise StimulationSafetyError(
                f"waveform is not charge-balanced (net {waveform.net_charge_nc:+.4g} nC); "
                "use biphasic_pulse() or add a recovery phase"
            )

    def check_rate(self, frequency_hz: float) -> None:
        if frequency_hz > self.max_pulse_rate_hz:
            raise StimulationSafetyError(
                f"pulse rate {frequency_hz:g} Hz exceeds the {self.max_pulse_rate_hz:g} Hz limit"
            )


@dataclass(order=True)
class StimEvent:
    """A pulse scheduled for (or already delivered to) the culture."""

    time_ms: float
    electrodes: tuple[int, ...] = field(compare=False)
    waveform: Waveform = field(compare=False)

    @property
    def charge_nc(self) -> float:
        return self.waveform.effective_charge_nc
