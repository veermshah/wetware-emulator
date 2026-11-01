"""wetware: a simulated cortical culture on a virtual multi-electrode array.

Quick start::

    from wetware import Culture, biphasic_pulse

    culture = Culture(n_neurons=1000, seed=42)
    culture.stimulate([27, 28], biphasic_pulse(amplitude_ua=10.0, phase_us=200))
    rec = culture.record(duration_ms=1000)
    print(rec.firing_rates_hz)
"""

from ._version import __version__
from .culture import (
    AgingParams,
    ClosedLoop,
    Culture,
    Health,
    LoopFrame,
    StimulationParams,
)
from .io import load_hdf5, save_hdf5
from .mea import MEA
from .network import NetworkParams
from .neurons import LIF, Izhikevich, NeuronModel, available_neuron_models, register_neuron_model
from .recorder import RecorderParams
from .recording import SpikeTrains, StimLog
from .stimulation import (
    Phase,
    PulseTrain,
    SafetyLimits,
    StimEvent,
    StimulationSafetyError,
    Waveform,
    asymmetric_biphasic_pulse,
    biphasic_pulse,
    pulse_train,
)

__all__ = [
    "AgingParams",
    "ClosedLoop",
    "Culture",
    "Health",
    "Izhikevich",
    "LIF",
    "LoopFrame",
    "MEA",
    "NetworkParams",
    "NeuronModel",
    "Phase",
    "PulseTrain",
    "RecorderParams",
    "SafetyLimits",
    "SpikeTrains",
    "StimEvent",
    "StimLog",
    "StimulationParams",
    "StimulationSafetyError",
    "Waveform",
    "__version__",
    "asymmetric_biphasic_pulse",
    "available_neuron_models",
    "biphasic_pulse",
    "load_hdf5",
    "pulse_train",
    "register_neuron_model",
    "save_hdf5",
]
