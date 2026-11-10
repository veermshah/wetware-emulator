"""Point-neuron models and the plugin hook for adding your own.

A neuron model owns the membrane state of every cell in the culture and
advances it one time step at a time. The network hands it a total input
current per neuron and gets back a boolean array of which cells spiked.

To add a model, subclass :class:`NeuronModel` and either pass an instance to
``Culture(neuron_model=...)`` or register a factory by name::

    from wetware.neurons import NeuronModel, register_neuron_model

    class MyModel(NeuronModel):
        name = "mine"
        def init_state(self, is_excitatory, rng): ...
        def step(self, current, dt): ...
        def depolarize(self, dv): ...
        @property
        def v(self): ...

    register_neuron_model("mine", MyModel)
    culture = Culture(neuron_model="mine")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class NeuronModel(ABC):
    """Interface every neuron model implements.

    Currents are in the same arbitrary units Izhikevich (2003) uses (roughly
    pA scaled so that ``I ~ 4`` drives tonic firing). Voltages are in mV.
    """

    #: Short identifier used in metadata and the model registry.
    name: str = "base"

    @abstractmethod
    def init_state(self, is_excitatory: BoolArray, rng: np.random.Generator) -> None:
        """Allocate per-neuron parameters and state.

        Args:
            is_excitatory: Boolean mask, one entry per neuron.
            rng: Generator to draw heterogeneous parameters from.
        """

    @abstractmethod
    def step(self, current: FloatArray, dt: float) -> BoolArray:
        """Advance every neuron by ``dt`` milliseconds.

        Args:
            current: Total input current per neuron for this step.
            dt: Step size in ms.

        Returns:
            Boolean mask of neurons that spiked during the step.
        """

    @abstractmethod
    def depolarize(self, dv: FloatArray) -> None:
        """Instantly shift membrane potentials by ``dv`` mV (used for stimulation)."""

    @property
    @abstractmethod
    def v(self) -> FloatArray:
        """Current membrane potentials in mV."""

    def silence(self, mask: BoolArray) -> None:  # noqa: B027 - optional hook
        """Permanently silence the neurons in ``mask`` (cell death).

        The default does nothing; the network also masks dead cells' output.
        """

    def get_state(self) -> dict[str, Any]:
        """Return a copy of the mutable state (for snapshots)."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: B027 - optional hook
        """Restore state produced by :meth:`get_state`."""


class Izhikevich(NeuronModel):
    """Izhikevich (2003) simple model with heterogeneous RS/FS cells.

    Excitatory cells are regular spiking with a random shift towards
    chattering; inhibitory cells are fast spiking. ``v`` is integrated in
    0.5 ms sub-steps for numerical stability regardless of ``dt``.
    """

    name = "izhikevich"

    def __init__(self, substep_ms: float = 0.5) -> None:
        self.substep_ms = substep_ms
        self._v: FloatArray = np.zeros(0)
        self.u: FloatArray = np.zeros(0)
        self._buf: FloatArray = np.zeros(0)
        self._h = 0.0
        self._ha: FloatArray | None = None
        self._hab: FloatArray = np.zeros(0)

    def init_state(self, is_excitatory: BoolArray, rng: np.random.Generator) -> None:
        n = is_excitatory.size
        r = rng.random(n)
        exc = is_excitatory
        self.a = np.where(exc, 0.02, 0.02 + 0.08 * r)
        self.b = np.where(exc, 0.2, 0.25 - 0.05 * r)
        self.c = np.where(exc, -65.0 + 15.0 * r**2, -65.0)
        self.d = np.where(exc, 8.0 - 6.0 * r**2, 2.0)
        self._v = np.full(n, -65.0) + rng.normal(0.0, 3.0, n)
        self.u = self.b * self._v
        self._alive = np.ones(n, dtype=bool)

    def step(self, current: FloatArray, dt: float) -> BoolArray:
        n_sub = max(1, int(round(dt / self.substep_ms)))
        h = dt / n_sub
        v, u = self._v, self.u
        if self._buf.size != v.size:
            self._buf = np.empty_like(v)
            self._ha = None
        if self._ha is None or self._h != h:
            self._h, self._hab = h, h * self.a * self.b
            self._ha = h * self.a
        ha = self._ha
        buf = self._buf
        fired = np.zeros(v.size, dtype=bool)
        for _ in range(n_sub):
            # v += h * (0.04 v^2 + 5 v + 140 - u + I), written in place.
            np.multiply(v, 0.04, out=buf)
            buf += 5.0
            buf *= v
            buf += 140.0
            buf -= u
            buf += current
            buf *= h
            v += buf
            # u += h a (b v - u)
            np.multiply(self._hab, v, out=buf)
            buf -= ha * u
            u += buf
            spk = v >= 30.0
            if spk.any():
                fired |= spk
                v[spk] = self.c[spk]
                u[spk] += self.d[spk]
        np.clip(v, -100.0, 30.0, out=v)
        fired &= self._alive
        return fired

    def depolarize(self, dv: FloatArray) -> None:
        self._v += dv

    @property
    def v(self) -> FloatArray:
        return self._v

    def silence(self, mask: BoolArray) -> None:
        self._alive &= ~mask

    def get_state(self) -> dict[str, Any]:
        return {"v": self._v.copy(), "u": self.u.copy(), "alive": self._alive.copy()}

    def set_state(self, state: dict[str, Any]) -> None:
        self._v = state["v"].copy()
        self.u = state["u"].copy()
        self._alive = state["alive"].copy()


class LIF(NeuronModel):
    """Leaky integrate-and-fire with spike-frequency adaptation.

    ``tau_m dv/dt = -(v - v_rest) + R (I - w)``, with an adaptation current
    ``w`` that jumps by ``b`` at each spike and decays with ``tau_w``.
    Cheaper than Izhikevich and useful as a baseline.
    """

    name = "lif"

    def __init__(
        self,
        tau_m: float = 20.0,
        v_rest: float = -65.0,
        v_thresh: float = -50.0,
        v_reset: float = -65.0,
        resistance: float = 5.0,
        refractory_ms: float = 2.0,
        tau_w: float = 300.0,
        b: float = 0.6,
    ) -> None:
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.resistance = resistance
        self.refractory_ms = refractory_ms
        self.tau_w = tau_w
        self.b = b
        self._v: FloatArray = np.zeros(0)

    def init_state(self, is_excitatory: BoolArray, rng: np.random.Generator) -> None:
        n = is_excitatory.size
        self._v = self.v_rest + rng.normal(0.0, 3.0, n)
        self.w = np.zeros(n)
        # Inhibitory cells adapt little, like fast-spiking interneurons.
        self.b_vec = np.where(is_excitatory, self.b, 0.05 * self.b)
        self.refr = np.zeros(n)
        self._alive = np.ones(n, dtype=bool)

    def step(self, current: FloatArray, dt: float) -> BoolArray:
        active = self.refr <= 0.0
        dv = (-(self._v - self.v_rest) + self.resistance * (current - self.w)) * (dt / self.tau_m)
        self._v += np.where(active, dv, 0.0)
        self.w -= self.w * (dt / self.tau_w)
        self.refr -= dt
        fired = (self._v >= self.v_thresh) & self._alive
        if fired.any():
            self._v[fired] = self.v_reset
            self.w[fired] += self.b_vec[fired]
            self.refr[fired] = self.refractory_ms
        np.clip(self._v, -100.0, self.v_thresh, out=self._v)
        return fired

    def depolarize(self, dv: FloatArray) -> None:
        self._v += np.where(self.refr <= 0.0, dv, 0.0)

    @property
    def v(self) -> FloatArray:
        return self._v

    def silence(self, mask: BoolArray) -> None:
        self._alive &= ~mask

    def get_state(self) -> dict[str, Any]:
        return {
            "v": self._v.copy(),
            "w": self.w.copy(),
            "refr": self.refr.copy(),
            "alive": self._alive.copy(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._v = state["v"].copy()
        self.w = state["w"].copy()
        self.refr = state["refr"].copy()
        self._alive = state["alive"].copy()


_REGISTRY: dict[str, Callable[[], NeuronModel]] = {
    "izhikevich": Izhikevich,
    "lif": LIF,
}


def register_neuron_model(name: str, factory: Callable[[], NeuronModel]) -> None:
    """Register a neuron model factory under ``name``.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    key = name.lower()
    if key in _REGISTRY:
        raise ValueError(f"neuron model {name!r} is already registered")
    _REGISTRY[key] = factory


def available_neuron_models() -> list[str]:
    """Names of all registered neuron models."""
    return sorted(_REGISTRY)


def make_neuron_model(spec: str | NeuronModel) -> NeuronModel:
    """Resolve a model name or pass through an instance."""
    if isinstance(spec, NeuronModel):
        return spec
    try:
        return _REGISTRY[spec.lower()]()
    except KeyError:
        raise ValueError(
            f"unknown neuron model {spec!r}; available: {available_neuron_models()}"
        ) from None
