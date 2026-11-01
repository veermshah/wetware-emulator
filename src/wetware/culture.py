"""The simulated culture: the main entry point of the package."""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._version import __version__
from .mea import MEA
from .network import Network, NetworkParams
from .neurons import NeuronModel, make_neuron_model
from .recorder import Recorder, RecorderParams
from .recording import SpikeTrains, StimLog
from .stimulation import (
    PulseTrain,
    SafetyLimits,
    StimEvent,
    StimulationSafetyError,
    Waveform,
)

FloatArray = NDArray[np.float64]


@dataclass
class StimulationParams:
    """How extracellular pulses depolarize nearby neurons.

    A pulse with effective charge ``Q`` (nC) on electrode ``e`` shifts the
    membrane potential of neuron ``i`` by
    ``gain_mv_per_nc * Q * excitability_i / (1 + (d_ei / r0_um)^2)`` mV,
    with log-normal trial-to-trial jitter.
    """

    gain_mv_per_nc: float = 30.0
    r0_um: float = 90.0
    max_distance_um: float = 400.0
    excitability_sigma: float = 0.3
    trial_jitter_sigma: float = 0.2


@dataclass
class AgingParams:
    """Daily drift of culture health. See ``docs/model.md``."""

    #: Day the culture is "mature": weights saturate around here.
    maturation_div: float = 18.0
    maturation_width: float = 3.0
    #: After this day, background drive and synaptic weights slowly decline.
    decline_onset_div: float = 35.0
    drive_decay_days: float = 45.0
    weight_decay_days: float = 120.0
    #: Daily probability of neuron death (base, plus increase per day past onset).
    neuron_death_base: float = 0.002
    neuron_death_slope: float = 0.0004
    #: Daily probability of electrode failure (base, plus increase per day past onset).
    electrode_failure_base: float = 0.002
    electrode_failure_slope: float = 0.0002


@dataclass
class Health:
    """Snapshot of the culture's condition."""

    div: float
    alive_fraction: float
    working_electrodes: int
    drive_scale: float
    weight_scale: float
    dead_electrodes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Culture:
    """A 2D dissociated cortical culture grown on a virtual MEA.

    Args:
        n_neurons: Number of neurons (80 % excitatory by default).
        seed: Seed for every random process; equal seeds give identical runs.
        mea: Electrode array. Defaults to an 8x8 array at 200 µm pitch.
        neuron_model: Registered model name (``"izhikevich"``, ``"lif"``) or a
            :class:`~wetware.neurons.NeuronModel` instance.
        dt_ms: Integration step in ms.
        plasticity: Enable STDP on excitatory synapses.
        div: Days in vitro at the start of the simulation.
        margin_um: How far the culture extends beyond the outer electrodes.
        params, recorder_params, stim_params, aging_params: Model parameters.
        safety: Stimulation limits; violations raise
            :class:`~wetware.stimulation.StimulationSafetyError`.

    Example:
        >>> from wetware import Culture, biphasic_pulse
        >>> c = Culture(n_neurons=500, seed=1)
        >>> c.stimulate([27], biphasic_pulse(10.0))
        >>> rec = c.record(500)
    """

    def __init__(
        self,
        n_neurons: int = 1000,
        seed: int | None = 0,
        mea: MEA | None = None,
        neuron_model: str | NeuronModel = "izhikevich",
        dt_ms: float = 0.5,
        plasticity: bool = True,
        div: float = 21.0,
        margin_um: float = 150.0,
        params: NetworkParams | None = None,
        recorder_params: RecorderParams | None = None,
        stim_params: StimulationParams | None = None,
        aging_params: AgingParams | None = None,
        safety: SafetyLimits | None = None,
    ) -> None:
        if n_neurons < 10:
            raise ValueError("n_neurons must be at least 10")
        if not 0.05 <= dt_ms <= 1.0:
            raise ValueError("dt_ms must be between 0.05 and 1.0")
        self.n_neurons = n_neurons
        self.seed = seed
        self.mea = mea if mea is not None else MEA()
        self.dt_ms = float(dt_ms)
        self.params = params or NetworkParams()
        self.recorder_params = recorder_params or RecorderParams()
        self.stim_params = stim_params or StimulationParams()
        self.aging_params = aging_params or AgingParams()
        self.safety = safety or SafetyLimits()
        self._div = float(div)

        ss = np.random.SeedSequence(seed)
        r_build, r_dyn, r_rec, r_age, r_stim = (np.random.default_rng(s) for s in ss.spawn(5))
        self._rng_age = r_age
        self._rng_stim = r_stim

        w, h = self.mea.extent_um
        half = np.array([w / 2 + margin_um, h / 2 + margin_um])
        self.positions = r_build.uniform(-half, half, (n_neurons, 2))

        self.model = make_neuron_model(neuron_model)
        self.network = Network(
            self.positions, self.model, self.params, r_build, r_dyn, plasticity=plasticity
        )
        self.recorder = Recorder(self.mea, self.positions, self.recorder_params, r_rec)
        self._jitter_rng = np.random.default_rng(ss.spawn(1)[0])

        sp = self.stim_params
        d = np.sqrt(((self.mea.positions[:, None, :] - self.positions[None, :, :]) ** 2).sum(-1))
        act = 1.0 / (1.0 + (d / sp.r0_um) ** 2)
        act[d > sp.max_distance_um] = 0.0
        excit = r_build.lognormal(0.0, sp.excitability_sigma, n_neurons)
        self._stim_profile = act * excit[None, :]  # (n_electrodes, n_neurons)

        self._step = 0
        self._queue: list[StimEvent] = []
        self._last_pulse_ms = np.full(self.mea.n_electrodes, -np.inf)
        self._apply_maturation()

    # ------------------------------------------------------------------ state
    @property
    def time_ms(self) -> float:
        """Current simulation time in ms."""
        return self._step * self.dt_ms

    @property
    def div(self) -> float:
        """Current age in days in vitro."""
        return self._div

    @property
    def plasticity(self) -> bool:
        return self.network.plasticity

    @plasticity.setter
    def plasticity(self, on: bool) -> None:
        self.network.plasticity = bool(on)

    @property
    def health(self) -> Health:
        """Current health snapshot."""
        return Health(
            div=self._div,
            alive_fraction=float(self.network.alive.mean()),
            working_electrodes=int(self.mea.alive_mask.sum()),
            drive_scale=float(self.network.drive_scale),
            weight_scale=float(self.network.weight_scale),
            dead_electrodes=self.mea.dead_electrodes,
        )

    @property
    def pending_stimuli(self) -> list[StimEvent]:
        """Pulses scheduled but not delivered yet, in time order."""
        return sorted(self._queue)

    def metadata(self) -> dict[str, Any]:
        """Description of this culture, stored with every recording."""
        return {
            "simulator": f"wetware {__version__}",
            "seed": self.seed,
            "n_neurons": self.n_neurons,
            "n_excitatory": int(self.network.is_exc.sum()),
            "neuron_model": self.model.name,
            "dt_ms": self.dt_ms,
            "plasticity": self.plasticity,
            "blank_ms": self.recorder_params.blank_ms,
            "div": self._div,
            "health": self.health.to_dict(),
            "mea": self.mea.to_dict(),
        }

    def info(self) -> dict[str, Any]:
        """Metadata plus the current simulation time."""
        return {**self.metadata(), "time_ms": self.time_ms}

    # ------------------------------------------------------------ stimulation
    def stimulate(
        self,
        electrodes: int | Sequence[int],
        waveform: Waveform | PulseTrain,
        delay_ms: float = 0.0,
    ) -> list[StimEvent]:
        """Schedule stimulation on one or more electrodes.

        Pulses are delivered when the simulation reaches them, i.e. during the
        next :meth:`record`, :meth:`advance` or closed-loop cycle.

        Args:
            electrodes: Electrode index or indices, stimulated simultaneously.
            waveform: A single pulse or a :class:`PulseTrain`.
            delay_ms: Offset from the current time to the first pulse.

        Returns:
            The scheduled events.

        Raises:
            StimulationSafetyError: The request breaks a safety limit, targets
                a dead electrode, or exceeds the per-electrode pulse rate.
            IndexError: An electrode index is out of range.
        """
        els = tuple(sorted({int(e) for e in np.atleast_1d(np.asarray(electrodes))}))
        if not els:
            raise ValueError("no electrodes given")
        for e in els:
            self.mea._check_index(e)
        dead = set(els) & set(self.mea.dead_electrodes)
        if dead:
            raise StimulationSafetyError(f"electrodes {sorted(dead)} are dead")
        if len(els) > self.safety.max_electrodes_per_pulse:
            raise StimulationSafetyError(
                f"{len(els)} electrodes at once exceeds the limit of "
                f"{self.safety.max_electrodes_per_pulse}"
            )
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")

        if isinstance(waveform, PulseTrain):
            pulse = waveform.pulse
            self.safety.check_rate(waveform.frequency_hz)
            onsets = waveform.onsets_ms + delay_ms
        elif isinstance(waveform, Waveform):
            pulse = waveform
            onsets = np.array([delay_ms])
        else:
            raise TypeError("waveform must be a Waveform or PulseTrain")
        self.safety.check_waveform(pulse, self.mea.electrode_area_cm2)

        t_now = self.time_ms
        times = t_now + onsets
        self._check_pulse_rate(els, times)
        events = [StimEvent(float(t), els, pulse) for t in times]
        for ev in events:
            heapq.heappush(self._queue, ev)
        return events

    def _check_pulse_rate(self, els: tuple[int, ...], new_times: FloatArray) -> None:
        min_gap = 1000.0 / self.safety.max_pulse_rate_hz - 1e-9
        for e in els:
            existing = [ev.time_ms for ev in self._queue if e in ev.electrodes]
            existing.append(float(self._last_pulse_ms[e]))
            allt = np.sort(np.concatenate([np.asarray(existing), new_times]))
            allt = allt[np.isfinite(allt)]
            if allt.size > 1 and np.diff(allt).min() < min_gap:
                raise StimulationSafetyError(
                    f"electrode {e}: pulses closer than {min_gap:.2f} ms "
                    f"(limit {self.safety.max_pulse_rate_hz:g} Hz)"
                )

    def cancel_stimulation(self) -> int:
        """Drop every pending pulse; returns how many were cancelled."""
        n = len(self._queue)
        self._queue.clear()
        return n

    # ------------------------------------------------------------- simulation
    def _simulate(
        self, duration_ms: float
    ) -> tuple[NDArray[np.int64], FloatArray, list[StimEvent]]:
        n_steps = int(round(duration_ms / self.dt_ms))
        dt = self.dt_ms
        net = self.network
        sp = self.stim_params
        spikes_n: list[NDArray[np.int64]] = []
        spikes_k: list[int] = []
        delivered: list[StimEvent] = []
        q = self._queue
        for _ in range(n_steps):
            k = self._step
            t_end = (k + 1) * dt
            dv = None
            while q and q[0].time_ms < t_end - 1e-9:
                ev = heapq.heappop(q)
                prof = self._stim_profile[list(ev.electrodes)].sum(axis=0)
                jitter = self._rng_stim.lognormal(0.0, sp.trial_jitter_sigma, self.n_neurons)
                kick = sp.gain_mv_per_nc * ev.charge_nc * prof * jitter
                dv = kick if dv is None else dv + kick
                self._last_pulse_ms[list(ev.electrodes)] = ev.time_ms
                delivered.append(ev)
            spk = net.step(dt, dv)
            if spk.size:
                spikes_n.append(spk)
                spikes_k.append(k)
            self._step += 1
        if spikes_n:
            neurons = np.concatenate(spikes_n)
            steps = np.repeat(np.asarray(spikes_k, dtype=float), [s.size for s in spikes_n])
            times = (steps + self._jitter_rng.random(steps.size)) * dt
        else:
            neurons = np.empty(0, dtype=np.int64)
            times = np.empty(0)
        return neurons, times, delivered

    def advance(self, duration_ms: float) -> None:
        """Run the culture forward without keeping a recording (e.g. warm-up)."""
        self.record(duration_ms, chunk_ms=max(duration_ms, 1.0), _discard=True)

    def record(
        self,
        duration_ms: float,
        raw: bool = False,
        ground_truth: bool = False,
        chunk_ms: float = 1000.0,
        _discard: bool = False,
    ) -> SpikeTrains:
        """Run the culture for ``duration_ms`` and record from every electrode.

        Args:
            duration_ms: Recording length in ms.
            raw: Also return the synthesized voltage traces (µV, float32).
                Memory use is ``n_electrodes * sample_rate * duration * 4`` bytes.
            ground_truth: Also return every neuron's spike times.
            chunk_ms: Internal processing chunk size.

        Returns:
            A :class:`~wetware.recording.SpikeTrains` with detected spikes.
        """
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        t_start = self.time_ms
        n_el = self.mea.n_electrodes
        el_times: list[list[FloatArray]] = [[] for _ in range(n_el)]
        raws: list[NDArray[np.float32]] = []
        gt_n: list[NDArray[np.int64]] = []
        gt_t: list[FloatArray] = []
        stim_all: list[StimEvent] = []
        remaining_steps = int(round(duration_ms / self.dt_ms))
        chunk_steps = max(1, int(round(chunk_ms / self.dt_ms)))
        while remaining_steps > 0:
            ns = min(chunk_steps, remaining_steps)
            t0 = self.time_ms
            neurons, times, delivered = self._simulate(ns * self.dt_ms)
            stim_all.extend(delivered)
            if _discard:
                remaining_steps -= ns
                continue
            det, x = self.recorder.process(
                t0, ns * self.dt_ms, neurons, times, delivered, keep_raw=raw
            )
            for e in range(n_el):
                el_times[e].append(det[e])
            if raw and x is not None:
                raws.append(x)
            if ground_truth:
                gt_n.append(neurons)
                gt_t.append(times)
            remaining_steps -= ns

        stim = StimLog(
            np.array([ev.time_ms for ev in stim_all], dtype=float),
            [ev.electrodes for ev in stim_all],
            np.array([ev.charge_nc for ev in stim_all], dtype=float),
            np.array([ev.waveform.peak_amplitude_ua for ev in stim_all], dtype=float),
        )
        return SpikeTrains(
            t_start_ms=t_start,
            duration_ms=self.time_ms - t_start,
            times=[np.concatenate(t) if t else np.empty(0) for t in el_times],
            electrode_positions_um=self.mea.positions,
            grid_shape=(self.mea.rows, self.mea.cols),
            dead_electrodes=self.mea.dead_electrodes,
            stim=stim,
            voltage_uv=np.concatenate(raws, axis=1) if raws else None,
            sample_rate_hz=self.mea.sample_rate_hz,
            gt_neurons=np.concatenate(gt_n) if ground_truth and gt_n else None,
            gt_times_ms=np.concatenate(gt_t) if ground_truth and gt_t else None,
            metadata=self.metadata(),
        )

    # ------------------------------------------------------------ closed loop
    @contextmanager
    def closed_loop(
        self,
        callback: Callable[[LoopFrame], Any],
        hz: float,
        latency_ms: float = 0.0,
        realtime: bool = False,
        speed: float = 1.0,
        raw: bool = False,
    ) -> Iterator[ClosedLoop]:
        """Run ``callback`` at ``hz`` on the simulation clock.

        Each cycle records one period of activity, hands it to ``callback``
        as a :class:`LoopFrame`, and lets the callback schedule stimulation
        through ``frame.stimulate``. Pulses requested in a callback are
        delivered ``latency_ms`` after the end of the cycle that produced
        them, which models the round trip of a real closed-loop rig.

        Args:
            callback: Function called once per cycle.
            hz: Loop rate on the simulation clock.
            latency_ms: Delay between decision and stimulation.
            realtime: Pace the loop against the wall clock.
            speed: Wall-clock speed-up factor when ``realtime`` is set.
            raw: Keep raw traces in the final recording.

        Yields:
            A :class:`ClosedLoop` with ``step()`` and ``run()``. After the
            ``with`` block, ``loop.recording`` holds the whole session.

        Example:
            >>> def cb(frame):
            ...     if frame.counts[10] > 2:
            ...         frame.stimulate([20], biphasic_pulse(8.0))
            >>> with culture.closed_loop(cb, hz=50) as loop:
            ...     loop.run(duration_ms=2000)
        """
        loop = ClosedLoop(self, callback, hz, latency_ms, realtime, speed, raw)
        try:
            yield loop
        finally:
            loop._finish()

    # ------------------------------------------------------------------ aging
    def _apply_maturation(self) -> None:
        a = self.aging_params
        mature = 1.0 / (1.0 + np.exp(-(self._div - a.maturation_div) / a.maturation_width))
        past = max(0.0, self._div - a.decline_onset_div)
        self.network.weight_scale = float(mature * np.exp(-past / a.weight_decay_days))
        self.network.drive_scale = float(np.exp(-past / a.drive_decay_days))

    def age(self, days: float) -> Health:
        """Let the culture age by ``days`` without simulating its activity.

        Applies maturation and decline of synaptic strength, decay of the
        background drive (lower firing rates, fewer bursts), random neuron
        death and electrode failures. Synaptic and short-term plasticity
        state is reset as after a long rest.

        Returns:
            The new :class:`Health`.
        """
        if days < 0:
            raise ValueError("days must be non-negative")
        a = self.aging_params
        rng = self._rng_age
        remaining = float(days)
        while remaining > 1e-12:
            step = min(1.0, remaining)
            past = max(0.0, self._div - a.decline_onset_div)
            p_die = (a.neuron_death_base + a.neuron_death_slope * past) * step
            die = (rng.random(self.n_neurons) < p_die) & self.network.alive
            if die.any():
                self.network.kill(die)
            p_fail = (a.electrode_failure_base + a.electrode_failure_slope * past) * step
            fail = np.flatnonzero((rng.random(self.mea.n_electrodes) < p_fail) & self.mea.alive_mask)
            if fail.size:
                self.mea.mark_dead(fail.tolist())
            self._div += step
            remaining -= step
        self._apply_maturation()
        net = self.network
        net.i_exc[:] = 0.0
        net.i_inh[:] = 0.0
        net.stp_x[:] = 1.0
        net.stp_u[:] = net.stp_U
        self._queue.clear()
        return self.health

    def __repr__(self) -> str:
        return (
            f"Culture(n_neurons={self.n_neurons}, seed={self.seed}, "
            f"mea={self.mea.rows}x{self.mea.cols}, div={self._div:g}, t={self.time_ms:.1f} ms)"
        )


@dataclass
class LoopFrame:
    """What a closed-loop callback sees on each cycle.

    Attributes:
        cycle: Zero-based cycle number.
        t_ms: Simulation time at the end of the cycle (the decision time).
        spikes: Detected spikes recorded during this cycle.
        counts: Spike count per electrode during this cycle.
    """

    cycle: int
    t_ms: float
    spikes: SpikeTrains
    counts: NDArray[np.int64]
    _loop: ClosedLoop = field(repr=False)

    def stimulate(
        self, electrodes: int | Sequence[int], waveform: Waveform | PulseTrain, delay_ms: float = 0.0
    ) -> list[StimEvent]:
        """Schedule stimulation after the loop latency (plus ``delay_ms``)."""
        return self._loop.culture.stimulate(
            electrodes, waveform, delay_ms=self._loop.latency_ms + delay_ms
        )

    def stop(self) -> None:
        """Ask the loop to stop after this cycle."""
        self._loop.stopped = True


class ClosedLoop:
    """Closed-loop session created by :meth:`Culture.closed_loop`."""

    def __init__(
        self,
        culture: Culture,
        callback: Callable[[LoopFrame], Any],
        hz: float,
        latency_ms: float,
        realtime: bool,
        speed: float,
        raw: bool,
    ) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        period = 1000.0 / hz
        if period < culture.dt_ms:
            raise ValueError(f"{hz} Hz is faster than the simulation step allows")
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        self.culture = culture
        self.callback = callback
        self.hz = hz
        self.period_ms = period
        self.latency_ms = latency_ms
        self.realtime = realtime
        self.speed = speed
        self.raw = raw
        self.t0_ms = culture.time_ms
        self.cycle = 0
        self.stopped = False
        #: Simulation times at which the callback ran.
        self.call_times_ms: list[float] = []
        #: Wall-clock seconds spent inside the callback per cycle.
        self.callback_wall_s: list[float] = []
        #: Cycles where the simulation fell behind the wall clock (realtime mode).
        self.overruns = 0
        self._parts: list[SpikeTrains] = []
        self._wall0 = time.perf_counter()
        self.recording: SpikeTrains | None = None

    def _target_ms(self, cycle: int) -> float:
        dt = self.culture.dt_ms
        return self.t0_ms + round(cycle * self.period_ms / dt) * dt

    def step(self) -> LoopFrame:
        """Run one cycle: record a period, then call the callback."""
        c = self.culture
        end = self._target_ms(self.cycle + 1)
        dur = end - c.time_ms
        if dur <= 0:  # period shorter than dt after rounding; skip ahead one step
            dur = c.dt_ms
        part = c.record(dur, raw=self.raw)
        self._parts.append(part)
        frame = LoopFrame(self.cycle, c.time_ms, part, part.counts, self)
        w0 = time.perf_counter()
        self.callback(frame)
        self.callback_wall_s.append(time.perf_counter() - w0)
        self.call_times_ms.append(c.time_ms)
        self.cycle += 1
        if self.realtime:
            due = self._wall0 + (c.time_ms - self.t0_ms) / 1000.0 / self.speed
            lag = due - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            else:
                self.overruns += 1
        return frame

    def run(self, duration_ms: float | None = None, cycles: int | None = None) -> None:
        """Run for a duration (sim ms) or a number of cycles, or until stopped."""
        if duration_ms is None and cycles is None:
            raise ValueError("give duration_ms or cycles")
        if cycles is None:
            assert duration_ms is not None
            cycles = int(round(duration_ms / self.period_ms))
        for _ in range(cycles):
            if self.stopped:
                break
            self.step()

    @property
    def timing_error_ms(self) -> float:
        """Largest deviation of callback times from the ideal schedule."""
        if not self.call_times_ms:
            return 0.0
        ideal = self.t0_ms + self.period_ms * np.arange(1, len(self.call_times_ms) + 1)
        return float(np.abs(np.asarray(self.call_times_ms) - ideal).max())

    def _finish(self) -> None:
        if self._parts:
            self.recording = SpikeTrains.concatenate(self._parts)
