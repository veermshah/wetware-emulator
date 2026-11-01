"""Extracellular signal synthesis and threshold spike detection.

The network produces ground-truth spike times. This module turns them into
what an MEA would actually show you:

1. Each spike adds a biphasic extracellular waveform to every electrode
   within reach, scaled by distance.
2. Each electrode adds its own white thermal noise.
3. Stimulation adds a large saturating artifact that decays exponentially.
4. Samples inside the blanking window after each pulse are zeroed.
5. Spikes are detected as negative threshold crossings at ``-k`` times the
   robust noise estimate ``median(|x|) / 0.6745`` (Quiroga et al., 2004),
   with a dead time after each detection.

Signals are processed in chunks with carry-over so waveforms and artifacts
that straddle a chunk boundary are not lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mea import MEA
from .stimulation import StimEvent

FloatArray = NDArray[np.float64]


@dataclass
class RecorderParams:
    """Parameters of the recording chain."""

    #: Peak extracellular amplitude of a spike right at the electrode (µV).
    spike_amplitude_uv: float = 150.0
    #: Heterogeneity (log-normal sigma) of per-neuron spike amplitude.
    spike_amplitude_sigma: float = 0.3
    #: Distance at which spike amplitude halves (µm).
    pickup_r0_um: float = 30.0
    #: Neurons further than this are invisible to an electrode (µm).
    pickup_max_um: float = 150.0
    #: Detection threshold in multiples of the noise estimate.
    threshold_k: float = 5.0
    #: Dead time after a detection (ms).
    dead_time_ms: float = 1.0
    #: Blanking window after each stimulus pulse, on every electrode (ms).
    blank_ms: float = 3.0
    #: Artifact peak on the stimulated electrode per nC (µV), and amplifier rail.
    artifact_uv_per_nc: float = 2000.0
    amplifier_rail_uv: float = 5000.0
    artifact_tau_ms: float = 0.4
    #: Length constant for artifact spread to other electrodes (µm).
    artifact_spread_um: float = 300.0
    #: Spread of per-electrode noise around the MEA's mean noise.
    noise_heterogeneity: float = 0.2


def spike_template(sample_rate_hz: float) -> FloatArray:
    """Normalised extracellular spike waveform (trough = -1), ~2 ms long."""
    t = np.arange(0.0, 2.0, 1000.0 / sample_rate_hz)
    w = -np.exp(-((t - 0.3) ** 2) / (2 * 0.08**2)) + 0.3 * np.exp(
        -((t - 0.75) ** 2) / (2 * 0.25**2)
    )
    return w / -w.min()


class Recorder:
    """Stateful recording chain for one culture/MEA pair."""

    def __init__(
        self,
        mea: MEA,
        neuron_positions: FloatArray,
        params: RecorderParams,
        rng: np.random.Generator,
    ) -> None:
        self.mea = mea
        self.params = p = params
        self.rng = rng
        self.fs = mea.sample_rate_hz
        self.template = spike_template(self.fs)

        el = mea.positions
        d = np.sqrt(((el[:, None, :] - neuron_positions[None, :, :]) ** 2).sum(-1))
        amp = p.spike_amplitude_uv * rng.lognormal(0.0, p.spike_amplitude_sigma, d.shape[1])
        gain = amp[None, :] / (1.0 + (d / p.pickup_r0_um) ** 2)
        gain[d > p.pickup_max_um] = 0.0
        self.gain = gain  # (n_electrodes, n_neurons), µV
        self.noise_rms = mea.noise_rms_uv * np.clip(
            rng.normal(1.0, p.noise_heterogeneity, mea.n_electrodes), 0.5, 2.0
        )
        de = np.sqrt(((el[:, None, :] - el[None, :, :]) ** 2).sum(-1))
        self.artifact_coupling = 1.0 / (1.0 + (de / p.artifact_spread_um) ** 2)

        self._tail = np.zeros((mea.n_electrodes, 0))
        self._last_sample = np.zeros(mea.n_electrodes)
        self._last_detect_ms = np.full(mea.n_electrodes, -np.inf)
        self._blank_until_ms = -np.inf
        self.threshold_uv = -p.threshold_k * self.noise_rms

    def visible_neurons(self, electrode: int, min_uv: float | None = None) -> NDArray[np.int64]:
        """Neurons whose spikes on ``electrode`` exceed ``min_uv`` (default: threshold)."""
        lim = abs(self.threshold_uv[electrode]) if min_uv is None else min_uv
        return np.flatnonzero(self.gain[electrode] >= lim)

    def process(
        self,
        t0_ms: float,
        duration_ms: float,
        spike_neurons: NDArray[np.int64],
        spike_times_ms: FloatArray,
        stim_events: list[StimEvent],
        keep_raw: bool = False,
    ) -> tuple[list[FloatArray], FloatArray | None]:
        """Synthesize and analyse one chunk.

        Returns:
            Per-electrode detected spike times (ms, absolute) and, if
            ``keep_raw``, the ``(n_electrodes, n_samples)`` trace in µV.
        """
        p = self.params
        n_el = self.mea.n_electrodes
        fs_ms = self.fs / 1000.0
        n = int(round(duration_ms * fs_ms))
        tlen = self.template.size
        art_len = int(np.ceil(10 * p.artifact_tau_ms * fs_ms)) + 1
        extra = max(tlen, art_len + int(np.ceil(fs_ms)))
        buf = np.zeros((n_el, n + extra))
        carry = min(self._tail.shape[1], buf.shape[1])
        buf[:, :carry] += self._tail[:, :carry]

        # 1. Spike waveforms: impulse train per electrode, then convolve.
        if spike_neurons.size:
            idx = np.floor((spike_times_ms - t0_ms) * fs_ms).astype(np.int64)
            ok = (idx >= 0) & (idx < n)
            idx, nrn = idx[ok], spike_neurons[ok]
            imp = np.zeros((n_el, n))
            g = self.gain[:, nrn]
            rows = np.nonzero(g.any(axis=1))[0]
            for e in rows:
                np.add.at(imp[e], idx, g[e])
                buf[e, : n + tlen - 1] += np.convolve(imp[e], self.template)

        # 2. Stimulation artifacts.
        blank = np.zeros(n + extra, dtype=bool)
        if self._blank_until_ms > t0_ms:
            blank[: int(np.ceil((self._blank_until_ms - t0_ms) * fs_ms))] = True
        decay = np.exp(-np.arange(art_len) / (p.artifact_tau_ms * fs_ms))
        for ev in stim_events:
            s = int(np.floor((ev.time_ms - t0_ms) * fs_ms))
            if not 0 <= s < n:
                continue
            on = max(1, int(np.ceil(ev.waveform.duration_us * 1e-3 * fs_ms)))
            peak = min(p.amplifier_rail_uv, p.artifact_uv_per_nc * ev.charge_nc)
            coupling = self.artifact_coupling[list(ev.electrodes)].max(axis=0)
            e_amp = peak * coupling
            seg = np.concatenate([np.full(on, -1.0), decay])  # saturate, then decay
            end = min(s + seg.size, buf.shape[1])
            buf[:, s:end] += e_amp[:, None] * seg[None, : end - s]
            b_end = int(np.ceil((ev.time_ms + p.blank_ms - t0_ms) * fs_ms))
            blank[s : min(b_end, blank.size)] = True
            self._blank_until_ms = max(self._blank_until_ms, ev.time_ms + p.blank_ms)
        np.clip(buf, -p.amplifier_rail_uv, p.amplifier_rail_uv, out=buf)

        # Carry the overhang into the next chunk.
        self._tail = buf[:, n:].copy()
        x = buf[:, :n]

        # 3. Noise; dead electrodes record only a small residual noise.
        x += self.rng.standard_normal((n_el, n)) * self.noise_rms[:, None]
        alive = self.mea.alive_mask
        if not alive.all():
            x[~alive] = self.rng.standard_normal((int((~alive).sum()), n)) * 0.5

        # 4. Blanking.
        bl = blank[:n]
        x[:, bl] = 0.0

        # 5. Threshold detection.
        unblanked = ~bl
        if n >= int(0.1 * self.fs) and unblanked.sum() > 0.5 * n:
            sigma = np.median(np.abs(x[:, unblanked]), axis=1) / 0.6745
            self.threshold_uv = -p.threshold_k * np.maximum(sigma, 1e-3)
        thr = self.threshold_uv[:, None]
        prev = np.concatenate([self._last_sample[:, None], x[:, :-1]], axis=1)
        cross = (x < thr) & (prev >= thr)
        cross[~alive] = False
        self._last_sample = x[:, -1].copy()

        times: list[FloatArray] = []
        dead = p.dead_time_ms
        for e in range(n_el):
            ci = np.flatnonzero(cross[e])
            if ci.size == 0:
                times.append(np.empty(0))
                continue
            tt = t0_ms + ci / fs_ms
            kept = []
            last = self._last_detect_ms[e]
            for t in tt:
                if t - last >= dead:
                    kept.append(t)
                    last = t
            self._last_detect_ms[e] = last
            times.append(np.asarray(kept))
        return times, (x.astype(np.float32) if keep_raw else None)
