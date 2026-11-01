"""Analysis helpers for :class:`~wetware.recording.SpikeTrains`.

Everything here works on detected electrode spikes, so it applies equally to
simulated recordings and to real data loaded into a ``SpikeTrains``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .recording import SpikeTrains

FloatArray = NDArray[np.float64]


@dataclass
class Burst:
    """One network burst."""

    start_ms: float
    stop_ms: float
    n_spikes: int
    n_electrodes: int
    peak_rate_hz: float

    @property
    def duration_ms(self) -> float:
        return self.stop_ms - self.start_ms


def detect_bursts(
    rec: SpikeTrains,
    bin_ms: float = 10.0,
    threshold: float | None = None,
    min_electrodes: int = 4,
    merge_gap_ms: float = 100.0,
) -> list[Burst]:
    """Detect network bursts from the array-wide spike rate.

    A burst is a run of bins where the population rate exceeds
    ``threshold`` (default: ``max(4 x median, median + 4 x MAD-sigma)``),
    with runs closer than ``merge_gap_ms`` merged, and at least
    ``min_electrodes`` electrodes taking part.
    """
    counts = rec.binned(bin_ms)
    pop = counts.sum(axis=0)
    if pop.size == 0 or pop.sum() == 0:
        return []
    if threshold is None:
        med = float(np.median(pop))
        sigma = float(np.median(np.abs(pop - med)) / 0.6745)
        threshold = max(4.0 * med, med + 4.0 * sigma, 3.0)
    above = pop > threshold
    if not above.any():
        return []
    edges = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    gap_bins = int(round(merge_gap_ms / bin_ms))
    merged: list[list[int]] = []
    for s, e in zip(starts, stops, strict=True):
        if merged and s - merged[-1][1] <= gap_bins:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    bursts = []
    for s, e in merged:
        seg = counts[:, s:e]
        n_el = int((seg.sum(axis=1) > 0).sum())
        if n_el < min_electrodes:
            continue
        bursts.append(
            Burst(
                start_ms=rec.t_start_ms + s * bin_ms,
                stop_ms=rec.t_start_ms + e * bin_ms,
                n_spikes=int(seg.sum()),
                n_electrodes=n_el,
                peak_rate_hz=float(pop[s:e].max() / (bin_ms / 1000.0)),
            )
        )
    return bursts


def burst_statistics(bursts: list[Burst], duration_ms: float) -> dict[str, float]:
    """Summary numbers for a list of bursts."""
    if not bursts:
        return {"rate_hz": 0.0, "mean_duration_ms": 0.0, "mean_spikes": 0.0, "mean_ibi_ms": 0.0}
    starts = np.array([b.start_ms for b in bursts])
    return {
        "rate_hz": len(bursts) / (duration_ms / 1000.0),
        "mean_duration_ms": float(np.mean([b.duration_ms for b in bursts])),
        "mean_spikes": float(np.mean([b.n_spikes for b in bursts])),
        "mean_ibi_ms": float(np.diff(starts).mean()) if starts.size > 1 else 0.0,
    }


def firing_rate_map(rec: SpikeTrains) -> FloatArray:
    """Mean detected firing rate per electrode on the ``(rows, cols)`` grid.

    Dead electrodes are ``NaN``.
    """
    rates = rec.firing_rates_hz.astype(float)
    rates[list(rec.dead_electrodes)] = np.nan
    return rates.reshape(rec.grid_shape)


def psth(
    rec: SpikeTrains,
    event_times_ms: FloatArray | None = None,
    window_ms: tuple[float, float] = (-50.0, 250.0),
    bin_ms: float = 5.0,
    electrodes: list[int] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Post-stimulus time histogram.

    Args:
        rec: Recording.
        event_times_ms: Trigger times; defaults to the recorded stimulus pulses.
        window_ms: ``(before, after)`` window around each event.
        bin_ms: Bin width.
        electrodes: Restrict to these electrodes (default: all).

    Returns:
        ``(bin_centres_ms, rate_hz)`` where rate is the mean per-electrode
        rate across events.
    """
    events = rec.stim.times_ms if event_times_ms is None else np.asarray(event_times_ms)
    els = range(rec.n_electrodes) if electrodes is None else electrodes
    edges = np.arange(window_ms[0], window_ms[1] + bin_ms / 2, bin_ms)
    centres = (edges[:-1] + edges[1:]) / 2
    if len(events) == 0:
        return centres, np.zeros(centres.size)
    hist = np.zeros(centres.size)
    n_el = 0
    for e in els:
        t = rec.times[e]
        n_el += 1
        if t.size == 0:
            continue
        rel = (t[None, :] - events[:, None]).ravel()
        hist += np.histogram(rel, edges)[0]
    rate = hist / (len(events) * max(n_el, 1) * bin_ms / 1000.0)
    return centres, rate


def evoked_response(
    rec: SpikeTrains,
    event_times_ms: FloatArray | None = None,
    window_ms: tuple[float, float] = (5.0, 50.0),
    electrodes: list[int] | None = None,
) -> FloatArray:
    """Spike count in ``window_ms`` after each event, summed over electrodes."""
    events = rec.stim.times_ms if event_times_ms is None else np.asarray(event_times_ms)
    els = range(rec.n_electrodes) if electrodes is None else electrodes
    out = np.zeros(len(events))
    for e in els:
        t = rec.times[e]
        if t.size == 0:
            continue
        lo = np.searchsorted(t, events + window_ms[0])
        hi = np.searchsorted(t, events + window_ms[1])
        out += hi - lo
    return out


def cross_correlogram(
    a: FloatArray, b: FloatArray, max_lag_ms: float = 50.0, bin_ms: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Histogram of ``b - a`` spike time differences within ``max_lag_ms``."""
    edges = np.arange(-max_lag_ms, max_lag_ms + bin_ms / 2, bin_ms)
    centres = (edges[:-1] + edges[1:]) / 2
    if a.size == 0 or b.size == 0:
        return centres, np.zeros(centres.size)
    b = np.sort(b)
    lo = np.searchsorted(b, a - max_lag_ms)
    hi = np.searchsorted(b, a + max_lag_ms)
    diffs = np.concatenate([b[i:j] - t for t, i, j in zip(a, lo, hi, strict=True)]) if a.size else []
    return centres, np.histogram(diffs, edges)[0].astype(float)


def functional_connectivity(
    rec: SpikeTrains,
    bin_ms: float = 5.0,
    max_lag_bins: int = 4,
    min_spikes: int = 10,
) -> FloatArray:
    """Pairwise functional connectivity from binned cross-correlation.

    For every electrode pair the binned spike counts are z-scored and the
    peak normalised cross-correlation within ``+-max_lag_bins`` is taken.
    Electrodes with fewer than ``min_spikes`` spikes get ``NaN`` rows.

    Returns:
        ``(n_electrodes, n_electrodes)`` symmetric matrix in ``[-1, 1]``.
    """
    x = rec.binned(bin_ms)
    n_el, n_bins = x.shape
    ok = x.sum(axis=1) >= min_spikes
    z = np.zeros_like(x)
    mu = x[ok].mean(axis=1, keepdims=True)
    sd = x[ok].std(axis=1, keepdims=True)
    z[ok] = (x[ok] - mu) / np.where(sd > 0, sd, 1.0)
    best = np.full((n_el, n_el), -np.inf)
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        if lag >= 0:
            c = z[:, : n_bins - lag] @ z[:, lag:].T / max(n_bins - lag, 1)
        else:
            c = z[:, -lag:] @ z[:, : n_bins + lag].T / max(n_bins + lag, 1)
        best = np.maximum(best, c)
    fc = np.maximum(best, best.T)
    fc[~ok, :] = np.nan
    fc[:, ~ok] = np.nan
    np.fill_diagonal(fc, np.where(ok, 1.0, np.nan))
    return fc


def summarize(rec: SpikeTrains) -> dict[str, float]:
    """One-line numeric summary of a recording, handy for batch statistics."""
    bursts = detect_bursts(rec)
    stats = burst_statistics(bursts, rec.duration_ms)
    rates = rec.firing_rates_hz
    alive = np.ones(rec.n_electrodes, dtype=bool)
    alive[list(rec.dead_electrodes)] = False
    return {
        "mean_rate_hz": float(rates[alive].mean()) if alive.any() else 0.0,
        "active_electrodes": float(len(rec.active_electrodes)),
        "burst_rate_hz": stats["rate_hz"],
        "mean_burst_duration_ms": stats["mean_duration_ms"],
        "burst_fraction": float(
            sum(b.n_spikes for b in bursts) / max(int(rec.counts.sum()), 1)
        ),
    }

