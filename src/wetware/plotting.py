"""Matplotlib helpers (install with ``pip install wetware[plot]``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import analysis
from .recording import SpikeTrains

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _plt() -> Any:
    try:
        import matplotlib

        if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg") and not _interactive():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install 'wetware[plot]'") from exc
    return plt


def _interactive() -> bool:
    import sys

    return hasattr(sys, "ps1") or "ipykernel" in sys.modules


def _ax(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = _plt().subplots(figsize=(8, 3))
    return ax


def raster(rec: SpikeTrains, ax: Axes | None = None, show_stim: bool = True, **kw: Any) -> Axes:
    """Electrode-by-time raster of detected spikes."""
    ax = _ax(ax)
    t, e = rec.all_spikes()
    ax.scatter(t / 1000.0, e, s=kw.pop("s", 1.5), c=kw.pop("c", "k"), marker="|", lw=0.6, **kw)
    if show_stim and len(rec.stim):
        for st in rec.stim.times_ms:
            ax.axvline(st / 1000.0, color="tab:red", lw=0.5, alpha=0.4)
    ax.set_xlim(rec.t_start_ms / 1000.0, rec.t_stop_ms / 1000.0)
    ax.set_ylim(-0.5, rec.n_electrodes - 0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("electrode")
    return ax


def population_rate(rec: SpikeTrains, bin_ms: float = 10.0, ax: Axes | None = None) -> Axes:
    """Array-wide spike rate over time, with detected bursts shaded."""
    ax = _ax(ax)
    pr = rec.population_rate(bin_ms)
    t = rec.t_start_ms + (np.arange(pr.size) + 0.5) * bin_ms
    ax.plot(t / 1000.0, pr / 1000.0, lw=0.8, color="tab:blue")
    for b in analysis.detect_bursts(rec, bin_ms=bin_ms):
        ax.axvspan(b.start_ms / 1000.0, b.stop_ms / 1000.0, color="tab:orange", alpha=0.2, lw=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("array rate (kHz)")
    ax.set_xlim(rec.t_start_ms / 1000.0, rec.t_stop_ms / 1000.0)
    return ax


def rate_map(rec: SpikeTrains, ax: Axes | None = None, cmap: str = "magma") -> Axes:
    """Firing-rate heatmap on the electrode grid (dead electrodes in grey)."""
    if ax is None:
        _, ax = _plt().subplots(figsize=(4, 3.6))
    m = analysis.firing_rate_map(rec)
    masked = np.ma.masked_invalid(m)
    cm = _plt().get_cmap(cmap).with_extremes(bad="0.75")
    im = ax.imshow(masked, cmap=cm, origin="upper")
    ax.figure.colorbar(im, ax=ax, label="rate (Hz)")
    ax.set_xticks(range(m.shape[1]))
    ax.set_yticks(range(m.shape[0]))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    return ax


def psth_plot(
    rec: SpikeTrains,
    ax: Axes | None = None,
    window_ms: tuple[float, float] = (-50.0, 250.0),
    bin_ms: float = 5.0,
    **kw: Any,
) -> Axes:
    """PSTH around recorded stimulus pulses."""
    ax = _ax(ax)
    c, r = analysis.psth(rec, window_ms=window_ms, bin_ms=bin_ms)
    ax.bar(c, r, width=bin_ms, color=kw.pop("color", "tab:purple"), **kw)
    blank = rec.metadata.get("blank_ms")
    ax.axvline(0, color="tab:red", lw=0.8)
    if blank:
        ax.axvspan(0, blank, color="0.8", alpha=0.6, lw=0)
    ax.set_xlabel("time from pulse (ms)")
    ax.set_ylabel("rate per electrode (Hz)")
    return ax


def connectivity_matrix(rec: SpikeTrains, ax: Axes | None = None, **kw: Any) -> Axes:
    """Functional connectivity matrix from :func:`analysis.functional_connectivity`."""
    if ax is None:
        _, ax = _plt().subplots(figsize=(4.2, 3.6))
    fc = analysis.functional_connectivity(rec, **kw)
    cm = _plt().get_cmap("viridis").with_extremes(bad="0.85")
    im = ax.imshow(np.ma.masked_invalid(fc), cmap=cm, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, label="peak xcorr")
    ax.set_xlabel("electrode")
    ax.set_ylabel("electrode")
    return ax


def summary_figure(rec: SpikeTrains, title: str | None = None) -> Figure:
    """Four-panel overview: raster, population rate, rate map, PSTH/connectivity."""
    plt = _plt()
    fig = plt.figure(figsize=(11, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    raster(rec, ax=fig.add_subplot(gs[0:2, 0:2]))
    population_rate(rec, ax=fig.add_subplot(gs[2, 0:2]))
    rate_map(rec, ax=fig.add_subplot(gs[0:2, 2]))
    ax = fig.add_subplot(gs[2, 2])
    if len(rec.stim):
        psth_plot(rec, ax=ax)
    else:
        connectivity_matrix(rec, ax=ax)
    if title:
        fig.suptitle(title)
    return fig
