"""Simulation speed benchmark.

The real-time factor (RTF) is simulated time divided by wall-clock time: an
RTF of 10 means one second of culture activity takes 0.1 s to compute.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from .culture import Culture


def measure(
    n_neurons: int = 1000,
    neuron_model: str = "izhikevich",
    duration_ms: float = 5000.0,
    raw: bool = False,
    plasticity: bool = True,
    seed: int = 0,
    repeats: int = 1,
) -> dict[str, Any]:
    """Time one configuration; returns the best of ``repeats`` runs."""
    walls = []
    spikes = 0
    for _ in range(repeats):
        c = Culture(
            n_neurons=n_neurons, seed=seed, neuron_model=neuron_model, plasticity=plasticity
        )
        c.advance(200.0)
        t0 = time.perf_counter()
        rec = c.record(duration_ms, raw=raw, ground_truth=True)
        walls.append(time.perf_counter() - t0)
        spikes = 0 if rec.gt_times_ms is None else int(rec.gt_times_ms.size)
    wall = float(np.min(walls))
    return {
        "n_neurons": n_neurons,
        "model": neuron_model,
        "raw": raw,
        "plasticity": plasticity,
        "sim_s": duration_ms / 1000.0,
        "wall_s": wall,
        "rtf": duration_ms / 1000.0 / wall,
        "mean_rate_hz": spikes / n_neurons / (duration_ms / 1000.0),
    }


def run_suite(
    sizes: Iterable[int] = (250, 500, 1000, 2000),
    models: Iterable[str] = ("izhikevich", "lif"),
    duration_ms: float = 5000.0,
) -> list[dict[str, Any]]:
    rows = [measure(n, m, duration_ms) for m in models for n in sizes]
    rows.append(measure(1000, "izhikevich", duration_ms, plasticity=False))
    rows.append(measure(1000, "izhikevich", duration_ms, raw=True))
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    head = "| neurons | model | STDP | raw | real-time factor | mean rate (Hz) |\n"
    head += "|---:|---|:---:|:---:|---:|---:|\n"
    body = "".join(
        f"| {r['n_neurons']} | {r['model']} | {'on' if r['plasticity'] else 'off'} | "
        f"{'yes' if r['raw'] else 'no'} | {r['rtf']:.1f}x | {r['mean_rate_hz']:.2f} |\n"
        for r in rows
    )
    return head + body


def machine() -> str:
    return f"{platform.machine()}, {platform.system()} {platform.release()}, Python {platform.python_version()}, NumPy {np.__version__}"
