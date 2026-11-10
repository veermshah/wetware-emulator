"""Run many cultures in parallel for statistics.

Each condition is a dict of :class:`~wetware.Culture` keyword arguments plus
optional ``age_days`` (aged before the experiment starts). The experiment is
a top-level function ``experiment(culture, **kwargs) -> dict`` so it can be
sent to worker processes::

    from wetware.batch import run_batch, grid, spontaneous_activity

    conditions = grid(seed=range(8), div=[14, 21, 35])
    rows = run_batch(spontaneous_activity, conditions, processes=8, duration_ms=10_000)
"""

from __future__ import annotations

import csv
import itertools
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .culture import Culture
from .mea import MEA

Experiment = Callable[..., dict[str, Any]]


def grid(**axes: Iterable[Any]) -> list[dict[str, Any]]:
    """Cartesian product of condition axes, e.g. ``grid(seed=range(4), div=[14, 28])``."""
    keys = list(axes)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*axes.values())]


def make_culture(condition: dict[str, Any]) -> Culture:
    """Build (and optionally pre-age) a culture from a condition dict."""
    cond = dict(condition)
    age_days = float(cond.pop("age_days", 0.0))
    mea_spec = cond.pop("mea", None)
    mea = MEA(**mea_spec) if isinstance(mea_spec, dict) else mea_spec
    c = Culture(mea=mea, **cond)
    if age_days:
        c.age(age_days)
    return c


def spontaneous_activity(
    culture: Culture, duration_ms: float = 10_000.0, warmup_ms: float = 1000.0
) -> dict[str, Any]:
    """Default experiment: record spontaneous activity and summarise it."""
    from .analysis import summarize

    if warmup_ms:
        culture.advance(warmup_ms)
    rec = culture.record(duration_ms)
    return {**summarize(rec), "alive_fraction": culture.health.alive_fraction}


def _run_one(args: tuple[Experiment, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    experiment, condition, kwargs = args
    t0 = time.perf_counter()
    culture = make_culture(condition)
    out = experiment(culture, **kwargs)
    return {**condition, **out, "wall_s": time.perf_counter() - t0}


def run_batch(
    experiment: Experiment,
    conditions: Sequence[dict[str, Any]],
    processes: int | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run ``experiment`` once per condition, in parallel processes.

    Args:
        experiment: Picklable top-level function taking a culture.
        conditions: Culture keyword arguments per run (plus ``age_days``).
        processes: Worker count; ``None`` uses all CPUs, ``1`` runs inline.
        **kwargs: Passed to every ``experiment`` call.

    Returns:
        One row per condition (condition keys + experiment outputs), in order.
    """
    jobs = [(experiment, dict(c), kwargs) for c in conditions]
    n = processes or os.cpu_count() or 1
    if n <= 1 or len(jobs) <= 1:
        return [_run_one(j) for j in jobs]
    # Keep each worker single-threaded so processes do not fight over BLAS threads.
    env_keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in env_keys}
    for k in env_keys:
        os.environ[k] = "1"
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(n, len(jobs)), mp_context=ctx) as ex:
            return list(ex.map(_run_one, jobs))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def aggregate(
    rows: list[dict[str, Any]], by: str, metric: str
) -> dict[Any, tuple[float, float, int]]:
    """Mean, standard error and count of ``metric`` grouped by ``by``."""
    groups: dict[Any, list[float]] = {}
    for r in rows:
        groups.setdefault(r[by], []).append(float(r[metric]))
    out = {}
    for k, v in sorted(groups.items()):
        a = np.asarray(v)
        sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
        out[k] = (float(a.mean()), sem, int(a.size))
    return out


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write batch rows to CSV (columns are the union of all keys)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        cols.extend(k for k in r if k not in cols)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return p
