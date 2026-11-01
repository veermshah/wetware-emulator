"""Declarative experiment protocols (YAML or dict).

A protocol describes the culture, what to record, and a list of blocks that
run one after another on the same culture. ``wetware run protocol.yaml``
executes it and writes an HDF5 recording plus a summary figure.

Example::

    name: frequency-sweep
    culture: {n_neurons: 1000, seed: 1, div: 21, mea: {rows: 8, cols: 8}}
    record: {raw: false, ground_truth: false}
    blocks:
      - {type: rest, duration_ms: 2000}
      - {type: train, electrodes: [27], amplitude_ua: 10, n_pulses: 20,
         frequency_hz: 5, tail_ms: 1000}
      - {type: ramp, electrodes: [27], start_ua: 2, stop_ua: 20, steps: 10,
         interval_ms: 500}
      - type: closed_loop
        hz: 20
        duration_ms: 5000
        rule: {type: threshold, watch: [10, 11], threshold: 3,
               stimulate: [27], amplitude_ua: 8}
      - {type: age, days: 7}
      - type: repeat
        times: 3
        blocks:
          - {type: pulse, electrodes: [12], amplitude_ua: 12, tail_ms: 500}

Block types
-----------
``rest``         record ``duration_ms`` with no stimulation.
``pulse``        one biphasic pulse, then record ``tail_ms``.
``train``        a pulse train, recorded until ``tail_ms`` after the last pulse.
``ramp``         one pulse per step with amplitude from ``start_ua`` to ``stop_ua``.
``paired_pulse`` two pulses ``isi_ms`` apart, then ``tail_ms``.
``closed_loop``  run a closed loop at ``hz`` for ``duration_ms`` using a rule.
``age``          age the culture by ``days`` (no recording).
``set``          change ``plasticity`` (true/false).
``repeat``       run nested ``blocks`` ``times`` times.

Closed-loop rules
-----------------
``threshold``  stimulate ``stimulate`` electrodes when the spike count on the
               ``watch`` electrodes in the last cycle reaches ``threshold``.
``periodic``   stimulate every ``every`` cycles regardless of activity.
``python``     call ``callback: "package.module:function"`` with each frame.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .culture import Culture, LoopFrame
from .mea import MEA
from .recording import SpikeTrains
from .stimulation import PulseTrain, Waveform, biphasic_pulse


class ProtocolError(ValueError):
    """The protocol file is malformed."""


BLOCK_TYPES = {
    "rest",
    "pulse",
    "train",
    "ramp",
    "paired_pulse",
    "closed_loop",
    "age",
    "set",
    "repeat",
}


@dataclass
class Epoch:
    """Time span of one executed block."""

    name: str
    type: str
    start_ms: float
    stop_ms: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolResult:
    """Output of :func:`run_protocol`."""

    name: str
    recording: SpikeTrains
    epochs: list[Epoch]
    summary: dict[str, Any]
    hdf5_path: Path | None = None
    figure_path: Path | None = None


def load_protocol(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load and validate a protocol from a YAML path, YAML string or dict."""
    if isinstance(source, dict):
        proto = dict(source)
    else:
        is_path = isinstance(source, Path) or (
            "\n" not in source and source.strip().endswith((".yaml", ".yml"))
        )
        text = Path(source).read_text() if is_path else str(source)
        proto = yaml.safe_load(text)
    if not isinstance(proto, dict):
        raise ProtocolError("a protocol must be a mapping")
    if "blocks" not in proto or not isinstance(proto["blocks"], list) or not proto["blocks"]:
        raise ProtocolError("protocol needs a non-empty 'blocks' list")
    _validate_blocks(proto["blocks"], "blocks")
    return proto


def _validate_blocks(blocks: list[Any], where: str) -> None:
    required = {
        "rest": ["duration_ms"],
        "pulse": ["electrodes", "amplitude_ua"],
        "train": ["electrodes", "amplitude_ua", "n_pulses", "frequency_hz"],
        "ramp": ["electrodes", "start_ua", "stop_ua", "steps", "interval_ms"],
        "paired_pulse": ["electrodes", "amplitude_ua", "isi_ms"],
        "closed_loop": ["hz", "duration_ms", "rule"],
        "age": ["days"],
        "set": [],
        "repeat": ["times", "blocks"],
    }
    for i, b in enumerate(blocks):
        loc = f"{where}[{i}]"
        if not isinstance(b, dict) or "type" not in b:
            raise ProtocolError(f"{loc}: each block needs a 'type'")
        t = b["type"]
        if t not in BLOCK_TYPES:
            raise ProtocolError(f"{loc}: unknown block type {t!r}; use one of {sorted(BLOCK_TYPES)}")
        missing = [k for k in required[t] if k not in b]
        if missing:
            raise ProtocolError(f"{loc} ({t}): missing {missing}")
        if t == "repeat":
            _validate_blocks(b["blocks"], f"{loc}.blocks")
        if t == "closed_loop":
            rt = b["rule"].get("type") if isinstance(b["rule"], dict) else None
            if rt not in ("threshold", "periodic", "python"):
                raise ProtocolError(f"{loc}: rule.type must be threshold, periodic or python")


def build_culture(spec: dict[str, Any]) -> Culture:
    """Create a :class:`Culture` from the ``culture`` section of a protocol."""
    spec = dict(spec or {})
    mea = MEA(**spec.pop("mea", {}))
    return Culture(mea=mea, **spec)


def _pulse(b: dict[str, Any], amp: float | None = None) -> Waveform:
    return biphasic_pulse(
        float(b["amplitude_ua"] if amp is None else amp),
        float(b.get("phase_us", 200.0)),
        float(b.get("interphase_us", 0.0)),
    )


def _make_rule(rule: dict[str, Any]) -> Callable[[LoopFrame], None]:
    kind = rule["type"]
    if kind == "python":
        mod_name, _, fn_name = str(rule["callback"]).partition(":")
        fn: Callable[[LoopFrame], None] = getattr(importlib.import_module(mod_name), fn_name)
        return fn
    pulse = biphasic_pulse(float(rule.get("amplitude_ua", 8.0)), float(rule.get("phase_us", 200.0)))
    targets = list(rule["stimulate"])
    if kind == "threshold":
        watch = list(rule["watch"])
        thr = float(rule["threshold"])

        def threshold_rule(frame: LoopFrame) -> None:
            if frame.counts[watch].sum() >= thr:
                frame.stimulate(targets, pulse)

        return threshold_rule
    every = int(rule.get("every", 1))

    def periodic_rule(frame: LoopFrame) -> None:
        if frame.cycle % every == 0:
            frame.stimulate(targets, pulse)

    return periodic_rule


def _run_blocks(
    culture: Culture,
    blocks: list[dict[str, Any]],
    rec_opts: dict[str, Any],
    parts: list[SpikeTrains],
    epochs: list[Epoch],
    prefix: str = "",
) -> None:
    raw = bool(rec_opts.get("raw", False))
    gt = bool(rec_opts.get("ground_truth", False))

    def rec(duration: float) -> None:
        if duration > 0:
            parts.append(culture.record(duration, raw=raw, ground_truth=gt))

    for i, b in enumerate(blocks):
        t = b["type"]
        name = b.get("name", f"{prefix}{i}:{t}")
        start = culture.time_ms
        if t == "rest":
            rec(float(b["duration_ms"]))
        elif t == "pulse":
            culture.stimulate(b["electrodes"], _pulse(b))
            rec(float(b.get("tail_ms", 500.0)))
        elif t == "train":
            train = PulseTrain(_pulse(b), int(b["n_pulses"]), float(b["frequency_hz"]))
            culture.stimulate(b["electrodes"], train)
            rec(float(train.onsets_ms[-1]) + float(b.get("tail_ms", 500.0)))
        elif t == "ramp":
            amps = np.linspace(float(b["start_ua"]), float(b["stop_ua"]), int(b["steps"]))
            for a in amps:
                culture.stimulate(b["electrodes"], _pulse(b, a))
                rec(float(b["interval_ms"]))
        elif t == "paired_pulse":
            culture.stimulate(b["electrodes"], _pulse(b))
            culture.stimulate(b["electrodes"], _pulse(b), delay_ms=float(b["isi_ms"]))
            rec(float(b["isi_ms"]) + float(b.get("tail_ms", 500.0)))
        elif t == "closed_loop":
            cb = _make_rule(b["rule"])
            with culture.closed_loop(
                cb, hz=float(b["hz"]), latency_ms=float(b.get("latency_ms", 0.0)), raw=raw
            ) as loop:
                loop.run(duration_ms=float(b["duration_ms"]))
            if loop.recording is not None:
                parts.append(loop.recording)
        elif t == "age":
            culture.age(float(b["days"]))
        elif t == "set":
            if "plasticity" in b:
                culture.plasticity = bool(b["plasticity"])
        elif t == "repeat":
            for k in range(int(b["times"])):
                _run_blocks(culture, b["blocks"], rec_opts, parts, epochs, f"{name}#{k}/")
            continue
        params = {k: v for k, v in b.items() if k not in ("type", "name", "blocks")}
        epochs.append(Epoch(name, t, start, culture.time_ms, params))


def run_protocol(
    source: str | Path | dict[str, Any],
    out_dir: str | Path | None = None,
    figure: bool = True,
    culture: Culture | None = None,
) -> ProtocolResult:
    """Execute a protocol.

    Args:
        source: YAML path, YAML text, or an already-parsed dict.
        out_dir: Where to write ``<name>.h5`` and ``<name>.png``. Nothing is
            written when ``None``.
        figure: Also render the summary figure (needs matplotlib).
        culture: Use this culture instead of building one from the protocol.
    """
    from . import analysis

    proto = load_protocol(source)
    name = str(proto.get("name", "protocol"))
    c = culture if culture is not None else build_culture(proto.get("culture", {}))
    parts: list[SpikeTrains] = []
    epochs: list[Epoch] = []
    _run_blocks(c, proto["blocks"], proto.get("record", {}) or {}, parts, epochs)
    if not parts:
        raise ProtocolError("the protocol never recorded anything")
    recording = SpikeTrains.concatenate(parts)
    recording.metadata["protocol"] = proto
    recording.metadata["epochs"] = [e.__dict__ for e in epochs]
    summary: dict[str, Any] = {
        "name": name,
        "duration_ms": recording.duration_ms,
        "n_spikes": int(recording.counts.sum()),
        "n_stim_pulses": len(recording.stim),
        "n_epochs": len(epochs),
        **analysis.summarize(recording),
        "health": c.health.to_dict(),
    }
    result = ProtocolResult(name, recording, epochs, summary)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.hdf5_path = recording.to_hdf5(out / f"{name}.h5", session_description=name)
        if figure:
            from . import plotting

            fig = plotting.summary_figure(recording, title=name)
            ax = fig.axes[0]
            for e in epochs:
                ax.axvline(e.start_ms / 1000.0, color="tab:green", lw=0.6, ls="--", alpha=0.6)
            result.figure_path = out / f"{name}.png"
            fig.savefig(result.figure_path, dpi=110)
            import matplotlib.pyplot as plt

            plt.close(fig)
    return result
