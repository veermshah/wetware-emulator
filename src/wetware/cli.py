"""Command-line interface: ``wetware <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__


def _parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cmd_demo(a: argparse.Namespace) -> int:
    from . import analysis
    from .culture import Culture
    from .stimulation import pulse_train

    c = Culture(n_neurons=a.neurons, seed=a.seed, neuron_model=a.model)
    print(f"{c}\nrecording {a.duration_ms / 1000:g} s of spontaneous activity...")
    rec = c.record(a.duration_ms, raw=a.raw)
    s = analysis.summarize(rec)
    print(f"  {int(rec.counts.sum())} spikes on {len(rec.active_electrodes)} active electrodes")
    print(
        f"  mean rate {s['mean_rate_hz']:.2f} Hz/electrode, {s['burst_rate_hz']:.2f} network bursts/s"
    )
    c.stimulate([27, 28], pulse_train(12.0, n_pulses=10, frequency_hz=2))
    evoked = c.record(5500)
    resp = analysis.evoked_response(evoked)
    print(f"  10 pulses on electrodes 27+28: {resp.mean():.1f} spikes in 5-50 ms per pulse")
    if a.h5:
        rec.to_hdf5(a.h5)
        print(f"  wrote {a.h5}")
    if a.plot:
        from . import plotting

        full = rec.concatenate([rec, evoked])
        plotting.summary_figure(full, title=f"wetware demo (seed {a.seed})").savefig(
            a.plot, dpi=110
        )
        print(f"  wrote {a.plot}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .server import serve

    serve(a.host, a.port)
    return 0


def cmd_dashboard(a: argparse.Namespace) -> int:
    from .culture import Culture
    from .dashboard import run_dashboard

    c = Culture(n_neurons=a.neurons, seed=a.seed, div=a.div, neuron_model=a.model)
    print(f"dashboard on http://{a.host}:{a.port}")
    run_dashboard(a.host, a.port, c, frame_ms=a.frame_ms, speed=a.speed)
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from .protocol import run_protocol

    res = run_protocol(a.protocol, out_dir=a.out, figure=not a.no_figure)
    print(
        f"protocol {res.name}: {len(res.epochs)} epochs, {res.recording.duration_ms / 1000:.1f} s"
    )
    for k in ("n_spikes", "n_stim_pulses", "mean_rate_hz", "burst_rate_hz"):
        v = res.summary[k]
        print(f"  {k}: {v:.3g}" if isinstance(v, float) else f"  {k}: {v}")
    if res.hdf5_path:
        print(f"  wrote {res.hdf5_path}")
    if res.figure_path:
        print(f"  wrote {res.figure_path}")
    return 0


def cmd_batch(a: argparse.Namespace) -> int:
    from .batch import aggregate, grid, run_batch, spontaneous_activity, write_csv

    conds = grid(seed=_parse_range(a.seeds), div=[float(x) for x in a.divs.split(",")])
    for c in conds:
        c["n_neurons"] = a.neurons
    print(f"running {len(conds)} cultures on {a.workers or 'all'} workers...")
    rows = run_batch(spontaneous_activity, conds, processes=a.workers, duration_ms=a.duration_ms)
    for div, (m, se, n) in aggregate(rows, "div", "mean_rate_hz").items():
        b = aggregate([r for r in rows if r["div"] == div], "div", "burst_rate_hz")[div]
        print(
            f"  DIV {div:g}: rate {m:.2f} +- {se:.2f} Hz, bursts {b[0]:.2f} +- {b[1]:.2f}/s (n={n})"
        )
    if a.out:
        print(f"  wrote {write_csv(rows, a.out)}")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    from .benchmark import format_table, machine, measure, run_suite

    rows = (
        [measure(1000, duration_ms=a.duration_ms)]
        if a.quick
        else run_suite(duration_ms=a.duration_ms)
    )
    print(machine())
    print(format_table(rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wetware", description="Simulated cortical culture on a virtual MEA."
    )
    p.add_argument("--version", action="version", version=f"wetware {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="record spontaneous and evoked activity and print a summary")
    d.add_argument("--neurons", type=int, default=1000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--model", default="izhikevich")
    d.add_argument("--duration-ms", type=float, default=10_000.0)
    d.add_argument("--raw", action="store_true", help="keep raw voltage traces")
    d.add_argument("--h5", type=Path, help="save the spontaneous recording to HDF5")
    d.add_argument("--plot", type=Path, help="save a summary figure")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="run the HTTP API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    db = sub.add_parser("dashboard", help="run the live web dashboard")
    db.add_argument("--host", default="127.0.0.1")
    db.add_argument("--port", type=int, default=8050)
    db.add_argument("--neurons", type=int, default=1000)
    db.add_argument("--seed", type=int, default=0)
    db.add_argument("--div", type=float, default=21.0)
    db.add_argument("--model", default="izhikevich")
    db.add_argument("--frame-ms", type=float, default=100.0)
    db.add_argument("--speed", type=float, default=1.0)
    db.set_defaults(func=cmd_dashboard)

    r = sub.add_parser("run", help="execute a YAML protocol")
    r.add_argument("protocol", type=Path)
    r.add_argument("--out", type=Path, default=Path("outputs"))
    r.add_argument("--no-figure", action="store_true")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="spontaneous-activity statistics over many cultures")
    b.add_argument("--seeds", default="0-7", help="e.g. 0-7 or 1,2,5")
    b.add_argument("--divs", default="14,21,35", help="comma-separated days in vitro")
    b.add_argument("--neurons", type=int, default=1000)
    b.add_argument("--duration-ms", type=float, default=10_000.0)
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--out", type=Path, help="CSV output path")
    b.set_defaults(func=cmd_batch)

    be = sub.add_parser("bench", help="measure simulation speed")
    be.add_argument("--quick", action="store_true")
    be.add_argument("--duration-ms", type=float, default=5000.0)
    be.set_defaults(func=cmd_bench)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
