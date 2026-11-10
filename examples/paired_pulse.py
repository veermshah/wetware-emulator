"""Paired-pulse ratio and a tetanus plasticity protocol.

Part 1: two identical pulses separated by an inter-stimulus interval (ISI).
The paired-pulse ratio (response to pulse 2 / response to pulse 1) shows
short-term depression at short ISIs and recovery at long ones.

Part 2: probe the evoked response at 0.33 Hz, deliver a theta-burst tetanus,
and probe again. This is repeated with STDP switched on and off, so the
STDP-off run is a control for everything that is not long-term plasticity.

    python examples/paired_pulse.py
    python examples/paired_pulse.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wetware import Culture, biphasic_pulse, pulse_train
from wetware.analysis import evoked_response

ISIS = [20, 30, 50, 75, 100, 150, 250, 500, 1000, 2000]
ELS = [27, 28]
WINDOW = (3.0, 18.0)


def paired_pulse(seed: int, neurons: int, reps: int) -> tuple[np.ndarray, float]:
    c = Culture(n_neurons=neurons, seed=seed, plasticity=False)
    c.advance(1000)
    spont = c.record(10_000)
    rng = np.random.default_rng(seed)
    base = float(
        evoked_response(
            spont, rng.uniform(spont.t_start_ms, spont.t_stop_ms - 30, 500), WINDOW
        ).mean()
    )
    pulse = biphasic_pulse(12.0)
    out = np.zeros((len(ISIS), 2))
    for i, isi in enumerate(ISIS):
        for _ in range(reps):
            c.advance(5000)
            c.stimulate(ELS, pulse)
            c.stimulate(ELS, pulse, delay_ms=isi)
            rec = c.record(isi + 60)
            out[i] += evoked_response(rec, window_ms=WINDOW) - base
    return out / reps, base


def tetanus(
    seed: int, neurons: int, n_probe: int, plasticity: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    c = Culture(n_neurons=neurons, seed=seed, plasticity=False)
    c.advance(1000)
    probe = biphasic_pulse(12.0)

    def probes() -> np.ndarray:
        r = []
        for _ in range(n_probe):
            c.advance(2950)
            c.stimulate(ELS, probe)
            r.append(evoked_response(c.record(50), window_ms=WINDOW)[0])
        return np.array(r)

    pre = probes()
    w0 = c.network.mean_exc_weight()
    c.plasticity = plasticity
    for _ in range(5):  # theta burst: 10 bursts of 4 pulses at 100 Hz, 5 Hz burst rate
        for b in range(10):
            c.stimulate(ELS, pulse_train(12.0, n_pulses=4, frequency_hz=100), delay_ms=200.0 * b)
        c.advance(2000 + 3000)
    c.plasticity = False
    return pre, probes(), c.network.mean_exc_weight() / w0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--neurons", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--probes", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("docs/img/paired_pulse.png"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.neurons, a.reps, a.probes = 1, 400, 2, 4

    pp = np.array([paired_pulse(s, a.neurons, a.reps)[0] for s in range(a.seeds)])
    r1, r2 = pp[:, :, 0].mean(0), pp[:, :, 1].mean(0)
    ppr = r2 / np.maximum(r1, 1e-9)
    ppr_seed = pp[:, :, 1] / np.maximum(pp[:, :, 0], 1e-9)
    print("paired-pulse ratio:", {i: round(float(v), 2) for i, v in zip(ISIS, ppr, strict=True)})

    tet = {}
    for on in (True, False):
        res = [tetanus(s, a.neurons, a.probes, on) for s in range(a.seeds)]
        pre = np.array([r[0] for r in res])
        post = np.array([r[1] for r in res])
        wr = float(np.mean([r[2] for r in res]))
        tet[on] = (pre, post, wr)
        print(
            f"STDP {'on ' if on else 'off'}: probe response {pre.mean():.1f} -> {post.mean():.1f} spikes "
            f"({(post.mean() / pre.mean() - 1) * 100:+.0f}%), mean E weight x{wr:.3f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].plot(ISIS, r1, marker="o", label="pulse 1", color="tab:blue")
    axes[0].plot(ISIS, r2, marker="s", label="pulse 2", color="tab:orange")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("inter-stimulus interval (ms)")
    axes[0].set_ylabel("evoked spikes above baseline")
    axes[0].set_title("Paired-pulse responses")
    axes[0].legend(frameon=False)
    axes[1].errorbar(
        ISIS,
        ppr,
        yerr=ppr_seed.std(0) / np.sqrt(a.seeds),
        marker="o",
        capsize=3,
        color="tab:purple",
    )
    axes[1].axhline(1.0, color="0.5", ls="--", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("inter-stimulus interval (ms)")
    axes[1].set_ylabel("paired-pulse ratio (R2 / R1)")
    axes[1].set_title("Short-term depression and recovery")
    x = np.arange(a.probes)
    for on, color in ((True, "tab:red"), (False, "tab:gray")):
        pre, post, wr = tet[on]
        norm = pre.mean()
        lab = f"STDP {'on' if on else 'off'} (weights x{wr:.2f})"
        axes[2].plot(x - a.probes, pre.mean(0) / norm, color=color, marker=".", lw=0.8)
        axes[2].plot(x + 1, post.mean(0) / norm, color=color, marker=".", lw=0.8, label=lab)
    axes[2].axvline(0, color="k", lw=1.5)
    axes[2].axhline(1.0, color="0.5", ls="--", lw=0.8)
    axes[2].set_xlabel("probe number (tetanus at 0)")
    axes[2].set_ylabel("probe response / pre-tetanus mean")
    axes[2].set_title("Theta-burst tetanus")
    axes[2].legend(frameon=False, fontsize=8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
