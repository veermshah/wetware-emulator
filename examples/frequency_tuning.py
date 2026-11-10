"""Frequency-tuning sweep: how well does the culture follow pulse trains?

Stimulates one electrode pair with 20-pulse trains from 1 to 80 Hz and
counts the spikes evoked by each pulse across the array. Short-term synaptic
depression makes the response fall off along a train, faster at higher
rates. A second panel shows the input-output curve against pulse amplitude.

    python examples/frequency_tuning.py
    python examples/frequency_tuning.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wetware import Culture, biphasic_pulse, pulse_train
from wetware.analysis import evoked_response, psth

FREQS = [1, 2, 5, 10, 20, 40, 80]
WINDOW = (3.0, 12.0)  # ms after each pulse; fits inside the 80 Hz inter-pulse interval


def baseline_count(rec, n: int, rng: np.random.Generator) -> float:
    """Spikes expected in WINDOW by chance, from random trigger times."""
    t = rng.uniform(rec.t_start_ms, rec.t_stop_ms - 20, n)
    return float(evoked_response(rec, t, WINDOW).mean())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--neurons", type=int, default=1000)
    ap.add_argument("--pulses", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("docs/img/frequency_tuning.png"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.neurons, a.pulses = 1, 400, 8

    els = [27, 28]
    curves = {f: [] for f in FREQS}
    amps = [2, 4, 6, 8, 12, 16, 20, 25]
    io = []
    psth_acc = None
    for seed in range(a.seeds):
        c = Culture(n_neurons=a.neurons, seed=seed, plasticity=False)
        rng = np.random.default_rng(seed)
        c.advance(1000)
        base = baseline_count(c.record(10_000), 500, rng)
        for f in FREQS:
            c.advance(4000)  # let synaptic resources recover
            train = pulse_train(12.0, n_pulses=a.pulses, frequency_hz=f)
            c.stimulate(els, train)
            rec = c.record(train.duration_ms + 100)
            curves[f].append(evoked_response(rec, window_ms=WINDOW) - base)
            if f == 1:
                centres, r = psth(rec, window_ms=(-20, 120), bin_ms=2)
                psth_acc = r if psth_acc is None else psth_acc + r
        row = []
        for amp in amps:
            resp = []
            for _ in range(4):
                c.advance(3000)
                c.stimulate(els, biphasic_pulse(float(amp)))
                resp.append(evoked_response(c.record(100), window_ms=WINDOW)[0] - base)
            row.append(np.mean(resp))
        io.append(row)
        print(
            f"seed {seed}: baseline {base:.1f} spikes/window; "
            + ", ".join(f"{f} Hz {np.mean(curves[f][-1]):.1f}" for f in FREQS)
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    steady = []
    for i, f in enumerate(FREQS):
        m = np.mean(curves[f], axis=0)
        axes[0].plot(
            np.arange(1, m.size + 1),
            m,
            marker=".",
            color=cmap(i / (len(FREQS) - 1)),
            label=f"{f} Hz",
        )
        half = m.size // 2
        steady.append(
            (
                np.mean([np.mean(x[half:]) for x in curves[f]]),
                np.std([np.mean(x[half:]) for x in curves[f]]) / np.sqrt(len(curves[f])),
            )
        )
    axes[0].set_xlabel("pulse number in train")
    axes[0].set_ylabel(f"evoked spikes, {WINDOW[0]:g}-{WINDOW[1]:g} ms (above baseline)")
    axes[0].set_title("Response along the train")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    s = np.array(steady)
    axes[1].errorbar(FREQS, s[:, 0], yerr=s[:, 1], marker="o", capsize=3, color="tab:purple")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("stimulation frequency (Hz)")
    axes[1].set_ylabel("steady-state evoked spikes per pulse")
    axes[1].set_title("Frequency tuning")
    io_arr = np.array(io)
    axes[2].errorbar(
        amps,
        io_arr.mean(0),
        yerr=io_arr.std(0) / np.sqrt(len(io)),
        marker="o",
        capsize=3,
        color="tab:red",
    )
    axes[2].set_xlabel("pulse amplitude (µA, 200 µs/phase)")
    axes[2].set_ylabel("evoked spikes per pulse")
    axes[2].set_title("Input-output curve")
    axes[3].bar(centres, psth_acc / a.seeds, width=2, color="tab:blue")
    axes[3].axvspan(0, 3, color="0.8")
    axes[3].set_xlabel("time from pulse (ms)")
    axes[3].set_ylabel("rate per electrode (Hz)")
    axes[3].set_title("PSTH at 1 Hz (grey: blanked)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(
        "steady-state per pulse:",
        {f: round(float(v), 1) for f, v in zip(FREQS, s[:, 0], strict=True)},
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
