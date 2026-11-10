"""Closed-loop Pong on a simulated culture.

A loose re-creation of the game-world setup in Kagan et al. (2022, Neuron):

* Sensory electrodes (column 3 of the 8x8 array) get a place code for the
  ball's vertical position and a rate code (4-40 Hz) for its distance.
* Two motor regions (columns 0-1 and 6-7) control the paddle: whichever
  region fires more than its running baseline moves the paddle its way.
* Feedback after each hit/miss depends on the condition:
    - ``stimulus``: a predictable 100 Hz burst on the sensory electrodes on a
      hit, and unpredictable random pulses for 4 s on a miss.
    - ``silent``: no feedback stimulation at all (the game still pauses).

The culture is simulated, so treat any difference between conditions as a
property of this model and its plasticity rules, not as evidence about real
neurons.

    python examples/closed_loop_pong.py               # 4 seeds x 2 conditions x 8 sim-minutes
    python examples/closed_loop_pong.py --quick       # smoke run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wetware import Culture, StimulationSafetyError, biphasic_pulse
from wetware.batch import run_batch

SENSORY = [r * 8 + 3 for r in range(8)]
MOTOR_UP = [r * 8 + c for r in range(8) for c in (0, 1)]
MOTOR_DOWN = [r * 8 + c for r in range(8) for c in (6, 7)]
LOOP_HZ = 50
PADDLE_H = 0.3


def play(culture: Culture, condition: str, minutes: float, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1000)
    pulse = biphasic_pulse(10.0, 200)
    fb_pulse = biphasic_pulse(8.0, 200)
    alive = set(range(culture.mea.n_electrodes)) - set(culture.mea.dead_electrodes)
    sensory = [e for e in SENSORY if e in alive]
    up = [e for e in MOTOR_UP if e in alive]
    down = [e for e in MOTOR_DOWN if e in alive]

    g = {
        "bx": 0.5,
        "by": 0.5,
        "vx": -0.5,
        "vy": 0.3,
        "py": 0.5,
        "pause_until": 0.0,
        "phase": 0.0,
        "hits": 0,
        "base_up": 1.0,
        "base_down": 1.0,
    }
    rallies: list[tuple[float, int]] = []

    def reset_ball() -> None:
        g["bx"], g["by"] = 0.9, rng.uniform(0.1, 0.9)
        ang = rng.uniform(-0.6, 0.6)
        g["vx"], g["vy"] = -0.5 * np.cos(ang), 0.5 * np.sin(ang)

    def safe_stim(frame, els, wave, delay=0.0) -> None:
        try:
            frame.stimulate(els, wave, delay_ms=delay)
        except StimulationSafetyError:
            pass  # a pulse would break the rate limit; skip it

    def cb(frame) -> None:
        t = frame.t_ms
        dt = 1.0 / LOOP_HZ
        cu = frame.counts[up].sum()
        cd = frame.counts[down].sum()
        # Running baselines so motor output is relative to each region's own rate.
        g["base_up"] = 0.98 * g["base_up"] + 0.02 * cu
        g["base_down"] = 0.98 * g["base_down"] + 0.02 * cd
        drive = (cu - g["base_up"]) / (g["base_up"] + 1) - (cd - g["base_down"]) / (
            g["base_down"] + 1
        )
        g["py"] = float(
            np.clip(g["py"] - 0.6 * np.tanh(drive) * dt * 3, PADDLE_H / 2, 1 - PADDLE_H / 2)
        )
        if t < g["pause_until"]:
            return

        g["bx"] += g["vx"] * dt
        g["by"] += g["vy"] * dt
        if not 0.0 < g["by"] < 1.0:
            g["vy"] *= -1
            g["by"] = float(np.clip(g["by"], 0.0, 1.0))
        if g["bx"] >= 1.0:
            g["vx"] *= -1
            g["bx"] = 1.0
        if g["bx"] <= 0.0:
            if abs(g["by"] - g["py"]) <= PADDLE_H / 2:  # hit
                g["hits"] += 1
                g["vx"] = abs(g["vx"])
                g["bx"] = 0.0
                if condition == "stimulus":
                    for k in range(10):  # predictable: 100 Hz for 100 ms
                        safe_stim(frame, sensory, fb_pulse, delay=5.0 + 10.0 * k)
                g["pause_until"] = t + 120.0
            else:  # miss
                rallies.append((t, g["hits"]))
                g["hits"] = 0
                if condition == "stimulus":
                    for k in range(20):  # unpredictable: random sites at ~5 Hz for 4 s
                        els = list(rng.choice(sorted(alive), size=4, replace=False))
                        safe_stim(frame, els, fb_pulse, delay=5.0 + 200.0 * k + rng.uniform(0, 150))
                g["pause_until"] = t + 4000.0
                reset_ball()
            return

        # Sensory encoding: place code (which electrode) + rate code (how often).
        row = int(np.clip(g["by"] * len(sensory), 0, len(sensory) - 1))
        rate = 4.0 + 36.0 * (1.0 - g["bx"])
        g["phase"] += rate * dt
        if g["phase"] >= 1.0:
            g["phase"] -= 1.0
            safe_stim(frame, [sensory[row]], pulse)

    with culture.closed_loop(cb, hz=LOOP_HZ, latency_ms=2.0) as loop:
        loop.run(duration_ms=minutes * 60_000.0)
    return {"rallies": rallies}


def experiment(culture: Culture, condition: str, minutes: float, seed: int) -> dict:
    culture.advance(2000)
    return play(culture, condition, minutes, seed)


def run_condition(culture: Culture, condition: str = "stimulus", minutes: float = 5.0) -> dict:
    return experiment(culture, condition, minutes, int(culture.seed or 0))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--neurons", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("docs/img/pong.png"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.minutes, a.neurons = 1, 0.5, 400

    rows = []
    for cond in ("stimulus", "silent"):
        conds = [{"seed": s, "n_neurons": a.neurons} for s in range(a.seeds)]
        res = run_batch(
            run_condition, conds, processes=a.workers, condition=cond, minutes=a.minutes
        )
        for r in res:
            rows.append((cond, r["seed"], r["rallies"]))

    bin_min = max(a.minutes / 4, 0.25)
    edges = np.arange(0, a.minutes + 1e-9, bin_min)
    summary = {}
    for cond in ("stimulus", "silent"):
        per_seed = []
        all_r = []
        for c, _s, rallies in rows:
            if c != cond:
                continue
            t = np.array([x[0] for x in rallies]) / 60_000.0 - 2.0 / 60.0
            h = np.array([x[1] for x in rallies], dtype=float)
            all_r.extend(h.tolist())
            means = [
                h[(t >= lo) & (t < lo + bin_min)].mean()
                if np.any((t >= lo) & (t < lo + bin_min))
                else np.nan
                for lo in edges[:-1]
            ]
            per_seed.append(means)
        arr = np.array(per_seed, dtype=float)
        summary[cond] = (
            np.nanmean(arr, axis=0),
            np.nanstd(arr, axis=0) / np.sqrt(max(arr.shape[0], 1)),
            np.array(all_r),
        )
        print(
            f"{cond:9s}: {len(all_r)} rallies, mean length {np.mean(all_r) if all_r else 0:.2f} hits, "
            f"aces (0-hit rallies) {np.mean(np.array(all_r) == 0) * 100 if all_r else 0:.0f}%"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    centres = edges[:-1] + bin_min / 2
    colors = {"stimulus": "tab:orange", "silent": "tab:gray"}
    for cond, (m, se, _) in summary.items():
        axes[0].errorbar(centres, m, yerr=se, marker="o", capsize=3, label=cond, color=colors[cond])
    axes[0].set_xlabel("time in game (sim minutes)")
    axes[0].set_ylabel("mean rally length (hits)")
    axes[0].set_title("Rally length over time")
    axes[0].legend(frameon=False)
    data = [summary[c][2] for c in ("stimulus", "silent")]
    axes[1].boxplot(data, tick_labels=["stimulus", "silent"], showmeans=True)
    axes[1].set_ylabel("hits per rally")
    axes[1].set_title(f"All rallies ({a.seeds} cultures per condition)")
    fig.suptitle("Closed-loop Pong on a simulated culture")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
