# API overview

All times are in milliseconds, distances in micrometres, currents in µA,
pulse phases in µs, charge in nC and voltages in µV. The generated reference
for every class and function is on the [reference](reference.md) page.

## Culture

```python
from wetware import Culture, MEA

culture = Culture(
    n_neurons=1000,  # 80 % excitatory
    seed=42,  # same seed -> identical run
    mea=MEA(rows=8, cols=8, pitch_um=200),
    neuron_model="izhikevich",  # or "lif", or a NeuronModel instance
    dt_ms=0.5,
    plasticity=True,  # STDP on excitatory synapses
    div=21,  # days in vitro at t = 0
)
```

| Member | What it does |
|---|---|
| `culture.stimulate(electrodes, waveform, delay_ms=0)` | Schedule a pulse or pulse train. Checked against `culture.safety`; raises `StimulationSafetyError`. |
| `culture.record(duration_ms, raw=False, ground_truth=False)` | Run the simulation and return a `SpikeTrains`. |
| `culture.advance(duration_ms)` | Run without recording (warm-up, rest periods). |
| `culture.closed_loop(callback, hz, latency_ms=0, realtime=False)` | Context manager for closed-loop sessions on the simulation clock. |
| `culture.age(days)` | Apply health drift; returns `Health`. |
| `culture.health`, `culture.time_ms`, `culture.div`, `culture.info()` | State. |
| `culture.plasticity = False` | Freeze the weights. |
| `culture.cancel_stimulation()`, `culture.pending_stimuli` | Manage the pulse queue. |

Stimulation is queued, not instantaneous: pulses are delivered when the
simulation reaches their time during the next `record`, `advance` or loop
cycle.

## MEA

`MEA(rows, cols, pitch_um, electrode_diameter_um=30, noise_rms_uv=5, sample_rate_hz=10000)`.
Electrodes are numbered row-major from the top-left. `mea.positions`,
`mea.index(row, col)`, `mea.grid_position(e)`, `mea.dead_electrodes`,
`mea.mark_dead(e)`.

## Stimulation

```python
from wetware import biphasic_pulse, asymmetric_biphasic_pulse, pulse_train
from wetware import Waveform, Phase, PulseTrain, SafetyLimits

pulse = biphasic_pulse(amplitude_ua=10, phase_us=200, interphase_us=0)  # cathodic first
train = pulse_train(amplitude_ua=10, n_pulses=20, frequency_hz=10)
custom = Waveform((Phase(-12, 100), Phase(0, 50), Phase(3, 400)))  # must balance
```

`SafetyLimits` caps amplitude, phase length, charge per phase, charge
density, pulse rate per electrode and electrodes per pulse, and requires
charge balance. Pass your own to `Culture(safety=...)`. A violating request
schedules nothing.

## SpikeTrains

| Member | Meaning |
|---|---|
| `times` / `spike_times(e)` | Detected spike times per electrode (ms). |
| `counts`, `firing_rates_hz`, `active_electrodes` | Per-electrode totals. |
| `mua(bin_ms)`, `binned(bin_ms)`, `population_rate(bin_ms)` | Multi-unit activity as rates or counts per bin. |
| `stim` | `StimLog` with pulse times, electrodes, charge and amplitude. |
| `voltage_uv`, `sample_rate_hz` | Synthesized traces when `raw=True`. |
| `gt_neurons`, `gt_times_ms` | Every neuron's true spikes when `ground_truth=True`. |
| `window(a, b)`, `SpikeTrains.concatenate(parts)` | Slice and join. |
| `to_hdf5(path)`, `SpikeTrains.from_hdf5(path)` | [NWB-like HDF5](file-format.md). |
| `to_dict()`, `from_dict()` | JSON form used by the server. |

## Closed loop

```python
def callback(frame):  # called once per cycle
    if frame.counts[[10, 11]].sum() >= 3:
        frame.stimulate([27], biphasic_pulse(8))


with culture.closed_loop(callback, hz=50, latency_ms=2) as loop:
    loop.run(duration_ms=10_000)  # or loop.step() / loop.run(cycles=n)

loop.recording  # SpikeTrains for the whole session
loop.call_times_ms  # simulation time of every callback
loop.timing_error_ms  # worst deviation from the ideal schedule
```

The loop is driven by simulated time, so it is exactly periodic no matter how
slow the callback is. `frame.stimulate` adds `latency_ms`, the round-trip
delay of a real rig. `realtime=True` additionally paces the loop against the
wall clock and counts `loop.overruns`.

## Analysis and plotting

`wetware.analysis`: `detect_bursts`, `burst_statistics`, `firing_rate_map`,
`psth`, `evoked_response`, `cross_correlogram`, `functional_connectivity`,
`summarize`.

`wetware.plotting` (needs matplotlib): `raster`, `population_rate`,
`rate_map`, `psth_plot`, `connectivity_matrix`, `summary_figure`.

## Protocols

`wetware run protocol.yaml` or `wetware.protocol.run_protocol(source, out_dir)`.
Block types: `rest`, `pulse`, `train`, `ramp`, `paired_pulse`, `closed_loop`,
`age`, `set`, `repeat`. Closed-loop rules: `threshold`, `periodic`, `python`.
See `protocols/` for examples.

## Batches

```python
from wetware.batch import run_batch, grid, spontaneous_activity, aggregate

rows = run_batch(
    spontaneous_activity, grid(seed=range(8), div=[14, 21, 35]), processes=8, duration_ms=10_000
)
aggregate(rows, by="div", metric="burst_rate_hz")
```

The experiment is any top-level function `f(culture, **kwargs) -> dict`.

## Custom neuron models

```python
from wetware.neurons import NeuronModel, register_neuron_model


class Quadratic(NeuronModel):
    name = "qif"

    def init_state(self, is_excitatory, rng): ...
    def step(self, current, dt): ...  # returns a bool array of spikes
    def depolarize(self, dv): ...  # used by stimulation
    @property
    def v(self): ...


register_neuron_model("qif", Quadratic)
Culture(neuron_model="qif")
```

## Server and client

`wetware serve` starts the HTTP API; `wetware.client.WetwareClient` gives a
`RemoteCulture` with the same `stimulate` / `record` / `advance` / `age` /
`closed_loop` methods.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness and version |
| GET | `/neuron-models` | registered models |
| POST | `/cultures` | create (`n_neurons`, `seed`, `rows`, `cols`, `pitch_um`, `neuron_model`, `dt_ms`, `plasticity`, `div`) |
| GET | `/cultures`, `/cultures/{id}` | list, inspect |
| DELETE | `/cultures/{id}` | discard |
| POST | `/cultures/{id}/stimulate` | `electrodes` plus `amplitude_ua`/`phase_us` or explicit `phases`; `n_pulses`, `frequency_hz`, `delay_ms`. 422 on a safety violation. |
| POST | `/cultures/{id}/record` | `duration_ms`, `raw`, `ground_truth`, `format` (`json` or `hdf5`) |
| POST | `/cultures/{id}/advance` | `duration_ms` |
| POST | `/cultures/{id}/age` | `days` |

Interactive OpenAPI docs are at `/docs` on a running server. The server keeps
cultures in memory and has no authentication; bind it to localhost.

## CLI

```text
wetware demo [--neurons N] [--seed S] [--plot fig.png] [--h5 out.h5]
wetware serve [--host H] [--port P]
wetware dashboard [--port P] [--neurons N] [--speed X]
wetware run protocol.yaml [--out DIR]
wetware batch [--seeds 0-7] [--divs 14,21,35] [--workers N] [--out file.csv]
wetware bench [--quick]
```
