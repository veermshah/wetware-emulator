# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-09

### Added

- `Culture`: 2D dissociated-culture simulation with Izhikevich neurons,
  distance-dependent connectivity, short-term plasticity, STDP and
  spontaneous network bursts.
- `MEA`: virtual electrode grid with noise, stimulation artifacts, blanking
  and threshold spike detection.
- Stimulation waveforms (`biphasic_pulse`, `pulse_train`, custom phases)
  checked against `SafetyLimits`.
- `SpikeTrains` recordings with MUA, optional raw traces and ground truth.
- `Culture.closed_loop` for closed-loop sessions on the simulation clock.
- `Culture.age` health drift: maturation, decline, cell death, electrode
  dropout.
- NWB-like HDF5 export and import.
- FastAPI server (`wetware serve`) and `WetwareClient`.
- Live dashboard (`wetware dashboard`) streaming over WebSocket.
- YAML protocols (`wetware run`).
- `wetware.analysis` and `wetware.plotting`.
- Parallel multi-culture batches (`wetware batch`).
- Neuron-model plugin interface with Izhikevich and LIF built in.
- Benchmark (`wetware bench`).
- Examples: closed-loop Pong, frequency tuning, paired pulse and tetanus.

[0.1.0]: https://github.com/veermshah/wetware-emulator/releases/tag/v0.1.0
