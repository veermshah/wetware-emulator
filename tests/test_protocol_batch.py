import h5py
import pytest

from wetware import MEA, Culture
from wetware.batch import aggregate, grid, run_batch, spontaneous_activity, write_csv
from wetware.protocol import ProtocolError, load_protocol, run_protocol

PROTO = """
name: smoke
culture: {n_neurons: 200, seed: 2, mea: {rows: 4, cols: 4}}
blocks:
  - {type: rest, duration_ms: 300}
  - {type: train, electrodes: [5], amplitude_ua: 10, n_pulses: 3, frequency_hz: 10, tail_ms: 100}
  - {type: ramp, electrodes: [6], start_ua: 2, stop_ua: 10, steps: 3, interval_ms: 100}
  - {type: paired_pulse, electrodes: [7], amplitude_ua: 10, isi_ms: 50, tail_ms: 100}
  - type: closed_loop
    hz: 20
    duration_ms: 500
    rule: {type: periodic, every: 5, stimulate: [9], amplitude_ua: 6}
  - {type: set, plasticity: false}
  - {type: age, days: 3}
  - type: repeat
    times: 2
    blocks:
      - {type: pulse, electrodes: [10], amplitude_ua: 8, tail_ms: 100}
"""


def test_protocol_runs_and_writes_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    res = run_protocol(PROTO, out_dir=tmp_path)
    # 3 train + 3 ramp + 2 paired + 2 closed-loop + 2 repeat pulses
    assert res.summary["n_stim_pulses"] == 12
    assert res.recording.duration_ms == pytest.approx(300 + 300 + 300 + 150 + 500 + 200)
    assert res.hdf5_path.exists() and res.figure_path.exists()
    assert [e.type for e in res.epochs].count("pulse") == 2
    with h5py.File(res.hdf5_path) as f:
        assert f["stimulus/presentation/stim_pulses/timestamps"].shape == (12,)
    assert res.summary["health"]["div"] == pytest.approx(24)


def test_threshold_rule_triggers():
    proto = {
        "culture": {"n_neurons": 300, "seed": 1, "mea": {"rows": 4, "cols": 4}},
        "blocks": [
            {
                "type": "closed_loop",
                "hz": 50,
                "duration_ms": 2000,
                "rule": {"type": "threshold", "watch": list(range(16)), "threshold": 1,
                         "stimulate": [0], "amplitude_ua": 5},
            }
        ],
    }
    res = run_protocol(proto)
    assert res.summary["n_stim_pulses"] > 0


@pytest.mark.parametrize(
    "bad",
    [
        {"blocks": []},
        {"blocks": [{"type": "explode"}]},
        {"blocks": [{"type": "train", "electrodes": [1]}]},
        {"blocks": [{"type": "closed_loop", "hz": 1, "duration_ms": 1, "rule": {"type": "x"}}]},
    ],
)
def test_protocol_validation(bad):
    with pytest.raises(ProtocolError):
        load_protocol(bad)


def test_grid_and_inline_batch(tmp_path):
    conds = grid(seed=[0, 1], n_neurons=[150], mea=[{"rows": 2, "cols": 2}])
    assert len(conds) == 2
    rows = run_batch(spontaneous_activity, conds, processes=1, duration_ms=300, warmup_ms=0)
    assert [r["seed"] for r in rows] == [0, 1]
    assert "mean_rate_hz" in rows[0]
    agg = aggregate(rows, "n_neurons", "mean_rate_hz")
    assert agg[150][2] == 2
    assert write_csv(rows, tmp_path / "b.csv").read_text().startswith("seed,")


def test_parallel_batch_matches_inline():
    conds = [
        {"seed": s, "n_neurons": 150, "mea": {"rows": 2, "cols": 2}, "age_days": 2} for s in (3, 4)
    ]
    a = run_batch(spontaneous_activity, conds, processes=1, duration_ms=300, warmup_ms=0)
    b = run_batch(spontaneous_activity, conds, processes=2, duration_ms=300, warmup_ms=0)
    for x, y in zip(a, b, strict=True):
        assert x["mean_rate_hz"] == y["mean_rate_hz"]


def test_protocol_uses_given_culture():
    c = Culture(n_neurons=120, seed=0, mea=MEA(rows=2, cols=2))
    res = run_protocol({"blocks": [{"type": "rest", "duration_ms": 100}]}, culture=c)
    assert res.recording.metadata["n_neurons"] == 120
