import h5py
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from wetware import StimulationSafetyError, biphasic_pulse, pulse_train
from wetware.client import WetwareClient, WetwareError
from wetware.server import create_app


@pytest.fixture
def http():
    with TestClient(create_app()) as tc:
        yield tc


SPEC = {"n_neurons": 200, "seed": 3, "rows": 4, "cols": 4}


def test_health_and_models(http):
    r = http.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "izhikevich" in http.get("/neuron-models").json()


def test_culture_lifecycle(http):
    r = http.post("/cultures", json=SPEC)
    assert r.status_code == 201
    cid = r.json()["id"]
    assert http.get("/cultures").json()[0]["id"] == cid
    r = http.post(f"/cultures/{cid}/stimulate", json={"electrodes": [1], "amplitude_ua": 10})
    assert r.status_code == 200 and r.json()["scheduled"] == 1
    r = http.post(f"/cultures/{cid}/record", json={"duration_ms": 200})
    assert r.status_code == 200
    body = r.json()
    assert len(body["times"]) == 16
    assert body["stim"]["times_ms"] == [0.0]
    assert http.post(f"/cultures/{cid}/advance", json={"duration_ms": 50}).json()["time_ms"] == 250
    h = http.post(f"/cultures/{cid}/age", json={"days": 5}).json()
    assert h["div"] == 26
    assert http.delete(f"/cultures/{cid}").status_code == 204
    assert http.get(f"/cultures/{cid}").status_code == 404


def test_safety_violation_is_422(http):
    cid = http.post("/cultures", json=SPEC).json()["id"]
    r = http.post(
        f"/cultures/{cid}/stimulate",
        json={"electrodes": [0], "phases": [[-10, 200]]},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "stimulation_safety"
    r = http.post(f"/cultures/{cid}/stimulate", json={"electrodes": [0], "amplitude_ua": 500})
    assert r.status_code == 422


def test_validation_errors(http):
    assert http.post("/cultures", json={"n_neurons": 1}).status_code == 422
    cid = http.post("/cultures", json=SPEC).json()["id"]
    assert http.post(f"/cultures/{cid}/record", json={"duration_ms": -5}).status_code == 422
    r = http.post(f"/cultures/{cid}/record", json={"duration_ms": 20_000, "raw": True})
    assert r.status_code == 422


def test_hdf5_download(http, tmp_path):
    cid = http.post("/cultures", json=SPEC).json()["id"]
    r = http.post(f"/cultures/{cid}/record", json={"duration_ms": 100, "format": "hdf5"})
    assert r.headers["content-type"] == "application/x-hdf5"
    p = tmp_path / "x.h5"
    p.write_bytes(r.content)
    with h5py.File(p) as f:
        assert "processing/ecephys/detected_spikes" in f


def test_python_client_mirrors_local_api(http, tmp_path):
    client = WetwareClient(client=http)
    assert client.health()["status"] == "ok"
    c = client.create_culture(**SPEC)
    times = c.stimulate([2, 3], pulse_train(10.0, n_pulses=3, frequency_hz=20))
    assert times == [0.0, 50.0, 100.0]
    rec = c.record(300, ground_truth=True)
    assert len(rec.stim) == 3
    assert rec.gt_times_ms.size > 0
    assert c.time_ms == 300
    with pytest.raises(StimulationSafetyError):
        c.stimulate(0, biphasic_pulse(49.0, phase_us=900))
    h = c.age(2)
    assert h.div == 23
    p = c.record_hdf5(100, tmp_path / "r.h5")
    assert p.exists()
    c2 = client.culture(c.id)
    assert c2.info()["time_ms"] == pytest.approx(400)
    c.delete()
    with pytest.raises(WetwareError):
        c.info()


def test_remote_closed_loop(http):
    client = WetwareClient(client=http)
    c = client.create_culture(**SPEC)
    calls = []

    def cb(frame):
        calls.append(frame.t_ms)
        if frame.cycle == 1:
            frame.stimulate([0], biphasic_pulse(8.0))

    with c.closed_loop(cb, hz=20) as loop:
        loop.run(cycles=4)
    np.testing.assert_allclose(calls, [50, 100, 150, 200])
    assert len(loop.recording.stim) == 1
