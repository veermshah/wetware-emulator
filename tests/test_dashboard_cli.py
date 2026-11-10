import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from wetware import MEA, Culture
from wetware.cli import _parse_range, main
from wetware.dashboard import create_dashboard_app


@pytest.fixture
def dash():
    c = Culture(n_neurons=200, seed=1, mea=MEA(rows=4, cols=4))
    app = create_dashboard_app(c, frame_ms=50, autostart=False)
    with TestClient(app) as tc:
        yield tc


def test_dashboard_page_and_frames(dash):
    page = dash.get("/")
    assert page.status_code == 200 and '<canvas id="heat"' in page.text
    f = dash.post("/api/step").json()
    assert f["type"] == "frame" and len(f["counts"]) == 16 and f["t_ms"] == 50
    with dash.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["t_ms"] == 50


def test_dashboard_controls(dash):
    r = dash.post("/api/stimulate", json={"electrodes": [3], "amplitude_ua": 10, "n_pulses": 3})
    assert r.status_code == 200 and r.json()["scheduled"] == 3
    assert (
        dash.post("/api/stimulate", json={"electrodes": [3], "amplitude_ua": 90}).status_code == 422
    )
    f = dash.post("/api/step").json()
    assert len(f["stim"]) >= 1
    assert dash.post("/api/age", json={"days": 7}).json()["div"] == 28
    assert dash.post("/api/control", json={"paused": True, "speed": 2}).json() == {
        "paused": True,
        "speed": 2,
    }
    assert dash.get("/api/info").json()["events"][0]["kind"] == "stim"


def test_parse_range():
    assert _parse_range("0-3,7") == [0, 1, 2, 3, 7]


def test_cli_demo_and_run(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    assert (
        main(["demo", "--neurons", "200", "--duration-ms", "500", "--h5", str(tmp_path / "d.h5")])
        == 0
    )
    assert (tmp_path / "d.h5").exists()
    proto = tmp_path / "p.yaml"
    proto.write_text(
        "name: cli\nculture: {n_neurons: 150, mea: {rows: 2, cols: 2}}\n"
        "blocks:\n  - {type: rest, duration_ms: 200}\n"
    )
    assert main(["run", str(proto), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "cli.h5").exists() and (tmp_path / "o" / "cli.png").exists()
    out = capsys.readouterr().out
    assert "protocol cli" in out
