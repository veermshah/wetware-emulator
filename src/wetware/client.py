"""Python client for ``wetware serve``.

The client mirrors the local :class:`~wetware.Culture` API, so the same
experiment code runs against a local culture or a remote one::

    from wetware.client import WetwareClient
    from wetware import biphasic_pulse

    client = WetwareClient("http://127.0.0.1:8000")
    culture = client.create_culture(n_neurons=1000, seed=1)
    culture.stimulate([27], biphasic_pulse(10.0))
    rec = culture.record(1000)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .culture import ClosedLoop, Health, LoopFrame
from .recording import SpikeTrains
from .stimulation import PulseTrain, StimulationSafetyError, Waveform

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise ImportError("the client needs httpx: pip install 'wetware[server]'") from exc


class WetwareError(RuntimeError):
    """Server returned an error that is not a safety violation."""


def _raise_for(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = resp.text
    if isinstance(detail, dict) and detail.get("error") == "stimulation_safety":
        raise StimulationSafetyError(detail.get("message", "unsafe stimulation"))
    if isinstance(detail, dict):
        detail = detail.get("message", detail)
    raise WetwareError(f"HTTP {resp.status_code}: {detail}")


class WetwareClient:
    """Connection to a wetware server.

    Args:
        base_url: Server address, e.g. ``http://127.0.0.1:8000``.
        client: Optional pre-built ``httpx.Client`` (for example FastAPI's
            ``TestClient``); ``base_url`` is ignored when given.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        client: httpx.Client | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.http = (
            client if client is not None else httpx.Client(base_url=base_url, timeout=timeout)
        )

    def health(self) -> dict[str, Any]:
        r = self.http.get("/health")
        _raise_for(r)
        return dict(r.json())

    def neuron_models(self) -> list[str]:
        r = self.http.get("/neuron-models")
        _raise_for(r)
        return list(r.json())

    def create_culture(self, **spec: Any) -> RemoteCulture:
        """Create a culture. Keyword arguments match ``server.CultureSpec``
        (``n_neurons``, ``seed``, ``rows``, ``cols``, ``pitch_um``,
        ``neuron_model``, ``dt_ms``, ``plasticity``, ``div``)."""
        r = self.http.post("/cultures", json=spec)
        _raise_for(r)
        info = r.json()
        return RemoteCulture(self, info["id"], info)

    def list_cultures(self) -> list[dict[str, Any]]:
        r = self.http.get("/cultures")
        _raise_for(r)
        return list(r.json())

    def culture(self, culture_id: str) -> RemoteCulture:
        """Attach to an existing culture by id."""
        r = self.http.get(f"/cultures/{culture_id}")
        _raise_for(r)
        return RemoteCulture(self, culture_id, r.json())

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> WetwareClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _stim_payload(
    electrodes: int | Sequence[int], waveform: Waveform | PulseTrain, delay_ms: float
) -> dict[str, Any]:
    els = [int(e) for e in np.atleast_1d(np.asarray(electrodes))]
    if isinstance(waveform, PulseTrain):
        pulse, n, f = waveform.pulse, waveform.n_pulses, waveform.frequency_hz
    else:
        pulse, n, f = waveform, 1, None
    return {
        "electrodes": els,
        "phases": [[p.amplitude_ua, p.duration_us] for p in pulse.phases],
        "n_pulses": n,
        "frequency_hz": f,
        "delay_ms": delay_ms,
    }


class RemoteCulture:
    """A culture living on a server. Same method names as :class:`Culture`."""

    def __init__(self, client: WetwareClient, culture_id: str, info: dict[str, Any]) -> None:
        self.client = client
        self.id = culture_id
        self._info = info
        self.time_ms: float = float(info.get("time_ms", 0.0))
        self.dt_ms: float = float(info.get("dt_ms", 0.5))

    @property
    def _http(self) -> httpx.Client:
        return self.client.http

    def info(self) -> dict[str, Any]:
        r = self._http.get(f"/cultures/{self.id}")
        _raise_for(r)
        self._info = r.json()
        self.time_ms = float(self._info["time_ms"])
        return self._info

    @property
    def health(self) -> Health:
        h = self.info()["health"]
        return Health(**h)

    def stimulate(
        self,
        electrodes: int | Sequence[int],
        waveform: Waveform | PulseTrain,
        delay_ms: float = 0.0,
    ) -> list[float]:
        """Schedule pulses; returns their scheduled times (ms)."""
        r = self._http.post(
            f"/cultures/{self.id}/stimulate", json=_stim_payload(electrodes, waveform, delay_ms)
        )
        _raise_for(r)
        return list(r.json()["times_ms"])

    def record(
        self, duration_ms: float, raw: bool = False, ground_truth: bool = False
    ) -> SpikeTrains:
        r = self._http.post(
            f"/cultures/{self.id}/record",
            json={"duration_ms": duration_ms, "raw": raw, "ground_truth": ground_truth},
        )
        _raise_for(r)
        d = r.json()
        rec = SpikeTrains.from_dict(d)
        if "gt_times_ms" in d:
            rec.gt_neurons = np.asarray(d["gt_neurons"], dtype=np.int64)
            rec.gt_times_ms = np.asarray(d["gt_times_ms"], dtype=float)
        self.time_ms = rec.t_stop_ms
        return rec

    def record_hdf5(
        self, duration_ms: float, path: str | Path, raw: bool = False, ground_truth: bool = False
    ) -> Path:
        """Record on the server and save the HDF5 file it returns."""
        r = self._http.post(
            f"/cultures/{self.id}/record",
            json={
                "duration_ms": duration_ms,
                "raw": raw,
                "ground_truth": ground_truth,
                "format": "hdf5",
            },
        )
        _raise_for(r)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        self.time_ms += duration_ms
        return p

    def advance(self, duration_ms: float) -> None:
        r = self._http.post(f"/cultures/{self.id}/advance", json={"duration_ms": duration_ms})
        _raise_for(r)
        self.time_ms = float(r.json()["time_ms"])

    def age(self, days: float) -> Health:
        r = self._http.post(f"/cultures/{self.id}/age", json={"days": days})
        _raise_for(r)
        return Health(**r.json())

    @contextmanager
    def closed_loop(
        self,
        callback: Callable[[LoopFrame], Any],
        hz: float,
        latency_ms: float = 0.0,
        realtime: bool = False,
        speed: float = 1.0,
    ) -> Iterator[ClosedLoop]:
        """Client-side closed loop: one ``record`` request per cycle."""
        loop = ClosedLoop(self, callback, hz, latency_ms, realtime, speed, raw=False)
        try:
            yield loop
        finally:
            loop._finish()

    def delete(self) -> None:
        r = self._http.delete(f"/cultures/{self.id}")
        _raise_for(r)

    def __repr__(self) -> str:
        return f"RemoteCulture(id={self.id!r}, t={self.time_ms:.1f} ms)"
