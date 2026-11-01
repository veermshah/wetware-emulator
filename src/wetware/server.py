"""HTTP server exposing cultures as remote resources.

Start it with ``wetware serve`` (needs ``pip install wetware[server]``) and
drive it with :class:`wetware.client.WetwareClient`. The point is to make a
simulated experiment look like remote-platform work: you create a culture,
send stimulation, pull recordings, and never touch the simulator directly.

Endpoints (JSON unless noted):

=======  ==================================  =========================================
Method   Path                                Purpose
=======  ==================================  =========================================
GET      ``/health``                         liveness check and version
GET      ``/neuron-models``                  registered neuron models
POST     ``/cultures``                       create a culture
GET      ``/cultures``                       list cultures
GET      ``/cultures/{id}``                  culture info and health
DELETE   ``/cultures/{id}``                  discard a culture
POST     ``/cultures/{id}/stimulate``        schedule pulses (422 on safety violation)
POST     ``/cultures/{id}/record``           run and return spikes (JSON or HDF5 bytes)
POST     ``/cultures/{id}/advance``          run without recording
POST     ``/cultures/{id}/age``              age the culture by some days
=======  ==================================  =========================================
"""

from __future__ import annotations

import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from ._version import __version__
from .culture import Culture
from .mea import MEA
from .neurons import available_neuron_models
from .stimulation import Phase, PulseTrain, StimulationSafetyError, Waveform

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("the server needs FastAPI: pip install 'wetware[server]'") from exc


MAX_CULTURES = 32
MAX_NEURONS = 5000
MAX_RECORD_MS = 120_000.0
MAX_RAW_MS = 10_000.0


class CultureSpec(BaseModel):
    n_neurons: int = Field(1000, ge=10, le=MAX_NEURONS)
    seed: int | None = 0
    rows: int = Field(8, ge=1, le=32)
    cols: int = Field(8, ge=1, le=32)
    pitch_um: float = Field(200.0, gt=0)
    neuron_model: str = "izhikevich"
    dt_ms: float = Field(0.5, ge=0.05, le=1.0)
    plasticity: bool = True
    div: float = Field(21.0, ge=0)


class StimRequest(BaseModel):
    electrodes: list[int] = Field(..., min_length=1)
    #: Custom waveform as ``[[amplitude_ua, duration_us], ...]``; overrides the
    #: biphasic shortcut fields below.
    phases: list[tuple[float, float]] | None = None
    amplitude_ua: float | None = Field(None, ge=0)
    phase_us: float = Field(200.0, gt=0)
    interphase_us: float = Field(0.0, ge=0)
    n_pulses: int = Field(1, ge=1, le=10_000)
    frequency_hz: float | None = Field(None, gt=0)
    delay_ms: float = Field(0.0, ge=0)

    def waveform(self) -> Waveform | PulseTrain:
        from .stimulation import biphasic_pulse

        if self.phases:
            pulse = Waveform(tuple(Phase(a, t) for a, t in self.phases))
        elif self.amplitude_ua is not None:
            pulse = biphasic_pulse(self.amplitude_ua, self.phase_us, self.interphase_us)
        else:
            raise ValueError("give either phases or amplitude_ua")
        if self.n_pulses > 1:
            if self.frequency_hz is None:
                raise ValueError("frequency_hz is required when n_pulses > 1")
            return PulseTrain(pulse, self.n_pulses, self.frequency_hz)
        return pulse


class RecordRequest(BaseModel):
    duration_ms: float = Field(..., gt=0, le=MAX_RECORD_MS)
    raw: bool = False
    ground_truth: bool = False
    format: Literal["json", "hdf5"] = "json"


class AdvanceRequest(BaseModel):
    duration_ms: float = Field(..., gt=0, le=MAX_RECORD_MS)


class AgeRequest(BaseModel):
    days: float = Field(..., ge=0, le=3650)


class _Slot:
    def __init__(self, culture: Culture) -> None:
        self.culture = culture
        self.lock = threading.Lock()


def create_app() -> FastAPI:
    """Build the FastAPI application (cultures live in memory)."""
    app = FastAPI(
        title="wetware",
        version=__version__,
        description="Simulated cortical cultures on virtual multi-electrode arrays.",
    )
    slots: dict[str, _Slot] = {}
    registry_lock = threading.Lock()
    app.state.slots = slots

    def get(cid: str) -> _Slot:
        slot = slots.get(cid)
        if slot is None:
            raise HTTPException(404, f"no culture {cid!r}")
        return slot

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "cultures": len(slots)}

    @app.get("/neuron-models")
    def neuron_models() -> list[str]:
        return available_neuron_models()

    @app.post("/cultures", status_code=201)
    def create(spec: CultureSpec) -> dict[str, Any]:
        with registry_lock:
            if len(slots) >= MAX_CULTURES:
                raise HTTPException(429, f"at most {MAX_CULTURES} cultures per server")
            try:
                c = Culture(
                    n_neurons=spec.n_neurons,
                    seed=spec.seed,
                    mea=MEA(rows=spec.rows, cols=spec.cols, pitch_um=spec.pitch_um),
                    neuron_model=spec.neuron_model,
                    dt_ms=spec.dt_ms,
                    plasticity=spec.plasticity,
                    div=spec.div,
                )
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            cid = uuid.uuid4().hex[:12]
            slots[cid] = _Slot(c)
        return {"id": cid, **c.info()}

    @app.get("/cultures")
    def list_cultures() -> list[dict[str, Any]]:
        return [{"id": cid, **s.culture.info()} for cid, s in slots.items()]

    @app.get("/cultures/{cid}")
    def info(cid: str) -> dict[str, Any]:
        s = get(cid)
        with s.lock:
            return {"id": cid, **s.culture.info()}

    @app.delete("/cultures/{cid}", status_code=204)
    def delete(cid: str) -> Response:
        get(cid)
        with registry_lock:
            slots.pop(cid, None)
        return Response(status_code=204)

    @app.post("/cultures/{cid}/stimulate")
    def stimulate(cid: str, req: StimRequest) -> dict[str, Any]:
        s = get(cid)
        with s.lock:
            try:
                events = s.culture.stimulate(req.electrodes, req.waveform(), req.delay_ms)
            except StimulationSafetyError as e:
                raise HTTPException(
                    422, {"error": "stimulation_safety", "message": str(e)}
                ) from e
            except (ValueError, IndexError) as e:
                raise HTTPException(422, {"error": "invalid_request", "message": str(e)}) from e
            return {
                "scheduled": len(events),
                "times_ms": [ev.time_ms for ev in events],
                "time_ms": s.culture.time_ms,
            }

    @app.post("/cultures/{cid}/record", response_model=None)
    def record(cid: str, req: RecordRequest) -> dict[str, Any] | Response:
        if req.raw and req.duration_ms > MAX_RAW_MS:
            raise HTTPException(422, f"raw recordings are limited to {MAX_RAW_MS:g} ms")
        s = get(cid)
        with s.lock:
            rec = s.culture.record(req.duration_ms, raw=req.raw, ground_truth=req.ground_truth)
        if req.format == "hdf5":
            with tempfile.TemporaryDirectory() as td:
                p = rec.to_hdf5(Path(td) / "recording.h5")
                data = p.read_bytes()
            return Response(
                data,
                media_type="application/x-hdf5",
                headers={"Content-Disposition": f'attachment; filename="{cid}.h5"'},
            )
        d = rec.to_dict(include_raw=req.raw)
        if req.ground_truth and rec.gt_times_ms is not None and rec.gt_neurons is not None:
            d["gt_neurons"] = rec.gt_neurons.tolist()
            d["gt_times_ms"] = rec.gt_times_ms.tolist()
        return d

    @app.post("/cultures/{cid}/advance")
    def advance(cid: str, req: AdvanceRequest) -> dict[str, Any]:
        s = get(cid)
        with s.lock:
            s.culture.advance(req.duration_ms)
            return {"time_ms": s.culture.time_ms}

    @app.post("/cultures/{cid}/age")
    def age(cid: str, req: AgeRequest) -> dict[str, Any]:
        s = get(cid)
        with s.lock:
            return s.culture.age(req.days).to_dict()

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Run the server with uvicorn."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
