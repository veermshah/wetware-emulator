"""Live web dashboard for a running culture.

``wetware dashboard`` starts a culture, runs it continuously on a background
task, and streams frames over a WebSocket to a single-page UI showing an
electrode heatmap, a scrolling spike raster, the array firing rate, and
stimulation controls. Everything is served locally; the page has no
external dependencies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import AsyncIterator
from importlib import resources
from typing import Any

import numpy as np

from .analysis import detect_bursts
from .culture import Culture
from .stimulation import PulseTrain, StimulationSafetyError, biphasic_pulse

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("the dashboard needs FastAPI: pip install 'wetware[server]'") from exc


MAX_SPIKES_PER_FRAME = 3000


class DashStim(BaseModel):
    electrodes: list[int] = Field(..., min_length=1)
    amplitude_ua: float = Field(10.0, gt=0)
    phase_us: float = Field(200.0, gt=0)
    n_pulses: int = Field(1, ge=1, le=200)
    frequency_hz: float = Field(10.0, gt=0)


class DashControl(BaseModel):
    paused: bool | None = None
    speed: float | None = Field(None, gt=0, le=20)


class DashAge(BaseModel):
    days: float = Field(..., ge=0, le=365)


class Runner:
    """Advances a culture in fixed frames and fans frames out to listeners."""

    def __init__(self, culture: Culture, frame_ms: float = 100.0, speed: float = 1.0) -> None:
        self.culture = culture
        self.frame_ms = frame_ms
        self.speed = speed
        self.paused = False
        self.lock = threading.Lock()
        self.queues: set[asyncio.Queue[str]] = set()
        self.last_frame: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    def step(self) -> dict[str, Any]:
        with self.lock:
            rec = self.culture.record(self.frame_ms)
            health = self.culture.health.to_dict()
            t_ms = self.culture.time_ms
        t, e = rec.all_spikes()
        if t.size > MAX_SPIKES_PER_FRAME:
            keep = np.sort(np.random.default_rng(0).choice(t.size, MAX_SPIKES_PER_FRAME, False))
            t, e = t[keep], e[keep]
        counts = rec.counts
        bursting = bool(detect_bursts(rec, bin_ms=10.0)) if counts.sum() else False
        frame = {
            "type": "frame",
            "t_ms": t_ms,
            "frame_ms": self.frame_ms,
            "grid": list(rec.grid_shape),
            "counts": counts.tolist(),
            "rate_hz": float(counts.sum() / (self.frame_ms / 1000.0)),
            "spikes": np.column_stack([np.round(t, 2), e]).tolist() if t.size else [],
            "stim": [[float(s), list(el)] for s, el in zip(rec.stim.times_ms, rec.stim.electrodes, strict=True)],
            "dead": rec.dead_electrodes,
            "bursting": bursting,
            "health": health,
            "speed": self.speed,
            "paused": self.paused,
        }
        self.last_frame = frame
        return frame

    async def broadcast(self, frame: dict[str, Any]) -> None:
        msg = json.dumps(frame)
        for q in list(self.queues):
            if q.qsize() > 20:  # slow client: drop its backlog
                while not q.empty():
                    q.get_nowait()
            q.put_nowait(msg)

    async def loop(self) -> None:
        aio_loop = asyncio.get_running_loop()
        next_due = time.perf_counter()
        while True:
            if self.paused:
                await asyncio.sleep(0.1)
                next_due = time.perf_counter()
                continue
            frame = await aio_loop.run_in_executor(None, self.step)
            await self.broadcast(frame)
            next_due += self.frame_ms / 1000.0 / self.speed
            delay = next_due - time.perf_counter()
            if delay < -1.0:  # fell far behind; resync instead of bursting
                next_due = time.perf_counter()
            await asyncio.sleep(max(delay, 0.0))


def create_dashboard_app(
    culture: Culture | None = None,
    frame_ms: float = 100.0,
    speed: float = 1.0,
    autostart: bool = True,
) -> FastAPI:
    """Build the dashboard app around ``culture`` (a default one if omitted)."""
    runner = Runner(culture or Culture(n_neurons=1000, seed=0), frame_ms, speed)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(runner.loop()) if autostart else None
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="wetware dashboard", lifespan=lifespan)
    app.state.runner = runner

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("wetware").joinpath("static/dashboard.html").read_text()

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        with runner.lock:
            return {
                **runner.culture.info(),
                "frame_ms": runner.frame_ms,
                "speed": runner.speed,
                "paused": runner.paused,
                "events": runner.events[-50:],
            }

    @app.post("/api/step")
    def step() -> dict[str, Any]:
        """Advance one frame by hand (useful when paused or in tests)."""
        return runner.step()

    @app.post("/api/stimulate")
    def stimulate(req: DashStim) -> dict[str, Any]:
        pulse = biphasic_pulse(req.amplitude_ua, req.phase_us)
        wave = PulseTrain(pulse, req.n_pulses, req.frequency_hz) if req.n_pulses > 1 else pulse
        with runner.lock:
            try:
                events = runner.culture.stimulate(req.electrodes, wave)
            except StimulationSafetyError as e:
                raise HTTPException(422, str(e)) from e
            except (ValueError, IndexError) as e:
                raise HTTPException(422, str(e)) from e
            t = runner.culture.time_ms
        ev = {"t_ms": t, "kind": "stim", "detail": req.model_dump()}
        runner.events.append(ev)
        return {"scheduled": len(events), **ev}

    @app.post("/api/age")
    def age(req: DashAge) -> dict[str, Any]:
        with runner.lock:
            h = runner.culture.age(req.days).to_dict()
            t = runner.culture.time_ms
        runner.events.append({"t_ms": t, "kind": "age", "detail": {"days": req.days}})
        return h

    @app.post("/api/control")
    def control(req: DashControl) -> dict[str, Any]:
        if req.paused is not None:
            runner.paused = req.paused
        if req.speed is not None:
            runner.speed = req.speed
        return {"paused": runner.paused, "speed": runner.speed}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        q: asyncio.Queue[str] = asyncio.Queue()
        runner.queues.add(q)
        try:
            if runner.last_frame is not None:
                await socket.send_text(json.dumps(runner.last_frame))
            while True:
                await socket.send_text(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            runner.queues.discard(q)

    return app


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8050,
    culture: Culture | None = None,
    frame_ms: float = 100.0,
    speed: float = 1.0,
) -> None:  # pragma: no cover
    """Serve the dashboard with uvicorn."""
    import uvicorn

    uvicorn.run(create_dashboard_app(culture, frame_ms, speed), host=host, port=port)
