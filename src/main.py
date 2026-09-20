import asyncio
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .agent import IncidentAgent
from .assets import ASSETS, ROOT, demo_status
from .detector import Detector
from .preview import annotate
from .schemas import Alert, Incident, StartRequest
from .stream_reader import StreamReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.reader: StreamReader | None = None
        self.worker: asyncio.Task | None = None
        self.agent_worker: asyncio.Task | None = None
        self.events: asyncio.Queue[tuple[Incident, bytes]] = asyncio.Queue(maxsize=16)
        self.alerts: deque[Alert] = deque(maxlen=200)
        self.agent = IncidentAgent()
        self.processed = 0
        self.dropped_events = 0
        self.error: str | None = None
        self.backend: str | None = None
        self.latest_jpeg: bytes | None = None
        self.snapshots: dict[str, bytes] = {}
        self.stream_id: str | None = None
        self.source_kind: str | None = None
        self.started_at = 0.0
        self.elapsed = 0.0
        self.people = self.in_zone = 0
        self.latency_ms = 0.0
        self.last_frame_at = 0.0

    async def start(self, config: StartRequest) -> dict:
        async with self.lock:
            if self.worker and not self.worker.done():
                raise HTTPException(409, "A stream is already active")
            if self.reader and self.reader.thread.is_alive():
                raise HTTPException(409, "Previous capture is still shutting down")
            if config.source == "demo":
                if not demo_status()["ready"]:
                    raise HTTPException(400, "Real demo assets are missing. Run pip install -r requirements-onnx.txt, then python -m src.assets")
                config = config.model_copy(update={"source": str(ROOT / ASSETS["video"]["path"]),
                                                   "model_path": str(ROOT / ASSETS["model"]["path"]),
                                                   "backend": "onnx", "fps": 10})
            elif (config.source != "synthetic" and config.backend == "auto" and config.model_path is None
                  and (ROOT / ASSETS["model"]["path"]).is_file()):
                config = config.model_copy(update={"model_path": str(ROOT / ASSETS["model"]["path"])})
            self.reader = StreamReader(config.source, config.fps, config.loop,
                                       config.batch_size, config.gstreamer)
            try:
                detector = await asyncio.to_thread(Detector, config.backend, config.model_path,
                                                   config.source == "synthetic", config.confidence)
                await asyncio.to_thread(self.reader.start)
            except Exception as exc:
                await asyncio.to_thread(self.reader.stop)
                raise HTTPException(400, str(exc)) from exc
            self.processed = self.dropped_events = 0
            self.error = None
            self.backend = detector.backend
            self.latest_jpeg = None
            self.stream_id = str(uuid4())
            self.source_kind = ("simulation" if config.source == "synthetic" else
                                "recording" if isinstance(config.source, str) and Path(config.source).is_file() else "live")
            self.started_at = time.monotonic()
            self.elapsed = self.latency_ms = self.last_frame_at = 0.0
            self.people = self.in_zone = 0
            self.worker = asyncio.create_task(self._process(config, detector))
            return self.status()

    async def stop(self) -> dict:
        async with self.lock:
            if self.reader:
                await asyncio.to_thread(self.reader.stop)
            if self.worker:
                await self.worker
            return self.status()

    def status(self) -> dict:
        running = bool(self.worker and not self.worker.done())
        elapsed = time.monotonic() - self.started_at if running else self.elapsed
        return {"running": bool(self.worker and not self.worker.done()),
                "capture_alive": bool(self.reader and self.reader.thread.is_alive()),
                "backend": self.backend, "processed_frames": self.processed,
                "dropped_frames": self.reader.dropped if self.reader else 0,
                "pending_events": self.events.qsize(), "dropped_events": self.dropped_events,
                "stream_id": self.stream_id, "source_kind": self.source_kind,
                "people": self.people, "in_zone": self.in_zone,
                "latency_ms": round(self.latency_ms, 1),
                "processing_fps": round(self.processed / max(elapsed, 0.001), 1),
                "frame_age_seconds": round(time.monotonic() - self.last_frame_at, 1) if self.last_frame_at else None,
                **self.agent.status(),
                "error": self.error or (self.reader.error if self.reader else None)}

    async def _process(self, config: StartRequest, detector: Detector) -> None:
        reader = self.reader
        last_event = float("-inf")
        zone_since: float | None = None
        try:
            while not reader.stop_event.is_set():
                frames = reader.get_batch(config.batch_size)
                if not frames:
                    if reader.done.is_set():
                        break
                    await asyncio.sleep(0.01)
                    continue
                detections = await asyncio.to_thread(detector.detect_batch, [f.image for f in frames])
                self.backend = detector.backend
                for frame, found in zip(frames, detections):
                    self.processed += 1
                    x1, y1, x2, y2 = config.zone
                    intrusions = [d for d in found if d.label == "person" and
                                  x1 <= (d.box[0]+d.box[2])/2 <= x2 and y1 <= d.box[3] <= y2]
                    latency = (time.monotonic() - frame.captured_at) * 1000
                    self.latest_jpeg = await asyncio.to_thread(annotate, frame.image, found, config.zone,
                                                               detector.backend == "mock")
                    self.people, self.in_zone = len(found), len(intrusions)
                    self.latency_ms = latency
                    self.last_frame_at = time.monotonic()
                    now = time.monotonic()
                    zone_since = (zone_since if zone_since is not None else now) if intrusions else None
                    if self.processed == 1 or self.processed % 30 == 0:
                        logger.info("FRAME %s objects=%s zone=%s latency=%.1fms dropped=%s",
                                    frame.id, len(found), len(intrusions), latency, reader.dropped)
                    if (zone_since is not None and now - zone_since >= config.dwell_seconds
                            and now - last_event >= config.cooldown_seconds):
                        event = Incident(frame_id=frame.id, detections=intrusions,
                                         backend=detector.backend, capture_latency_ms=latency,
                                         stream_id=self.stream_id, source_kind=self.source_kind)
                        event.snapshot_url = f"/alerts/{event.id}/snapshot.jpg"
                        try:
                            self.events.put_nowait((event, self.latest_jpeg))
                            self.snapshots[event.id] = self.latest_jpeg
                            while len(self.snapshots) > 200:
                                self.snapshots.pop(next(iter(self.snapshots)))
                            last_event = time.monotonic()
                        except asyncio.QueueFull:
                            self.dropped_events += 1
        except Exception as exc:
            self.error = f"Processing failed: {type(exc).__name__}"
            logger.exception("Pipeline failed")
        finally:
            self.elapsed = time.monotonic() - self.started_at
            await asyncio.to_thread(reader.stop)

    async def consume_events(self) -> None:
        while True:
            event, jpeg = await self.events.get()
            try:
                alert = await self.agent.run(event, jpeg)
                self.store(alert)
                url = os.getenv("ALERT_WEBHOOK_URL")
                if url:
                    try:
                        async with httpx.AsyncClient(timeout=5) as client:
                            response = await client.post(url, json=alert.model_dump(mode="json"))
                            response.raise_for_status()
                    except Exception as exc:
                        logger.warning("Webhook delivery failed (%s); alert retained locally", type(exc).__name__)
            except Exception:
                logger.exception("Agent failed for incident %s", event.id)
            finally:
                self.events.task_done()

    def store(self, alert: Alert) -> None:
        if not any(a.incident.id == alert.incident.id for a in self.alerts):
            self.alerts.append(alert)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = Pipeline()
    app.state.pipeline = pipeline
    pipeline.agent_worker = asyncio.create_task(pipeline.consume_events())
    try:
        yield
    finally:
        await pipeline.stop()
        try:
            await asyncio.wait_for(pipeline.events.join(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("Shutdown deadline reached; remaining events are not persisted")
        pipeline.agent_worker.cancel()
        try:
            await pipeline.agent_worker
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Smart City RTSP Vision & Agentic Pipeline", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "src" / "static"), name="static")


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(ROOT / "src" / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/demo-status")
async def demo_assets_status() -> dict:
    return demo_status()


@app.get("/frame.jpg", include_in_schema=False)
async def latest_frame() -> Response:
    frame = app.state.pipeline.latest_jpeg
    if frame is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/alerts/{incident_id}/snapshot.jpg", include_in_schema=False)
async def alert_snapshot(incident_id: str) -> Response:
    frame = app.state.pipeline.snapshots.get(incident_id)
    if frame is None:
        raise HTTPException(404, "Snapshot expired or not available")
    return Response(frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", **app.state.pipeline.status()}


@app.post("/start-stream")
async def start_stream(config: StartRequest) -> dict:
    return await app.state.pipeline.start(config)


@app.post("/stop-stream")
async def stop_stream() -> dict:
    return await app.state.pipeline.stop()


@app.get("/get-latest-alerts", response_model=list[Alert])
async def get_latest_alerts(limit: int = Query(default=20, ge=1, le=200)) -> list[Alert]:
    return list(reversed(app.state.pipeline.alerts))[:limit]


@app.post("/webhooks/alerts", status_code=202)
async def receive_alert(alert: Alert) -> dict:
    app.state.pipeline.store(alert)
    return {"accepted": True, "incident_id": alert.incident.id}
