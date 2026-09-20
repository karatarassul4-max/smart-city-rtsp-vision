import time

import numpy as np
from fastapi.testclient import TestClient

from src.detector import Detector
from src.main import app
from src.stream_reader import StreamReader


def test_mock_uses_pixels_and_batches():
    detector = Detector(backend="mock")
    result = detector.detect_batch([StreamReader.synthetic(30), np.zeros((360, 640, 3), dtype=np.uint8)])
    assert result[0][0].label == "person"
    assert result[1] == []


def test_api_incident_lifecycle():
    with TestClient(app) as client:
        assert client.post("/start-stream", json={"zone": [1, 0, 0, 1]}).status_code == 422
        assert client.post("/start-stream", json={"source": "synthetic", "zone": [0, 0, 1, 1]}).status_code == 200
        assert client.post("/start-stream", json={}).status_code == 409
        deadline = time.monotonic() + 10
        alerts = []
        while time.monotonic() < deadline:
            alerts = client.get("/get-latest-alerts").json()
            if alerts:
                break
            time.sleep(0.05)
        assert alerts and alerts[0]["report_source"] == "mock"
        assert alerts[0]["action"] == "notify_operator"
        assert client.post("/webhooks/alerts", json=alerts[0]).status_code == 202
        assert len(client.get("/get-latest-alerts").json()) == len(alerts)
        assert client.post("/stop-stream").json()["running"] is False
        assert client.post("/start-stream", json={}).status_code == 200
        assert client.post("/stop-stream").status_code == 200


def test_file_eof_and_invalid_source(tmp_path):
    import cv2

    video = str(tmp_path / "short.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    assert writer.isOpened()
    for i in range(3):
        writer.write(StreamReader.synthetic(i))
    writer.release()
    reader = StreamReader(video, fps=60, loop=False)
    reader.start()
    assert reader.done.wait(5)
    reader.stop()
    assert reader.error is None
    assert reader.get_batch(1)[0].id == 2
    with TestClient(app) as client:
        assert client.post("/start-stream", json={"source": str(tmp_path / "absent.mp4")}).status_code == 400
        assert client.post("/start-stream", json={}).status_code == 200


def test_missing_onnx_falls_back():
    detector = Detector(backend="tensorrt", model_path="missing.onnx")
    assert detector.backend == "opencv_hog"
    assert detector.detect_batch([np.zeros((360, 640, 3), dtype=np.uint8)]) == [[]]


def test_llm_failure_preserves_policy(monkeypatch):
    import asyncio
    import httpx
    from src.agent import IncidentAgent
    from src.schemas import Detection, Incident

    async def fail(*args, **kwargs):
        raise httpx.ConnectError("Unavailable")

    monkeypatch.setenv("LLM_MODE", "openai")
    monkeypatch.setattr(httpx.AsyncClient, "post", fail)
    person = Detection(label="person", confidence=0.9, box=(0.4, 0.2, 0.5, 0.8))
    event = Incident(frame_id=1, detections=[person] * 3, backend="mock", capture_latency_ms=1)
    alert = asyncio.run(IncidentAgent().run(event))
    assert alert.report_source == "mock_fallback"
    assert alert.severity == "critical"
    assert alert.action == "request_urgent_review"
