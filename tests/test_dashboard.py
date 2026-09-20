import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.assets import ASSETS, ROOT, demo_status
from src.detector import Detector
from src.main import app


def test_dashboard_preview_and_snapshot():
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert 'id="video"' in page.text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/frame.jpg").status_code == 204
        assert client.post("/start-stream", json={"source": "synthetic", "zone": [0, 0, 1, 1], "dwell_seconds": 0}).status_code == 200
        deadline = time.monotonic() + 8
        alerts = []
        while time.monotonic() < deadline:
            alerts = client.get("/get-latest-alerts").json()
            if alerts:
                break
            time.sleep(0.05)
        assert alerts
        status = client.get("/health").json()
        assert status["source_kind"] == "simulation"
        assert status["people"] == 1
        assert status["processing_fps"] > 0
        frame = client.get("/frame.jpg")
        assert frame.headers["content-type"] == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR) is not None
        event = alerts[0]["incident"]
        assert event["stream_id"] == status["stream_id"]
        assert client.get(event["snapshot_url"]).status_code == 200
        assert client.get("/alerts/absent/snapshot.jpg").status_code == 404


def test_zone_confirmation_suppresses_brief_presence():
    with TestClient(app) as client:
        client.post("/start-stream", json={"source": "synthetic", "zone": [0, 0, 1, 1], "dwell_seconds": 30})
        time.sleep(0.3)
        assert client.get("/get-latest-alerts").json() == []


@pytest.mark.skipif(not demo_status()["ready"], reason="Run python -m src.assets and install ONNX Runtime")
def test_real_video_yolo_and_agent():
    capture = cv2.VideoCapture(str(ROOT / ASSETS["video"]["path"]))
    ok, frame = capture.read()
    capture.release()
    assert ok
    detector = Detector("onnx", str(ROOT / ASSETS["model"]["path"]))
    found = detector.detect_batch([frame])[0]
    assert detector.backend.startswith("onnx:")
    assert len(found) >= 2
    assert all(d.label == "person" and d.confidence >= 0.4 for d in found)
    with TestClient(app) as client:
        response = client.post("/start-stream", json={"source": "demo"})
        assert response.status_code == 200
        assert response.json()["source_kind"] == "recording"
        assert response.json()["backend"].startswith("onnx:")
        alerts = []
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            alerts = client.get("/get-latest-alerts").json()
            if alerts:
                break
            time.sleep(0.1)
        assert alerts
        assert alerts[0]["incident"]["source_kind"] == "recording"
        assert alerts[0]["incident"]["backend"].startswith("onnx:")
        assert client.get(alerts[0]["incident"]["snapshot_url"]).status_code == 200
