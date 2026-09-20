"""Send one annotated public demo frame to Groq and print the validated assessment."""
import asyncio
import json

import cv2

from .agent import IncidentAgent
from .assets import ROOT, ASSETS, demo_status
from .detector import Detector
from .preview import annotate
from .schemas import Incident


async def check() -> None:
    agent = IncidentAgent()
    if agent.config.mode != "groq" or not agent.config.api_key:
        raise SystemExit("Set LLM_MODE=groq and GROQ_API_KEY in .env first")
    if not demo_status()["ready"]:
        raise SystemExit("Install requirements-onnx.txt and run python -m src.assets first")
    capture = cv2.VideoCapture(str(ROOT / ASSETS["video"]["path"]))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit("Cannot read demo frame")
    detector = Detector("onnx", str(ROOT / ASSETS["model"]["path"]))
    found = detector.detect_batch([frame])[0]
    zone = (0.35, 0.1, 0.8, 0.95)
    inside = [d for d in found if zone[0] <= (d.box[0]+d.box[2])/2 <= zone[2] and zone[1] <= d.box[3] <= zone[3]]
    event = Incident(frame_id=0, detections=inside, backend=detector.backend,
                     capture_latency_ms=0, source_kind="recording")
    alert = await agent.run(event, annotate(frame, found, zone, False))
    print(json.dumps(alert.model_dump(mode="json"), ensure_ascii=True, indent=2))
    if alert.report_source != "vlm":
        raise SystemExit("Groq check failed: " + str(alert.fallback_reason))


if __name__ == "__main__":
    asyncio.run(check())
