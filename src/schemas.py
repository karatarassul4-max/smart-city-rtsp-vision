from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class StartRequest(BaseModel):
    scenario: Literal["observe", "traffic", "interaction", "person_zone"] = "observe"
    allowed_direction: Literal["up", "down", "left", "right"] = "down"
    direction_zone: tuple[float, float, float, float] = (0.1, 0.25, 0.45, 0.98)
    source: str | int = "synthetic"
    backend: Literal["auto", "mock", "onnx", "tensorrt"] = "auto"
    model_path: str | None = None
    batch_size: int = Field(default=1, ge=1, le=8)
    fps: float = Field(default=15, ge=1, le=60)
    loop: bool = True
    gstreamer: bool = False
    cooldown_seconds: float = Field(default=5, ge=0, le=3600)
    confidence: float = Field(default=0.4, ge=0.1, le=0.95)
    dwell_seconds: float = Field(default=0.5, ge=0, le=30)
    zone: tuple[float, float, float, float] = (0.35, 0.1, 0.8, 0.95)

    @model_validator(mode="after")
    def validate_zone(self) -> "StartRequest":
        x1, y1, x2, y2 = self.zone
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("zone must be normalized [left, top, right, bottom]")
        x1, y1, x2, y2 = self.direction_zone
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("direction_zone must be a normalized rectangle")
        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("source cannot be empty")
        return self


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    box: tuple[float, float, float, float]
    track_id: int | None = None


class Incident(BaseModel):
    event_type: Literal["person_zone", "vehicle_zone_entry", "wrong_way", "interaction_candidate"] = "person_zone"
    evidence: str = ""
    evidence_times: list[float] = Field(default_factory=list)
    controlled_test: bool = False
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_id: int
    detections: list[Detection]
    backend: str
    capture_latency_ms: float
    stream_id: str = "legacy"
    source_kind: Literal["simulation", "recording", "live"] = "simulation"
    snapshot_url: str | None = None


class VisionAssessment(BaseModel):
    verdict: Literal["confirmed", "not_confirmed", "uncertain"]
    explanation: str = Field(min_length=1, max_length=3000)


class Alert(BaseModel):
    incident: Incident
    severity: Literal["info", "warning", "critical"]
    action: Literal["none", "notify_operator", "request_urgent_review"]
    report: str = Field(min_length=1, max_length=8000)
    report_source: Literal["mock", "llm", "vlm", "mock_fallback"]
    report_provider: str | None = None
    report_model: str | None = None
    fallback_reason: str | None = None
    vision_assessment: VisionAssessment | None = None
