"""Bounded baseline tracking and temporal candidates; never classify crowds as danger."""
from collections import deque
from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from .schemas import Detection, StartRequest

VEHICLES = {"car", "truck", "bus", "motorcycle"}


def point(d: Detection) -> tuple[float, float]:
    return ((d.box[0] + d.box[2]) / 2, d.box[3])


def inside(d: Detection, zone: tuple) -> bool:
    x, y = point(d)
    return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]


@dataclass
class Track:
    detection: Detection
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    outside_seen: bool = False
    entered_at: float | None = None
    emitted: set = field(default_factory=set)


@dataclass
class Candidate:
    kind: str
    detections: list[Detection]
    evidence: str


class EventEngine:
    def __init__(self, config: StartRequest) -> None:
        self.config = config
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.previous_gray: np.ndarray | None = None
        self.motion_times: deque[float] = deque(maxlen=40)
        self.last_interaction = float("-inf")
        self.last_motion_sample = float("-inf")

    def update(self, detections: list[Detection], timestamp: float, image: np.ndarray | None = None) -> list[Candidate]:
        self.tracks = {i: t for i, t in self.tracks.items() if timestamp - t.last_seen <= 0.7}
        available = set(self.tracks)
        # One-to-one nearest foot-point association with a strict distance gate.
        pairs = []
        for index, detection in enumerate(detections):
            for ident, track in self.tracks.items():
                if track.detection.label != detection.label:
                    continue
                distance = math.dist(point(detection), point(track.detection))
                if distance < 0.08:
                    pairs.append((distance, index, ident))
        assigned: dict[int, int] = {}
        for _, index, ident in sorted(pairs):
            if index not in assigned and ident in available:
                assigned[index] = ident
                available.remove(ident)
        candidates = []
        for index, detection in enumerate(detections[:100]):
            ident = assigned.get(index)
            if ident is None:
                if len(self.tracks) >= 100:
                    continue
                ident, self.next_id = self.next_id, self.next_id + 1
                self.tracks[ident] = Track(detection, timestamp)
            detection.track_id = ident
            track = self.tracks[ident]
            track.detection, track.last_seen = detection, timestamp
            track.history.append((timestamp, *point(detection)))
            while track.history and timestamp - track.history[0][0] > 2.5:
                track.history.popleft()
            if self.config.scenario == "traffic" and detection.label in VEHICLES:
                candidates.extend(self._traffic(track, timestamp))
            elif self.config.scenario == "person_zone" and detection.label == "person":
                if inside(detection, self.config.zone):
                    track.entered_at = timestamp if track.entered_at is None else track.entered_at
                    if timestamp - track.entered_at >= self.config.dwell_seconds and "person_zone" not in track.emitted:
                        candidates.append(Candidate("person_zone", [detection], "Explicit restricted-person-zone rule; no inference of danger."))
                        track.emitted.add("person_zone")
                else:
                    track.entered_at = None
        if self.config.scenario == "interaction" and image is not None:
            candidate = self._interaction(detections, timestamp, image)
            if candidate:
                candidates.append(candidate)
        return candidates

    def _traffic(self, track: Track, now: float) -> list[Candidate]:
        result = []
        detection = track.detection
        if not inside(detection, self.config.zone):
            track.outside_seen = True
            track.entered_at = None
        elif track.outside_seen:
            track.entered_at = now if track.entered_at is None else track.entered_at
            if now - track.entered_at >= max(0.4, self.config.dwell_seconds) and "zone" not in track.emitted:
                result.append(Candidate("vehicle_zone_entry", [detection],
                                        f"Track {detection.track_id} observed outside, then inside configured exclusion zone for {now-track.entered_at:.1f}s."))
                track.emitted.add("zone")
        history = list(track.history)
        if len(history) < 5 or now - history[0][0] < 1.0 or "wrong_way" in track.emitted:
            return result
        z = self.config.direction_zone
        if not all(z[0] <= x <= z[2] and z[1] <= y <= z[3] for _, x, y in history):
            return result
        _, x0, y0 = history[0]
        _, x1, y1 = history[-1]
        dx, dy = x1-x0, y1-y0
        direction = {"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}[self.config.allowed_direction]
        projection = dx*direction[0] + dy*direction[1]
        distance = math.hypot(dx,dy)
        path = sum(math.dist(a[1:],b[1:]) for a,b in zip(history,history[1:]))
        if projection < -0.035 and distance > 0 and projection/distance < -0.75 and distance/max(path,0.001) > 0.75:
            result.append(Candidate("wrong_way", [detection],
                                    f"Track {detection.track_id}: opposite displacement {abs(projection):.3f} of image size across {now-history[0][0]:.1f}s; allowed direction={self.config.allowed_direction}. Configured image-space rule, not a legal conclusion."))
            track.emitted.add("wrong_way")
        return result

    def _interaction(self, detections: list[Detection], now: float, image: np.ndarray) -> Candidate | None:
        if now - self.last_motion_sample < 0.15:
            return None
        self.last_motion_sample = now
        gray = cv2.cvtColor(cv2.resize(image, (320,180)), cv2.COLOR_BGR2GRAY)
        previous, self.previous_gray = self.previous_gray, gray
        if previous is None:
            return None
        people = [d for d in detections if d.label == "person"]
        close = [d for d in people if any(other is not d and math.dist(point(d),point(other)) < 0.22 for other in people)]
        diff = cv2.absdiff(gray, previous) > 25
        active = False
        if len(close) >= 2 and np.mean(diff) < 0.35:
            x1 = max(0, int(min(d.box[0] for d in close)*320))
            y1 = max(0, int(min(d.box[1] for d in close)*180))
            x2 = min(320, int(max(d.box[2] for d in close)*320))
            y2 = min(180, int(max(d.box[3] for d in close)*180))
            roi = diff[y1:y2,x1:x2]
            active = bool(roi.size and roi.mean() > 0.08)
        if active:
            self.motion_times.append(now)
        while self.motion_times and now-self.motion_times[0] > 3:
            self.motion_times.popleft()
        if len(self.motion_times) >= 6 and now-self.motion_times[0] >= 2 and len(close) >= 2 and now-self.last_interaction >= max(10, self.config.cooldown_seconds):
            self.last_interaction = now
            return Candidate("interaction_candidate", close,
                             "Experimental proximity + local motion candidate over multiple frames. This is NOT a fight classifier: hugs, sports or gestures can trigger it. VLM must review the sequence; no urgency from headcount.")
        return None
