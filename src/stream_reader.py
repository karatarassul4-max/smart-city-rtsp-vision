"""A capture thread overwrites a bounded queue, never accumulating stale video."""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    id: int
    captured_at: float
    image: np.ndarray
    media_time: float | None = None
    segment: int = 0


class StreamReader:
    def __init__(self, source: str | int, fps: float = 15, loop: bool = True,
                 capacity: int = 1, gstreamer: bool = False) -> None:
        self.source, self.fps, self.loop = source, fps, loop
        self.gstreamer = gstreamer
        self.frames: queue.Queue[Frame] = queue.Queue(maxsize=capacity)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.done = threading.Event()
        self.error: str | None = None
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="capture")

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(8):
            self.stop_event.set()
            raise RuntimeError("Capture initialization timed out")
        if self.error:
            raise RuntimeError(self.error)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=7)
        if self.thread.is_alive():
            logger.warning("Capture backend still blocked; daemon will exit after read returns")

    def get_batch(self, size: int) -> list[Frame]:
        result: list[Frame] = []
        for _ in range(size):
            try:
                result.append(self.frames.get_nowait())
            except queue.Empty:
                break
        return result

    @staticmethod
    def synthetic(index: int) -> np.ndarray:
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        image[:] = (30, 30, 30)
        x = 30 + (index * 8) % 520
        cv2.rectangle(image, (x, 100), (x + 50, 300), (0, 230, 0), -1)
        cv2.putText(image, "SIMULATED PERSON", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        return image

    def _run(self) -> None:
        capture = None
        try:
            synthetic = self.source == "synthetic"
            is_file = isinstance(self.source, str) and Path(self.source).is_file()
            if not synthetic:
                source = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
                if self.gstreamer:
                    capture = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
                elif isinstance(source, str) and "://" in source:
                    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
                else:
                    capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    raise RuntimeError("Cannot open video source (check path, device or backend)")
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.ready.set()
            index = 0
            segment = 0
            source_index = 0
            source_fps = capture.get(cv2.CAP_PROP_FPS) if is_file else self.fps
            if not source_fps or source_fps <= 0:
                source_fps = self.fps
            while not self.stop_event.is_set():
                started = time.monotonic()
                if synthetic:
                    image = self.synthetic(index)
                else:
                    ok, image = capture.read()
                    if not ok and is_file and self.loop:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        segment += 1
                        source_index = 0
                        ok, image = capture.read()
                    if not ok:
                        if not is_file:
                            self.error = "Live capture disconnected or read timed out"
                        break
                frame = Frame(index, time.monotonic(), image,
                              source_index / source_fps if is_file or synthetic else None, segment)
                if self.frames.full():
                    try:
                        self.frames.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass
                self.frames.put_nowait(frame)
                index += 1
                source_index += 1
                # Drain live streams immediately; pace files/synthetic streams for a demo.
                if synthetic or is_file:
                    self.stop_event.wait(max(0, 1 / self.fps - (time.monotonic() - started)))
        except Exception as exc:
            self.error = f"Capture failed: {type(exc).__name__}"
            logger.error(self.error)
        finally:
            if capture is not None:
                capture.release()
            self.ready.set()
            self.done.set()
