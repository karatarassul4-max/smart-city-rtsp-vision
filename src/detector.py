"""YOLOv8/11 raw ONNX outputs [B,84,N], CPU HOG, and explicit synthetic mock."""
import logging
from pathlib import Path

import cv2
import numpy as np

from .schemas import Detection

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, backend: str = "auto", model_path: str | None = None,
                 synthetic: bool = False) -> None:
        self.session = None
        self.backend = "mock" if synthetic or backend == "mock" else "opencv_hog"
        if backend in ("auto", "onnx", "tensorrt") and model_path:
            try:
                import onnxruntime as ort

                if not Path(model_path).is_file():
                    raise FileNotFoundError("ONNX file not found")
                available = ort.get_available_providers()
                preferred = (["TensorrtExecutionProvider", "CUDAExecutionProvider"]
                             if backend == "tensorrt" else [])
                providers = [p for p in preferred if p in available] + ["CPUExecutionProvider"]
                try:
                    self.session = ort.InferenceSession(model_path, providers=providers)
                except Exception:
                    self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self.input = self.session.get_inputs()[0]
                shape = self.input.shape
                if len(shape) != 4 or shape[1] != 3 or self.input.type != "tensor(float)":
                    raise ValueError("Expected float32 NCHW RGB input")
                self.height = shape[2] if isinstance(shape[2], int) else 640
                self.width = shape[3] if isinstance(shape[3], int) else 640
                self.fixed_batch = shape[0] if isinstance(shape[0], int) else None
                self.backend = "onnx:" + self.session.get_providers()[0]
            except Exception as exc:
                self.session = None
                logger.warning("ONNX unavailable (%s); fallback=%s", type(exc).__name__, self.backend)
        elif backend in ("onnx", "tensorrt"):
            logger.warning("No model_path; fallback=%s", self.backend)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("Detector backend=%s", self.backend)

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        if self.session is not None:
            try:
                return self._onnx(images)
            except Exception as exc:
                logger.warning("ONNX inference failed (%s); switching to CPU HOG", type(exc).__name__)
                self.session = None
                self.backend = "opencv_hog"
        return [self._cpu(image) for image in images]

    def _cpu(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        if self.backend == "mock":
            mask = cv2.inRange(image, np.array([0, 150, 0]), np.array([80, 255, 80]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]
            scores = [0.99] * len(boxes)
        else:
            if h < 128 or w < 64:
                return []
            scale = min(1.0, 640 / w)
            small = cv2.resize(image, (max(64, int(w * scale)), max(128, int(h * scale))))
            found, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
            sx, sy = w / small.shape[1], h / small.shape[0]
            boxes = [(x * sx, y * sy, bw * sx, bh * sy) for x, y, bw, bh in found]
            scores = [float(1 / (1 + np.exp(-float(v)))) for v in weights]
        return [Detection(label="person", confidence=s, box=(x/w, y/h, (x+bw)/w, (y+bh)/h))
                for (x, y, bw, bh), s in zip(boxes, scores)]

    def _onnx(self, images: list[np.ndarray]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        step = self.fixed_batch or len(images)
        for start in range(0, len(images), step):
            chunk = images[start:start + step]
            transforms = []
            tensors = []
            for image in chunk:
                h, w = image.shape[:2]
                scale = min(self.width / w, self.height / h)
                nw, nh = round(w * scale), round(h * scale)
                px, py = (self.width - nw) // 2, (self.height - nh) // 2
                canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
                canvas[py:py+nh, px:px+nw] = cv2.resize(image, (nw, nh))
                tensors.append(canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255)
                transforms.append((w, h, scale, px, py))
            while len(tensors) < step:
                tensors.append(tensors[-1])
            output = self.session.run(None, {self.input.name: np.stack(tensors)})[0]
            if output.ndim != 3 or output.shape[1] != 84:
                raise ValueError("Only raw COCO YOLOv8/11 [B,84,N] exports without NMS are supported")
            for rows, (w, h, scale, px, py) in zip(output, transforms):
                rows = rows.T
                # COCO class 0 is person; discard other winning classes.
                rows = rows[(np.argmax(rows[:, 4:], axis=1) == 0) & (rows[:, 4] >= 0.45)]
                boxes, scores = [], []
                for cx, cy, bw, bh, score, *_ in rows:
                    boxes.append([float(cx-bw/2), float(cy-bh/2), float(bw), float(bh)])
                    scores.append(float(score))
                keep = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.45)
                detected = []
                for i in np.asarray(keep).reshape(-1):
                    x, y, bw, bh = boxes[int(i)]
                    box = np.clip([(x-px)/scale/w, (y-py)/scale/h,
                                   (x+bw-px)/scale/w, (y+bh-py)/scale/h], 0, 1)
                    detected.append(Detection(label="person", confidence=scores[int(i)], box=tuple(box)))
                results.append(detected)
        return results
