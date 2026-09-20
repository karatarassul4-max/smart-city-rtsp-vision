"""Encode annotated frames once; all dashboard clients reuse the same JPEG."""
import cv2
import numpy as np

from .schemas import Detection


def annotate(image: np.ndarray, detections: list[Detection],
             zone: tuple[float, float, float, float], simulated: bool) -> bytes:
    h, w = image.shape[:2]
    scale = min(1, 960 / w)
    canvas = cv2.resize(image, (round(w * scale), round(h * scale)))
    h, w = canvas.shape[:2]
    x1, y1, x2, y2 = zone
    a, b = (round(x1*w), round(y1*h)), (round(x2*w), round(y2*h))
    overlay = canvas.copy()
    cv2.rectangle(overlay, a, b, (70, 160, 240), -1)
    canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0)
    cv2.rectangle(canvas, a, b, (70, 160, 240), 2)
    cv2.putText(canvas, "RESTRICTED ZONE", (a[0]+5, max(18, a[1]-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 160, 240), 2)
    for detection in detections:
        left, top, right, bottom = detection.box
        center = (left + right) / 2
        inside = x1 <= center <= x2 and y1 <= bottom <= y2
        color = (90, 90, 250) if inside else (185, 220, 50)
        cv2.rectangle(canvas, (round(left*w), round(top*h)), (round(right*w), round(bottom*h)), color, 2)
        cv2.circle(canvas, (round(center*w), round(bottom*h)), 4, color, -1)
        cv2.putText(canvas, f"person {detection.confidence:.0%}", (round(left*w), max(18, round(top*h)-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if simulated:
        cv2.putText(canvas, "SIMULATION", (12, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 190, 255), 2)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("Could not encode preview")
    return encoded.tobytes()
