"""Encode annotated frames once; all dashboard clients reuse the same JPEG."""
import cv2
import numpy as np

from .schemas import Detection


def temporal_sheet(images: list[bytes], timestamps: list[float]) -> bytes:
    """One image containing the same chronological evidence, reducing vision token cost."""
    panels = []
    for index, jpeg in enumerate(images[:3]):
        frame = cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid evidence JPEG")
        frame = cv2.resize(frame,(768,round(frame.shape[0]*768/frame.shape[1])))
        panel = cv2.copyMakeBorder(frame,30,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
        stamp = timestamps[index] if index < len(timestamps) else index
        cv2.putText(panel,f"FRAME {index+1} | t={stamp:.2f}s",(12,21),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        panels.append(panel)
    ok, encoded = cv2.imencode('.jpg',np.concatenate(panels,axis=0),[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:
        raise ValueError("Could not encode temporal evidence")
    return encoded.tobytes()


def annotate(image: np.ndarray, detections: list[Detection],
             zone: tuple[float, float, float, float], simulated: bool,
             scenario: str = "person_zone", direction_zone: tuple = (0.1,0.25,0.45,0.98),
             allowed_direction: str = "down", controlled_test: bool = False) -> bytes:
    h, w = image.shape[:2]
    scale = min(1, 960 / w)
    canvas = cv2.resize(image, (round(w * scale), round(h * scale)))
    h, w = canvas.shape[:2]
    x1, y1, x2, y2 = zone
    a, b = (round(x1*w), round(y1*h)), (round(x2*w), round(y2*h))
    overlay = canvas.copy()
    if scenario in {"traffic","person_zone"}:
        cv2.rectangle(overlay, a, b, (70, 160, 240), -1)
    canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0)
    if scenario in {"traffic","person_zone"}:
        cv2.rectangle(canvas, a, b, (70, 160, 240), 2)
        cv2.putText(canvas, "EXCLUSION ZONE", (a[0]+5, max(18, a[1]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 160, 240), 2)
    if scenario == "traffic":
        left,top,right,bottom = direction_zone
        cv2.rectangle(canvas,(round(left*w),round(top*h)),(round(right*w),round(bottom*h)),(245,180,70),1)
        cx,cy=round((left+right)/2*w),round((top+bottom)/2*h)
        dx,dy={"up":(0,-45),"down":(0,45),"left":(-45,0),"right":(45,0)}[allowed_direction]
        cv2.arrowedLine(canvas,(cx-dx,cy-dy),(cx+dx,cy+dy),(245,180,70),3,tipLength=0.25)
    for detection in detections:
        left, top, right, bottom = detection.box
        center = (left + right) / 2
        inside = scenario in {"traffic","person_zone"} and x1 <= center <= x2 and y1 <= bottom <= y2
        color = (90, 90, 250) if inside else (185, 220, 50)
        cv2.rectangle(canvas, (round(left*w), round(top*h)), (round(right*w), round(bottom*h)), color, 2)
        cv2.circle(canvas, (round(center*w), round(bottom*h)), 4, color, -1)
        cv2.putText(canvas, f"{detection.label} #{detection.track_id or '-'} {detection.confidence:.0%}", (round(left*w), max(18, round(top*h)-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if simulated:
        cv2.putText(canvas, "SIMULATION", (12, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 190, 255), 2)
    if controlled_test:
        cv2.putText(canvas, "CONTROLLED TEST: REVERSED PLAYBACK", (12,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(80,190,255),2)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("Could not encode preview")
    return encoded.tobytes()
