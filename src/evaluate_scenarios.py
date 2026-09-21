"""Small reproducible smoke evaluation; not a trained model or accuracy benchmark."""
import argparse
from collections import Counter
import json
import time

import cv2

from .assets import ASSETS, PRESETS, ROOT
from .detector import Detector
from .events import EventEngine
from .schemas import StartRequest


def evaluate(name: str) -> dict:
    preset = PRESETS[name]
    config = StartRequest(**{k:v for k,v in preset.items() if k != "path"})
    engine = EventEngine(config)
    detector = Detector("onnx",str(ROOT/ASSETS["model"]["path"]))
    capture = cv2.VideoCapture(str(ROOT/preset["path"]))
    if not capture.isOpened():
        raise RuntimeError("Run python -m src.assets --scenarios")
    fps = capture.get(cv2.CAP_PROP_FPS)
    counts: Counter = Counter()
    events = []
    processed = 0
    index = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % 3 == 0:
                found = detector.detect_batch([frame])[0]
                candidates = engine.update(found,index/fps,frame)
                counts.update(c.kind for c in candidates)
                events.extend({"type":c.kind,"time":round(index/fps,2),"tracks":[d.track_id for d in c.detections]} for c in candidates)
                processed += 1
            index += 1
    finally:
        capture.release()
    return {"preset":name,"backend":detector.backend,"processed_frames":processed,
            "candidate_counts":dict(counts),"events":events,"runtime_seconds":round(time.monotonic()-started,2)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets",nargs="+",default=["demo","traffic","traffic-reversed","interaction","fight","nonviolent"],choices=list(PRESETS))
    args=parser.parse_args()
    results=[]
    for name in args.presets:
        result=evaluate(name)
        results.append(result)
        print(json.dumps(result),flush=True)
    (ROOT/"media/scenario-evaluation.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
