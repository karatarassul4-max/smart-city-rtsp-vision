"""Explicit live Groq smoke test of temporal evidence, one request per selected clip."""
import argparse
import asyncio
from collections import deque
import json
import time

import cv2

from .agent import IncidentAgent
from .assets import ROOT, PRESETS, ASSETS
from .detector import Detector
from .events import EventEngine
from .preview import annotate
from .schemas import Incident, StartRequest


def first_candidate(name: str) -> tuple[Incident,list[bytes]] | None:
    preset=PRESETS[name]
    config=StartRequest(**{k:v for k,v in preset.items() if k!="path"})
    detector=Detector("onnx",str(ROOT/ASSETS["model"]["path"]))
    engine=EventEngine(config)
    capture=cv2.VideoCapture(str(ROOT/preset["path"]))
    fps=capture.get(cv2.CAP_PROP_FPS)
    samples=deque(maxlen=7)
    index=0
    try:
        while True:
            ok,image=capture.read()
            if not ok:
                return None
            if index%3==0:
                found=detector.detect_batch([image])[0]
                candidates=engine.update(found,index/fps,image)
                jpeg=annotate(image,found,config.zone,False,config.scenario,config.direction_zone,config.allowed_direction,name=="traffic-reversed")
                if not samples or index/fps-samples[-1][0]>=0.45:
                    samples.append((index/fps,jpeg))
                if candidates:
                    sequence=list(samples)
                    if sequence[-1][0]!=index/fps:
                        sequence.append((index/fps,jpeg))
                    midpoint=(sequence[0][0]+sequence[-1][0])/2
                    middle=min(range(len(sequence)),key=lambda i:abs(sequence[i][0]-midpoint))
                    selected=[sequence[i] for i in sorted({0,middle,len(sequence)-1})]
                    c=candidates[0]
                    return Incident(frame_id=index,detections=c.detections,backend=detector.backend,
                                    capture_latency_ms=0,source_kind="recording",event_type=c.kind,
                                    evidence=c.evidence,evidence_times=[t for t,_ in selected],
                                    controlled_test=name=="traffic-reversed"),[jpeg for _,jpeg in selected]
            index+=1
    finally:
        capture.release()


async def check(names: list[str]) -> None:
    agent=IncidentAgent()
    if agent.config.mode!="groq" or not agent.config.api_key:
        raise SystemExit("Configure Groq in .env first")
    results=[]
    for name in names:
        candidate=await asyncio.to_thread(first_candidate,name)
        if candidate is None:
            result={"preset":name,"candidate":False}
        else:
            event,images=candidate
            await asyncio.sleep(max(0,agent.next_request_at-time.monotonic()))
            alert=await agent.run(event,images[-1],images)
            result={"preset":name,"image_count":len(images),"alert":alert.model_dump(mode="json")}
        results.append(result)
        print(json.dumps(result,ensure_ascii=True),flush=True)
    (ROOT/"media/temporal-vlm-evaluation.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets",nargs="+",choices=list(PRESETS),default=["traffic-reversed","fight","nonviolent"])
    asyncio.run(check(parser.parse_args().presets))
