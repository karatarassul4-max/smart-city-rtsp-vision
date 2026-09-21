import asyncio

import numpy as np
from fastapi.testclient import TestClient

from src.agent import IncidentAgent
from src.events import EventEngine
from src.main import app
from src.schemas import Detection, Incident, StartRequest


def vehicle(y: float, x: float = 0.4) -> Detection:
    return Detection(label="car",confidence=0.9,box=(x-0.03,y-0.08,x+0.03,y))


def test_direction_requires_persistent_opposite_movement():
    config = StartRequest(scenario="traffic",direction_zone=(0,0,1,1),zone=(0.8,0.1,0.95,0.2))
    for delta, expected in [(0.01,0),(-0.01,1),(0.0,0)]:
        engine=EventEngine(config)
        results=[]
        for i in range(20):
            results.extend(engine.update([vehicle(0.6+i*delta)],i*0.1))
        assert sum(c.kind=="wrong_way" for c in results) == expected


def test_zone_requires_observed_entry_and_dwell():
    config=StartRequest(scenario="traffic",zone=(0.3,0.5,0.6,0.8),direction_zone=(0,0,0.1,0.1))
    engine=EventEngine(config)
    events=[]
    for i in range(20):
        events.extend(engine.update([vehicle(0.4+i*0.01)],i*0.1))
    assert [e.kind for e in events] == ["vehicle_zone_entry"]
    engine=EventEngine(config)
    assert all(not engine.update([vehicle(0.6)],i*0.1) for i in range(20))


def test_default_crowd_has_no_alerts_or_urgency():
    people=[Detection(label="person",confidence=0.9,box=(0.1+i*0.04,0.2,0.13+i*0.04,0.5)) for i in range(15)]
    engine=EventEngine(StartRequest())
    assert all(not engine.update(people,i*0.1,np.zeros((180,320,3),np.uint8)) for i in range(20))
    event=Incident(frame_id=0,detections=people,backend="mock",capture_latency_ms=0,event_type="interaction_candidate")
    alert=asyncio.run(IncidentAgent().run(event))
    assert alert.severity == "info" and alert.action == "none"


def test_disappeared_track_does_not_connect_distant_positions():
    engine=EventEngine(StartRequest(scenario="traffic",direction_zone=(0,0,1,1)))
    first=vehicle(0.8)
    engine.update([first],0)
    second=vehicle(0.2)
    assert engine.update([second],2) == []
    assert first.track_id != second.track_id


def test_default_api_and_direction_validation():
    with TestClient(app) as client:
        assert client.post('/start-stream',json={'direction_zone':[1,0,0,1]}).status_code==422
        response=client.post('/start-stream',json={'source':'synthetic'})
        assert response.status_code==200
        assert response.json()['scenario']=='observe'
