import asyncio
import base64
import json

import httpx
import pytest

from src.agent import IncidentAgent
from src.schemas import Detection, Incident


def incident() -> Incident:
    return Incident(frame_id=42, detections=[Detection(label="person", confidence=0.8,
                    box=(0.4, 0.2, 0.5, 0.8))], backend="onnx:CPUExecutionProvider",
                    capture_latency_ms=80, source_kind="recording")


def configure(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-placeholder")


def test_groq_sends_exact_incident_image_and_validates_assessment(monkeypatch):
    configure(monkeypatch)
    calls = []
    jpeg = b"test-incident-jpeg"

    async def post(self, url, **kwargs):
        calls.append(kwargs["json"])
        assert url == "https://api.groq.com/openai/v1/chat/completions"
        image_url = kwargs["json"]["messages"][0]["content"][1]["image_url"]["url"]
        assert base64.b64decode(image_url.split(",", 1)[1]) == jpeg
        content = json.dumps({"assessment": {"verdict": "not_confirmed", "explanation": "Нет видимого человека."},
                              "report": "Детектор мог ошибиться. Проверьте кадр."})
        return httpx.Response(200, request=httpx.Request("POST", url),
                              json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    agent = IncidentAgent()
    first = asyncio.run(agent.run(incident(), jpeg))
    second = asyncio.run(agent.run(incident(), jpeg))
    assert first.report_source == "vlm"
    assert first.vision_assessment.verdict == "not_confirmed"
    assert first.action == "none"
    assert first.severity == "info"
    assert second.fallback_reason == "rate_limited_locally"
    assert len(calls) == 1
    assert "unit-test-placeholder" not in first.model_dump_json()
    assert "unit-test-placeholder" not in json.dumps(agent.status())
    assert "unit-test-placeholder" not in repr(agent.config)


@pytest.mark.parametrize("code,reason", [(401,"invalid_api_key"), (429,"provider_rate_limit"), (404,"model_not_available")])
def test_provider_errors_fall_back_without_retries(monkeypatch, code, reason):
    configure(monkeypatch)

    async def post(self, url, **kwargs):
        return httpx.Response(code, request=httpx.Request("POST", url), json={"error": "private-provider-body"})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    agent = IncidentAgent()
    alert = asyncio.run(agent.run(incident(), b"jpeg"))
    assert alert.report_source == "mock_fallback"
    assert alert.fallback_reason == reason
    assert alert.vision_assessment is None
    assert agent.status()["report_retry_in_seconds"] >= 59
    assert "private-provider-body" not in alert.model_dump_json()


def test_invalid_visual_json_and_missing_image(monkeypatch):
    configure(monkeypatch)

    async def post(self, url, **kwargs):
        return httpx.Response(200, request=httpx.Request("POST", url),
                              json={"choices": [{"message": {"content": '{"report":"missing assessment"}'}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    agent = IncidentAgent()
    assert asyncio.run(agent.run(incident())).fallback_reason == "missing_incident_image"
    result = asyncio.run(agent.run(incident(), b"jpeg"))
    assert result.fallback_reason == "invalid_provider_response"


def test_missing_key_never_calls_provider(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")

    async def post(*args, **kwargs):
        pytest.fail("Missing key must not issue an HTTP request")

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    assert asyncio.run(IncidentAgent().run(incident(), b"jpeg")).fallback_reason == "missing_api_key"


def test_temporal_evidence_is_one_ordered_contact_sheet(monkeypatch):
    import cv2
    import numpy as np
    configure(monkeypatch)
    images=[cv2.imencode('.jpg',np.full((90,160,3),color,np.uint8))[1].tobytes()
            for color in [(0,0,255),(0,255,0),(255,0,0)]]
    event=incident()
    event.event_type='wrong_way'
    event.evidence_times=[0,1,2]
    payload=IncidentAgent()._payload({'incident':event,'images':images,'jpeg':images[-1]})
    content=payload['messages'][0]['content']
    assert len([item for item in content if item['type']=='image_url'])==1
    data=base64.b64decode(content[1]['image_url']['url'].split(',',1)[1])
    sheet=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    height=sheet.shape[0]//3
    assert sheet[height//2,100,2]>240
    assert sheet[height+height//2,100,1]>240
    assert sheet[2*height+height//2,100,0]>240


def test_uncertain_interaction_never_becomes_urgent(monkeypatch):
    configure(monkeypatch)
    async def post(self,url,**kwargs):
        content=json.dumps({'assessment':{'verdict':'uncertain','explanation':'Could be a hug'},'report':'Needs context'})
        return httpx.Response(200,request=httpx.Request('POST',url),json={'choices':[{'message':{'content':content}}]})
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    event=incident();event.event_type='interaction_candidate'
    alert=asyncio.run(IncidentAgent().run(event,b'jpeg'))
    assert alert.severity=='info' and alert.action=='none'
