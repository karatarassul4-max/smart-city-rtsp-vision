import pytest


@pytest.fixture(autouse=True)
def offline_reports(monkeypatch):
    """Tests must never spend API credits or send images from a local .env."""
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("GROQ_API_KEY", "")
