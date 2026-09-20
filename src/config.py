"""Server-only configuration; API keys are never returned in health or alerts."""
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from .assets import ROOT

load_dotenv(ROOT / ".env", override=False)


@dataclass
class ModelConfig:
    mode: str
    model: str
    api_key: str = field(repr=False)
    base_url: str = "https://api.groq.com/openai/v1"

    @classmethod
    def read(cls) -> "ModelConfig":
        mode = os.getenv("LLM_MODE", "mock").strip().lower()
        if mode not in {"mock", "openai", "groq"}:
            raise ValueError("LLM_MODE must be mock, openai or groq")
        if mode == "groq":
            return cls(mode, os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"), os.getenv("GROQ_API_KEY", "").strip())
        return cls(mode, os.getenv("LLM_MODEL", "local-model"), os.getenv("OPENAI_API_KEY", "local"),
                   os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"))
