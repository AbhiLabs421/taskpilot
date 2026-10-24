"""LLM clients, tools and agents – all on the standard library."""

from .agent import Agent, AgentResult, AgentStep
from .base import LLM, ChatResponse, ToolCall, check_schema, extract_json
from .fake import FakeLLM
from .fallback import FallbackLLM
from .ollama import Ollama
from .openai import OpenAICompatible
from .tools import Tool, tool

__all__ = [
    "LLM", "ChatResponse", "ToolCall", "Ollama", "OpenAICompatible", "FakeLLM", "FallbackLLM",
    "Agent", "AgentResult", "AgentStep", "Tool", "tool", "extract_json", "check_schema",
]


def from_env(model=None):
    """Pick an LLM from environment variables, so no code is needed.

    ``TASKPILOT_MODEL`` names the model. Then the first match wins:
    ``AZURE_OPENAI_ENDPOINT`` (Azure), ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``
    (OpenAI or compatible), ``OLLAMA_API_KEY`` / ``OLLAMA_HOST`` (Ollama),
    otherwise a local Ollama at ``http://localhost:11434``.
    """
    import os

    model = model or os.environ.get("TASKPILOT_MODEL")
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        if not model:
            raise ValueError("Set TASKPILOT_MODEL to your Azure deployment name")
        return OpenAICompatible.azure(model)
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return OpenAICompatible(model or "gpt-4o-mini")
    return Ollama(model or "gpt-oss:120b")


__all__.append("from_env")
