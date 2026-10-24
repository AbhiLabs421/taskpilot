"""Keep working when a model is down: try the next model in line."""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..errors import LLMError
from .base import LLM, ChatResponse, Messages, ToolCall


class FallbackLLM(LLM):
    """Use the first model that answers.

    ::

        llm = FallbackLLM([Ollama("gpt-oss:120b"), Ollama("gpt-oss:20b", host="http://localhost:11434")])

    If a model fails (network down, rate limit, out of credits, bad reply),
    the same request goes to the next one. Only if every model fails is an
    :class:`LLMError` raised, listing each failure. Keep models of the same
    provider family in one chain, because tool-call formats differ between
    providers.
    """

    def __init__(self, models: Sequence[LLM]):
        models = list(models)
        if not models or not all(isinstance(m, LLM) for m in models):
            raise ValueError("FallbackLLM needs a list of LLM objects")
        self.models = models
        self.model = " | ".join(getattr(m, "model", "?") for m in models)
        self._local = threading.local()

    @property
    def last(self) -> LLM:
        """The model that answered the latest request on this thread."""
        return getattr(self._local, "last", self.models[0])

    def chat(self, messages: Messages, *, tools: Optional[List[dict]] = None, format: Any = None,
             options: Optional[Dict[str, Any]] = None, system: Optional[str] = None) -> ChatResponse:
        failures = []
        for m in self.models:
            try:
                resp = m.chat(messages, tools=tools, format=format, options=options, system=system)
                self._local.last = m
                return resp
            except LLMError as exc:
                failures.append(f"{getattr(m, 'model', m)}: {exc}")
        raise LLMError("Every model failed:\n- " + "\n- ".join(failures))

    def stream(self, messages: Messages, *, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        failures = []
        for m in self.models:
            started = False
            try:
                for piece in m.stream(messages, system=system, options=options):
                    started = True
                    yield piece
                self._local.last = m
                return
            except LLMError as exc:
                if started:  # can't switch models half-way through an answer
                    raise
                failures.append(f"{getattr(m, 'model', m)}: {exc}")
        raise LLMError("Every model failed:\n- " + "\n- ".join(failures))

    def tool_message(self, call: ToolCall, content: str) -> Dict[str, Any]:
        return self.last.tool_message(call, content)

    def __repr__(self) -> str:
        return f"FallbackLLM({self.models!r})"
