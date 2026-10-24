"""Stress-test a workflow against a bad model before your users find out.

:class:`ChaosLLM` wraps a real model and makes it misbehave on purpose:
outages, changed numbers, broken JSON. Run your evals with it to see whether
your guards (checks, verification, plan review, fallbacks) stop the damage.

::

    from taskpilot.testing import ChaosLLM
    shaky = ChaosLLM(Ollama("gpt-oss:120b"), outage=0.1, numbers=0.25, seed=1)
"""

from __future__ import annotations

import json
import random
import re
import threading
from typing import Any, Dict, List, Optional

from .errors import LLMError
from .llm.base import LLM, ChatResponse, Messages, ToolCall
from .llm.fake import FakeLLM

__all__ = ["ChaosLLM", "FakeLLM"]

_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


class ChaosLLM(LLM):
    """A model that fails and makes mistakes on purpose.

    Args:
        inner: the real model.
        outage: chance a call fails like a network/server error.
        numbers: chance one number in the reply (text or tool arguments) is changed.
        broken_json: chance a reply is cut in half (invalid JSON for planners).
        seed: makes the chaos repeatable.
    """

    def __init__(self, inner: LLM, *, outage: float = 0.0, numbers: float = 0.0, broken_json: float = 0.0,
                 seed: Optional[int] = None):
        self.inner = inner
        self.model = f"chaos({getattr(inner, 'model', '?')})"
        self.outage = outage
        self.numbers = numbers
        self.broken_json = broken_json
        self._rnd = random.Random(seed)
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "outages": 0, "numbers": 0, "broken_json": 0}

    def _roll(self, p: float) -> bool:
        with self._lock:
            return p > 0 and self._rnd.random() < p

    def _bump(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _change(self, text: str) -> str:
        found = list(_NUM.finditer(text))
        if not found:
            return text
        with self._lock:
            m = self._rnd.choice(found)
            factor = self._rnd.choice([10, 0.1, 2])
        old = float(m.group(0))
        new = old * factor if old else 7
        shown = f"{new:g}" if "." in m.group(0) or new != int(new) else str(int(new))
        return text[: m.start()] + shown + text[m.end():]

    def chat(self, messages: Messages, *, tools: Optional[List[dict]] = None, format: Any = None,
             options: Optional[Dict[str, Any]] = None, system: Optional[str] = None) -> ChatResponse:
        self._bump("calls")
        if self._roll(self.outage):
            self._bump("outages")
            raise LLMError("HTTP 503 (simulated outage)", status=503)
        resp = self.inner.chat(messages, tools=tools, format=format, options=options, system=system)
        content, calls = resp.content, resp.tool_calls
        if self._roll(self.numbers):
            if calls:
                call = calls[0]
                changed = json.loads(self._change(json.dumps(call.arguments)))
                calls = [ToolCall(call.name, changed, call.id)] + list(calls[1:])
            else:
                content = self._change(content)
            self._bump("numbers")
        if content and self._roll(self.broken_json):
            content = content[: max(1, len(content) // 2)]
            self._bump("broken_json")
        message = dict(resp.message)
        message["content"] = content
        if calls is not resp.tool_calls and "tool_calls" in message:
            message["tool_calls"] = [{"id": c.id, "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
        return ChatResponse(content, thinking=resp.thinking, tool_calls=calls, message=message,
                            usage=resp.usage, model=self.model, raw=resp.raw)

    def tool_message(self, call: ToolCall, content: str) -> Dict[str, Any]:
        return self.inner.tool_message(call, content)
