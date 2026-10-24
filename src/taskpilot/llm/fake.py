"""A scripted model for tests and offline development – no network needed."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .base import LLM, ChatResponse, Messages, ToolCall, to_messages

Reply = Union[str, dict, ChatResponse]


class FakeLLM(LLM):
    """Returns replies you script, and records every call.

    ::

        llm = FakeLLM(["first reply", "second reply"])
        llm = FakeLLM(fn=lambda messages, tools: "echo: " + messages[-1]["content"])

    A reply can be a string, a :class:`ChatResponse`, or a dict asking for a
    tool: ``{"tool": "add", "args": {"a": 1, "b": 2}}`` (or
    ``{"tool_calls": [{"name": ..., "arguments": {...}}, ...]}``).
    When the list runs out, the last reply repeats.
    """

    def __init__(self, replies: Optional[Sequence[Reply]] = None, *, fn: Optional[Callable[[List[dict], Optional[List[dict]]], Reply]] = None, model: str = "fake"):
        if replies is None and fn is None:
            replies = ["ok"]
        self.replies = list(replies or [])
        self.fn = fn
        self.model = model
        self.calls: List[Dict[str, Any]] = []
        self._i = 0
        self._lock = threading.Lock()

    def chat(
        self,
        messages: Messages,
        *,
        tools: Optional[List[dict]] = None,
        format: Any = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        msgs = to_messages(messages, system)
        with self._lock:
            self.calls.append({"messages": msgs, "tools": tools, "format": format, "options": options})
            if self.fn is None:
                reply = self.replies[min(self._i, len(self.replies) - 1)]
                self._i += 1
        if self.fn is not None:
            reply = self.fn(msgs, tools)  # outside the lock, so parallel calls really overlap
        return _to_response(reply)


def _to_response(reply: Reply) -> ChatResponse:
    if isinstance(reply, ChatResponse):
        return reply
    if isinstance(reply, dict):
        raw_calls = reply.get("tool_calls")
        if raw_calls is None and "tool" in reply:
            raw_calls = [{"name": reply["tool"], "arguments": reply.get("args", {})}]
        if raw_calls is not None:
            calls = [ToolCall(c["name"], c.get("arguments", {}), c.get("id", f"call_{i}")) for i, c in enumerate(raw_calls)]
            message = {
                "role": "assistant",
                "content": reply.get("content", ""),
                "tool_calls": [{"id": c.id, "function": {"name": c.name, "arguments": c.arguments}} for c in calls],
            }
            return ChatResponse(reply.get("content", ""), tool_calls=calls, message=message, model="fake")
        return ChatResponse(json.dumps(reply), model="fake")
    return ChatResponse(str(reply), model="fake")
