"""Ollama client – local (``ollama serve``) or Ollama Cloud (``https://ollama.com``)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional

from ..errors import LLMError
from .base import HTTP, LLM, ChatResponse, Messages, ToolCall, to_messages

LOCAL_HOST = "http://localhost:11434"
CLOUD_HOST = "https://ollama.com"


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if "://" not in host:
        host = "http://" + host
    # OLLAMA_HOST is often the server's bind address; a client can't use 0.0.0.0.
    return host.replace("://0.0.0.0", "://127.0.0.1")


class Ollama(LLM):
    """Chat with any Ollama model, e.g. ``gpt-oss:20b`` or ``gpt-oss:120b``.

    ::

        llm = Ollama("gpt-oss:120b")              # uses OLLAMA_API_KEY if set
        llm.ask("Capital of India?")              # 'New Delhi'

    Args:
        model: model name.
        host: server URL. Default: ``$OLLAMA_HOST``, else Ollama Cloud when an
            API key is set, else ``http://localhost:11434``.
        api_key: Ollama Cloud key. Default: ``$OLLAMA_API_KEY``.
        timeout: seconds per HTTP request.
        retries: extra tries on network errors, 429 and 5xx.
        options: default model options, e.g. ``{"temperature": 0}``.
        think: ``True``/``False`` or ``"low"|"medium"|"high"`` for reasoning models.
        keep_alive: how long the server keeps the model loaded, e.g. ``"5m"``.
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        *,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        retries: int = 3,
        options: Optional[Dict[str, Any]] = None,
        think: Any = None,
        keep_alive: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY")
        self.host = _normalize_host(host or os.environ.get("OLLAMA_HOST") or (CLOUD_HOST if key else LOCAL_HOST))
        self.options = dict(options or {})
        self.think = think
        self.keep_alive = keep_alive
        h = dict(headers or {})
        if key:
            h["Authorization"] = f"Bearer {key}"
        self._http = HTTP(h, timeout, retries)

    def _payload(self, messages: List[Dict[str, Any]], tools: Optional[List[dict]], format: Any, options: Optional[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        if format is not None:
            payload["format"] = format
        opts = {**self.options, **(options or {})}
        if opts:
            payload["options"] = opts
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

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
        data = self._http.post(f"{self.host}/api/chat", self._payload(msgs, tools, format, options, False))
        if not isinstance(data, dict):
            raise LLMError(f"Unexpected reply from Ollama: {data!r}")
        if data.get("error"):
            raise LLMError(f"Ollama error: {data['error']}")
        msg = data.get("message") or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {"_raw": args}
            calls.append(ToolCall(fn.get("name", ""), args, c.get("id") or f"call_{i}"))
        message = {"role": "assistant", "content": msg.get("content", "") or ""}
        if msg.get("thinking"):
            message["thinking"] = msg["thinking"]
        if msg.get("tool_calls"):
            message["tool_calls"] = msg["tool_calls"]
        return ChatResponse(
            msg.get("content", "") or "",
            thinking=msg.get("thinking", "") or "",
            tool_calls=calls,
            message=message,
            usage={"prompt_tokens": data.get("prompt_eval_count", 0) or 0, "completion_tokens": data.get("eval_count", 0) or 0},
            model=data.get("model", self.model),
            raw=data,
        )

    def stream(self, messages: Messages, *, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        """Yield the reply text piece by piece as the model writes it."""
        msgs = to_messages(messages, system)
        for line in self._http.post_lines(f"{self.host}/api/chat", self._payload(msgs, None, None, options, True)):
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if chunk.get("error"):
                raise LLMError(f"Ollama error: {chunk['error']}")
            piece = (chunk.get("message") or {}).get("content")
            if piece:
                yield piece
            if chunk.get("done"):
                break

    def embed(self, texts: Any, *, model: Optional[str] = None) -> List[List[float]]:
        """Embedding vectors for one text or a list of texts."""
        items = [texts] if isinstance(texts, str) else list(texts)
        data = self._http.post(f"{self.host}/api/embed", {"model": model or self.model, "input": items})
        return data.get("embeddings", [])

    def models(self) -> List[str]:
        """Names of the models available on the server."""
        data = self._http.get(f"{self.host}/api/tags")
        return [m.get("name", "") for m in data.get("models", [])]

    def __repr__(self) -> str:
        return f"Ollama(model={self.model!r}, host={self.host!r})"
