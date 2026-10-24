"""Client for any OpenAI-compatible chat API (OpenAI, Groq, vLLM, LM Studio...)."""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any, Dict, Iterator, List, Optional

from ..errors import LLMError
from .base import HTTP, LLM, ChatResponse, Messages, ToolCall, to_messages


class OpenAICompatible(LLM):
    """Chat with any server that speaks the ``/v1/chat/completions`` API.

    ::

        llm = OpenAICompatible("gpt-oss:20b", base_url="http://localhost:11434/v1")
        llm = OpenAICompatible("llama-3.3-70b", base_url="https://api.groq.com/openai/v1", api_key="...")

    ``base_url`` defaults to ``$OPENAI_BASE_URL`` or OpenAI; ``api_key`` to
    ``$OPENAI_API_KEY``. For Azure OpenAI use :meth:`azure`.

    Args:
        query: extra URL query parameters, e.g. ``{"api-version": "..."}``.
        api_key_header: header that carries the key. ``"Authorization"``
            (default) sends ``Bearer <key>``; any other name sends the key as is.
        headers: extra HTTP headers (e.g. ``{"OpenAI-Organization": "..."}``).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        retries: int = 3,
        options: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        query: Optional[Dict[str, str]] = None,
        api_key_header: str = "Authorization",
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        h = dict(headers or {})
        if key:
            h[api_key_header] = f"Bearer {key}" if api_key_header.lower() == "authorization" else key
        self.options = dict(options or {})
        self._query = ("?" + urllib.parse.urlencode(query)) if query else ""
        self._http = HTTP(h, timeout, retries)

    @classmethod
    def azure(
        cls,
        deployment: str,
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-10-21",
        **kwargs: Any,
    ) -> "OpenAICompatible":
        """Azure OpenAI: ``OpenAICompatible.azure("my-gpt4o", endpoint="https://NAME.openai.azure.com")``.

        ``endpoint`` defaults to ``$AZURE_OPENAI_ENDPOINT`` and ``api_key`` to
        ``$AZURE_OPENAI_API_KEY``.
        """
        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise LLMError("Azure OpenAI needs endpoint= (or $AZURE_OPENAI_ENDPOINT), e.g. https://NAME.openai.azure.com")
        key = api_key if api_key is not None else os.environ.get("AZURE_OPENAI_API_KEY")
        return cls(
            deployment,
            base_url=f"{endpoint.rstrip('/')}/openai/deployments/{deployment}",
            api_key=key or "",
            query={"api-version": api_version},
            api_key_header="api-key",
            **kwargs,
        )

    @property
    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions{self._query}"

    def _payload(self, messages: List[Dict[str, Any]], tools: Optional[List[dict]], format: Any, options: Optional[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, **self.options, **(options or {})}
        if stream:
            payload["stream"] = True
        if tools:
            payload["tools"] = tools
        if isinstance(format, dict):
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": format}}
        elif format == "json":
            payload["response_format"] = {"type": "json_object"}
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
        data = self._http.post(self._chat_url, self._payload(msgs, tools, format, options, False))
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected reply from {self.base_url}: {str(data)[:300]}") from None
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {"_raw": args}
            calls.append(ToolCall(fn.get("name", ""), args, c.get("id", "")))
        message: Dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            message["tool_calls"] = msg["tool_calls"]
        usage = data.get("usage") or {}
        return ChatResponse(
            msg.get("content") or "",
            thinking=msg.get("reasoning_content") or msg.get("reasoning") or "",
            tool_calls=calls,
            message=message,
            usage={"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
            model=data.get("model", self.model),
            raw=data,
        )

    def stream(self, messages: Messages, *, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        msgs = to_messages(messages, system)
        for line in self._http.post_lines(self._chat_url, self._payload(msgs, None, None, options, True)):
            text = line.decode("utf-8", "replace")
            if not text.startswith("data:"):
                continue
            body = text[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
                piece = chunk["choices"][0].get("delta", {}).get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if piece:
                yield piece

    def tool_message(self, call: ToolCall, content: str) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    def __repr__(self) -> str:
        return f"OpenAICompatible(model={self.model!r}, base_url={self.base_url!r})"
