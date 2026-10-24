"""Shared pieces for LLM clients: responses, HTTP with retries, JSON handling."""

from __future__ import annotations

import asyncio
import json
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from ..errors import LLMError

Messages = Union[str, Sequence[Dict[str, Any]]]

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524})


class ToolCall:
    """A request from the model to call one tool."""

    __slots__ = ("id", "name", "arguments")

    def __init__(self, name: str, arguments: Optional[Dict[str, Any]] = None, id: str = ""):
        self.name = name
        self.arguments = arguments or {}
        self.id = id or f"call_{name}"

    def __repr__(self) -> str:
        return f"ToolCall({self.name!r}, {self.arguments!r})"


class ChatResponse:
    """One reply from a model.

    ``str(response)`` is the text. ``message`` is the reply in the provider's
    own format, ready to append to a conversation.
    """

    __slots__ = ("content", "thinking", "tool_calls", "message", "usage", "model", "raw")

    def __init__(
        self,
        content: str = "",
        *,
        thinking: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
        message: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
        model: str = "",
        raw: Any = None,
    ):
        self.content = content or ""
        self.thinking = thinking or ""
        self.tool_calls = tool_calls or []
        self.message = message if message is not None else {"role": "assistant", "content": self.content}
        self.usage = usage or {}
        self.model = model
        self.raw = raw

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        text = self.content if len(self.content) < 60 else self.content[:57] + "..."
        calls = f", tool_calls={self.tool_calls}" if self.tool_calls else ""
        return f"ChatResponse({text!r}{calls})"


# ------------------------------------------------------------------ HTTP
def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


class HTTP:
    """Tiny JSON-over-HTTP client on top of :mod:`urllib` with retries."""

    def __init__(self, headers: Dict[str, str], timeout: float, retries: int):
        self.headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "taskpilot", **headers}
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self._ctx = _ssl_context()

    def _open(self, method: str, url: str, payload: Optional[dict]) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        attempt = 0
        while True:
            attempt += 1
            req = urllib.request.Request(url, data=data, method=method, headers=self.headers)
            try:
                ctx = self._ctx if url.startswith("https") else None
                return urllib.request.urlopen(req, timeout=self.timeout, context=ctx)
            except urllib.error.HTTPError as exc:
                body = _read_body(exc)
                if exc.code in RETRY_STATUS and attempt <= self.retries:
                    time.sleep(_backoff(attempt, exc.headers.get("Retry-After") if exc.headers else None))
                    continue
                raise LLMError(_explain(exc.code, body, url), status=exc.code, body=body) from None
            except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError) as exc:
                if attempt <= self.retries:
                    time.sleep(_backoff(attempt))
                    continue
                reason = getattr(exc, "reason", exc)
                raise LLMError(f"Could not reach {url}: {reason}. Is the server running and the host correct?") from None

    def post(self, url: str, payload: dict) -> Any:
        with self._open("POST", url, payload) as resp:
            return _parse(resp.read(), url)

    def get(self, url: str) -> Any:
        with self._open("GET", url, None) as resp:
            return _parse(resp.read(), url)

    def post_lines(self, url: str, payload: dict) -> Iterator[bytes]:
        """POST and yield the response body line by line (for streaming)."""
        resp = self._open("POST", url, payload)
        try:
            for line in resp:
                line = line.strip()
                if line:
                    yield line
        finally:
            resp.close()


def _read_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:2000]
    except Exception:
        return ""


def _parse(raw: bytes, url: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise LLMError(f"{url} returned invalid JSON: {raw[:200]!r}") from None


def _backoff(attempt: int, retry_after: Optional[str] = None) -> float:
    if retry_after:
        try:
            return min(30.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
    base = min(20.0, 0.5 * 2 ** (attempt - 1))
    return base + random.uniform(0, base * 0.25)


def _explain(status: int, body: str, url: str) -> str:
    detail = body
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            err = parsed.get("error", parsed)
            detail = err.get("message", err) if isinstance(err, dict) else err
    except ValueError:
        pass
    hint = {
        401: " Check your API key.",
        403: " The API key is not allowed to use this resource.",
        404: " Check the model name and the host URL.",
        429: " Rate limited – slow down or upgrade your plan.",
    }.get(status, "")
    return f"HTTP {status} from {url}: {detail}.{hint}"


# ------------------------------------------------------------------ JSON
def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and extra prose."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty reply")
    try:
        return json.loads(text)
    except ValueError:
        pass
    if "```" in text:
        for block in text.split("```")[1::2]:
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except ValueError:
                continue
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
                return value
            except ValueError:
                continue
    raise ValueError("no JSON object found in the reply")


_TYPES = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
    "array": list, "object": dict, "null": type(None),
}


def check_schema(value: Any, schema: Optional[dict], path: str = "$") -> List[str]:
    """Light JSON-schema check (type, required, properties, items, enum).

    Returns a list of problems; empty means the value fits.
    """
    if not schema or not isinstance(schema, dict):
        return []
    problems: List[str] = []
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path} must be one of {schema['enum']}")
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        ok = False
        for name in types:
            py = _TYPES.get(name)
            if py is None:
                ok = True
                break
            if isinstance(value, bool) and name in ("integer", "number"):
                continue
            if isinstance(value, py):  # type: ignore[arg-type]
                ok = True
                break
        if not ok:
            problems.append(f"{path} should be {t}, got {type(value).__name__}")
            return problems
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path} is missing required key '{key}'")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                problems.extend(check_schema(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            problems.extend(check_schema(item, schema["items"], f"{path}[{i}]"))
    return problems


# ------------------------------------------------------------------ base
def to_messages(prompt: Messages, system: Optional[str] = None) -> List[Dict[str, Any]]:
    if isinstance(prompt, str):
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    else:
        messages = [dict(m) for m in prompt]
    if system and not (messages and messages[0].get("role") == "system"):
        messages.insert(0, {"role": "system", "content": system})
    return messages


class LLM:
    """Base class for chat models. Subclasses implement :meth:`chat`."""

    model: str = ""

    # -- to implement -------------------------------------------------
    def chat(
        self,
        messages: Messages,
        *,
        tools: Optional[List[dict]] = None,
        format: Any = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        raise NotImplementedError

    def stream(self, messages: Messages, *, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        """Yield the reply in pieces. Default: one piece (no streaming)."""
        yield self.chat(messages, system=system, options=options).content

    def tool_message(self, call: ToolCall, content: str) -> Dict[str, Any]:
        """A message reporting a tool's result, in this provider's format."""
        return {"role": "tool", "content": content, "tool_name": call.name}

    # -- conveniences -------------------------------------------------
    def ask(self, prompt: Messages, *, system: Optional[str] = None, **options: Any) -> str:
        """Send a prompt, get the reply text."""
        return self.chat(prompt, system=system, options=options or None).content

    def json(
        self,
        prompt: Messages,
        *,
        schema: Optional[dict] = None,
        system: Optional[str] = None,
        repair: int = 2,
        **options: Any,
    ) -> Any:
        """Ask for JSON and return it parsed.

        If the reply is not valid JSON (or does not match ``schema``), the
        model is told what was wrong and asked again, up to ``repair`` times.
        """
        messages = to_messages(prompt, system)
        if schema is not None:
            hint = f"Reply with JSON only, matching this JSON schema:\n{json.dumps(schema)}"
        else:
            hint = "Reply with valid JSON only, no prose."
        messages[-1] = dict(messages[-1])
        messages[-1]["content"] = f"{messages[-1].get('content', '')}\n\n{hint}"
        last_error = ""
        for _ in range(repair + 1):
            resp = self.chat(messages, format=schema if schema is not None else "json", options=options or None)
            try:
                value = extract_json(resp.content)
                problems = check_schema(value, schema)
                if not problems:
                    return value
                last_error = "; ".join(problems[:5])
            except ValueError as exc:
                last_error = str(exc)
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": f"That reply was not valid: {last_error}. Reply again with corrected JSON only."},
            ]
        raise LLMError(f"Model did not return valid JSON after {repair + 1} tries: {last_error}")

    # -- async ---------------------------------------------------------
    async def achat(self, messages: Messages, **kw: Any) -> ChatResponse:
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self.chat(messages, **kw))

    async def aask(self, prompt: Messages, **kw: Any) -> str:
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self.ask(prompt, **kw))

    async def ajson(self, prompt: Messages, **kw: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self.json(prompt, **kw))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
