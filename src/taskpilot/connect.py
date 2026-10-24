"""Connect services you already have: REST APIs, OpenAPI specs and remote agents.

A REST endpoint as a tool::

    from taskpilot.connect import http_tool

    find = http_tool("https://crm.example.com/customers/{customer_id}", method="GET",
                     name="find_customer", description="Look up a customer by id",
                     params={"customer_id": str}, headers={"Authorization": "Bearer ..."})

Every operation of an OpenAPI (Swagger) spec as tools::

    from taskpilot.connect import openapi_tools

    tools = openapi_tools("https://api.example.com/openapi.json", headers={"X-API-Key": "..."})
    ws = Workspace(llm, tools, approve=[t.name for t in tools if t.http_method != "GET"])

A remote agent that speaks A2A (Agent2Agent, versions 0.3 and 1.0)::

    from taskpilot.connect import A2AAgent

    reporter = A2AAgent("https://agents.example.com/report")
    reporter.run("Q3 sales summary").output
    Supervisor(llm, [reporter, local_agent])          # works like any agent

Standard library only.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .errors import TaskPilotError, ToolError
from .llm.tools import Tool, type_schema

__all__ = ["http_tool", "openapi_tools", "A2AAgent", "A2AResult"]

_SAFE_RETRY = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


# ====================================================================== HTTP
def _request(method: str, url: str, *, query: Optional[Dict[str, Any]] = None, body: Any = None,
             headers: Optional[Dict[str, str]] = None, timeout: float = 30.0, retries: Optional[int] = None,
             what: str = "") -> Any:
    """Send one request; JSON replies are parsed. Raises ToolError with a readable message."""
    method = method.upper()
    if query:
        pairs = []
        for k, v in query.items():
            if v is None:
                continue
            for item in v if isinstance(v, (list, tuple)) else [v]:
                pairs.append((k, json.dumps(item) if isinstance(item, (dict, bool)) else str(item)))
        if pairs:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(pairs)
    data = None
    h = {"Accept": "application/json, text/plain;q=0.9, */*;q=0.5", "User-Agent": "taskpilot", **(headers or {})}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    # Only repeat requests that are safe to repeat: a retried POST could send an email twice.
    tries = (2 if method in _SAFE_RETRY else 0) if retries is None else max(0, retries)
    label = what or f"{method} {url}"
    for attempt in range(tries + 1):
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                ctype = (resp.headers.get("Content-Type") or "").lower()
            break
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")[:500]
            if exc.code in _RETRY_STATUS and attempt < tries:
                time.sleep(min(8.0, 0.5 * 2 ** attempt))
                continue
            hint = {401: " (check the API key / token)", 403: " (not allowed)", 404: " (not found)"}.get(exc.code, "")
            raise ToolError(f"{label}: HTTP {exc.code}{hint}: {text.strip() or exc.reason}") from None
        except (urllib.error.URLError, OSError) as exc:
            if attempt < tries:
                time.sleep(min(8.0, 0.5 * 2 ** attempt))
                continue
            raise ToolError(f"{label}: could not reach the service: {getattr(exc, 'reason', exc)}") from None
    text = raw.decode("utf-8", "replace")
    if not text.strip():
        return None
    if "json" in ctype or text.lstrip()[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def _pick(value: Any, path: Optional[str]) -> Any:
    if not path:
        return value
    for key in path.split("."):
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and key.lstrip("-").isdigit():
            value = value[int(key)] if -len(value) <= int(key) < len(value) else None
        else:
            return None
    return value


def _fill_path(url: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    rest = dict(args)

    def sub(m: "re.Match[str]") -> str:
        key = m.group(1)
        if key not in rest or rest[key] is None:
            raise ToolError(f"missing value for '{{{key}}}' in {url}")
        return urllib.parse.quote(str(rest.pop(key)), safe="")

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_.-]*)\}", sub, url), rest


def _params_schema(params: Any) -> Dict[str, Any]:
    if params is None:
        return {"type": "object", "properties": {}, "required": []}
    if isinstance(params, dict) and params.get("type") == "object" and isinstance(params.get("properties"), dict):
        return params
    props: Dict[str, Any] = {}
    required: List[str] = []
    for key, tp in dict(params).items():
        if isinstance(tp, dict):
            props[key] = tp
            if "default" not in tp:
                required.append(key)
        else:
            props[key] = type_schema(tp)
            required.append(key)
    return {"type": "object", "properties": props, "required": required}


def http_tool(url: str, *, name: str, description: str = "", method: str = "POST", params: Any = None,
              headers: Optional[Dict[str, str]] = None, timeout: float = 30.0, pick: Optional[str] = None,
              retries: Optional[int] = None) -> Tool:
    """A tool that calls one HTTP endpoint.

    Args:
        url: may contain ``{placeholders}`` filled from the arguments of the same name.
        params: the arguments, as ``{"name": type}`` (``str``, ``int``, ``list``...),
            ``{"name": {json schema}}`` (add ``"default"`` to make it optional),
            or a full JSON object schema.
        method: GET and DELETE send the other arguments as query parameters;
            POST, PUT and PATCH send them as a JSON body.
        pick: return only part of the JSON reply, e.g. ``"data.items"``.
        retries: how often to retry outages and 5xx/429 errors. By default only
            requests that are safe to repeat are retried (not POST or PATCH).
    """
    method = method.upper()
    schema = _params_schema(params)
    for key in re.findall(r"\{([A-Za-z_][A-Za-z0-9_.-]*)\}", url):
        if key not in schema["properties"]:
            schema["properties"][key] = {"type": "string"}
            schema["required"].append(key)

    def call(**args: Any) -> Any:
        full, rest = _fill_path(url, args)
        as_query = method in ("GET", "DELETE", "HEAD")
        value = _request(method, full, query=rest if as_query else None, body=None if as_query else rest,
                         headers=headers, timeout=timeout, retries=retries, what=name)
        return _pick(value, pick)

    t = Tool.from_schema(call, name, description or f"{method} {url}", schema)
    t.http_method = method  # type: ignore[attr-defined]
    return t


# ====================================================================== OpenAPI
def _load_spec(spec: Union[str, Dict[str, Any]], headers: Optional[Dict[str, str]], timeout: float) -> Tuple[Dict[str, Any], str]:
    if isinstance(spec, dict):
        return spec, ""
    if spec.startswith(("http://", "https://")):
        data = _request("GET", spec, headers=headers, timeout=timeout, what="OpenAPI spec")
        if isinstance(data, str):
            raise ToolError(f"{spec} is not a JSON OpenAPI spec (YAML is not supported; use the .json version)")
        return data, spec
    with open(os.fspath(spec), encoding="utf-8") as fh:
        text = fh.read()
    try:
        return json.loads(text), ""
    except ValueError:
        raise ToolError(f"{spec} is not JSON (YAML is not supported without dependencies; convert it to JSON)") from None


class _Refs:
    def __init__(self, root: Dict[str, Any]):
        self.root = root

    def get(self, ref: str) -> Any:
        if not ref.startswith("#/"):
            return {}  # remote refs are not followed
        node: Any = self.root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                return {}
        return node

    def resolve(self, node: Any, depth: int = 0, seen: Tuple[str, ...] = ()) -> Any:
        if depth > 25:
            return {}
        if isinstance(node, dict):
            if isinstance(node.get("$ref"), str):
                ref = node["$ref"]
                if ref in seen:
                    return {"type": "object"}  # a recursive schema: stop here
                merged = {**self.resolve(self.get(ref), depth + 1, seen + (ref,)),
                          **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: self.resolve(v, depth + 1, seen) for k, v in node.items()}
        if isinstance(node, list):
            return [self.resolve(v, depth + 1, seen) for v in node]
        return node


def _base_url(spec: Dict[str, Any], spec_url: str) -> str:
    servers = spec.get("servers") or []
    if servers and isinstance(servers[0], dict) and servers[0].get("url"):
        url = servers[0]["url"]
        for k, v in (servers[0].get("variables") or {}).items():
            url = url.replace("{" + k + "}", str(v.get("default", "")))
        return urllib.parse.urljoin(spec_url, url) if spec_url else url
    if spec.get("host"):  # Swagger 2
        scheme = (spec.get("schemes") or ["https"])[0]
        return f"{scheme}://{spec['host']}{spec.get('basePath', '')}"
    if spec_url:
        parts = urllib.parse.urlsplit(spec_url)
        return f"{parts.scheme}://{parts.netloc}"
    return ""


def openapi_tools(spec: Union[str, Dict[str, Any]], *, base_url: Optional[str] = None,
                  headers: Optional[Dict[str, str]] = None, include: Optional[Iterable[str]] = None,
                  exclude: Iterable[str] = (), prefix: str = "", timeout: float = 30.0) -> List[Tool]:
    """One tool per operation of an OpenAPI 3 / Swagger 2 spec (a dict, a JSON file or a URL).

    Tool names come from ``operationId`` (or method + path). Path, query and
    header parameters and the JSON body's fields become the tool's arguments.
    Every tool has ``http_method``, so you can ask for approval on writes::

        approve=[t.name for t in tools if t.http_method != "GET"]
    """
    doc, spec_url = _load_spec(spec, headers, timeout)
    if not isinstance(doc.get("paths"), dict):
        raise ToolError("This does not look like an OpenAPI spec: it has no 'paths'")
    refs = _Refs(doc)
    root = (base_url or _base_url(doc, spec_url)).rstrip("/")
    if not root:
        raise ToolError("The spec has no server URL; pass base_url=")
    wanted = set(include) if include is not None else None
    skip = set(exclude)
    tools: List[Tool] = []
    taken: set = set()
    for path, item in doc["paths"].items():
        if not isinstance(item, dict):
            continue
        item = refs.resolve(item)
        shared = item.get("parameters") or []
        for method in ("get", "post", "put", "patch", "delete"):
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId") or f"{method}_{path}"
            if op_id in skip or (wanted is not None and op_id not in wanted):
                continue
            t = _operation_tool(op_id, method.upper(), root + path, op, shared, prefix, taken, headers, timeout)
            taken.add(t.name)
            tools.append(t)
    if wanted:
        missing = wanted - {getattr(t, "operation_id") for t in tools}
        if missing:
            raise ToolError(f"No operation(s) {sorted(missing)} in the spec")
    return tools


def _operation_tool(op_id: str, method: str, url: str, op: Dict[str, Any], shared: Sequence[Dict[str, Any]],
                    prefix: str, taken: set, headers: Optional[Dict[str, str]], timeout: float) -> Tool:
    from .mcp import _safe_name

    props: Dict[str, Any] = {}
    required: List[str] = []
    where: Dict[str, Tuple[str, str]] = {}  # tool argument -> (location, real name)
    params = {(p.get("in"), p.get("name")): p for p in list(shared) + list(op.get("parameters") or [])
              if isinstance(p, dict) and p.get("name")}
    body_schema: Optional[Dict[str, Any]] = None
    for (loc, pname), p in params.items():
        if loc == "body":  # Swagger 2
            body_schema = p.get("schema") or {}
            continue
        if loc not in ("path", "query", "header"):
            continue
        arg = re.sub(r"[^A-Za-z0-9_]", "_", pname)
        schema = dict(p.get("schema") or {k: p[k] for k in ("type", "enum", "items", "format") if k in p} or {"type": "string"})
        if p.get("description"):
            schema["description"] = p["description"]
        props[arg] = schema
        where[arg] = (loc, pname)
        if p.get("required") or loc == "path":
            required.append(arg)
    content = (op.get("requestBody") or {}).get("content") or {}
    for ctype, media in content.items():
        if "json" in ctype:
            body_schema = (media or {}).get("schema") or {}
            break
    body_required = bool((op.get("requestBody") or {}).get("required"))
    flat = bool(body_schema and isinstance(body_schema.get("properties"), dict))
    if flat:
        for key, schema in body_schema["properties"].items():  # type: ignore[index]
            arg = key if key not in props else f"body_{key}"
            props[arg] = schema if isinstance(schema, dict) else {}
            where[arg] = ("body", key)
            if key in (body_schema.get("required") or []):  # type: ignore[union-attr]
                required.append(arg)
    elif body_schema is not None:
        props["body"] = body_schema or {"type": "object"}
        where["body"] = ("body", "")
        if body_required:
            required.append("body")
    name = _safe_name(prefix + op_id, taken)
    desc = " ".join(x for x in (op.get("summary"), op.get("description")) if x) or f"{method} {url}"

    def call(**args: Any) -> Any:
        path_vals, query, hdrs, body = {}, {}, dict(headers or {}), {}
        whole: Any = None
        for arg, value in args.items():
            loc, real = where.get(arg, ("query", arg))
            if loc == "path":
                path_vals[real] = value
            elif loc == "query":
                query[real] = value
            elif loc == "header":
                hdrs[real] = str(value)
            elif real:
                body[real] = value
            else:
                whole = value
        full = re.sub(r"\{([^}]+)\}", lambda m: urllib.parse.quote(str(path_vals.get(m.group(1), m.group(0))), safe=""), url)
        payload = whole if whole is not None else (body if (body or flat and method in ("POST", "PUT", "PATCH")) else None)
        return _request(method, full, query=query, body=payload, headers=hdrs, timeout=timeout, what=name)

    t = Tool.from_schema(call, name, desc[:1000], {"type": "object", "properties": props, "required": required})
    t.http_method = method  # type: ignore[attr-defined]
    t.operation_id = op_id  # type: ignore[attr-defined]
    return t


# ====================================================================== A2A
_TERMINAL = {"completed", "failed", "canceled", "rejected"}
_STOPS = _TERMINAL | {"input-required", "auth-required"}


def _state(raw: Any) -> str:
    s = str(raw or "").lower()
    if s.startswith("task_state_"):
        s = s[len("task_state_"):]
    return s.replace("_", "-")


def _parts_text(parts: Any) -> str:
    out = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("text"), str):
            out.append(p["text"])
        elif "data" in p:
            out.append(json.dumps(p["data"], ensure_ascii=False, default=str))
        elif p.get("file") or p.get("url") or p.get("raw"):
            f = p.get("file") or {}
            out.append(f"[file {p.get('filename') or f.get('name') or p.get('url') or f.get('uri') or ''}]")
    return "\n".join(x for x in out if x)


class A2AResult:
    """What :class:`A2AAgent` returns. ``str(result)`` is the answer."""

    def __init__(self, output: str, state: str, task_id: str = "", context_id: str = "", raw: Any = None):
        self.output = output
        #: completed, input-required, failed, working, ... ("completed" for a plain message reply)
        self.state = state
        self.task_id = task_id
        self.context_id = context_id
        self.raw = raw
        self.verified = None
        self.tools_used: List[str] = []

    @property
    def ok(self) -> bool:
        return self.state == "completed"

    @property
    def needs_input(self) -> bool:
        """The remote agent asked a question; answer with ``agent.reply(result, text)``."""
        return self.state in ("input-required", "auth-required")

    def __str__(self) -> str:
        return self.output

    def __repr__(self) -> str:
        return f"A2AResult({self.output[:50]!r}, state={self.state!r})"


class A2AAgent:
    """A remote agent reached with the A2A protocol (v1.0 or v0.3, JSON-RPC binding).

    It reads the agent card at ``/.well-known/agent-card.json`` to find the
    endpoint, name and skills, and then works where a taskpilot agent works:
    ``run()``, ``as_tool()`` and inside ``Supervisor``.

    Args:
        url: the agent's base URL (or the agent card's URL).
        endpoint: the JSON-RPC URL, if the agent has no card.
        version: "1.0" or "0.3"; read from the card when not given.
    """

    def __init__(self, url: str, *, name: Optional[str] = None, description: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None, timeout: float = 120.0, poll: float = 1.0,
                 endpoint: Optional[str] = None, version: Optional[str] = None):
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.poll = poll
        self._name = name
        self._description = description
        self._endpoint = endpoint
        self._version = version
        self._card: Optional[Dict[str, Any]] = None
        self._ids = 0

    # ------------------------------------------------------------ card
    @property
    def card(self) -> Dict[str, Any]:
        if self._card is None:
            self._card = self._fetch_card()
        return self._card

    def _fetch_card(self) -> Dict[str, Any]:
        if self.url.endswith(".json"):
            urls = [self.url]
        else:
            base = self.url.rstrip("/")
            urls = [base + "/.well-known/agent-card.json", base + "/.well-known/agent.json"]
        problem = ""
        for u in urls:
            try:
                card = _request("GET", u, headers=self.headers, timeout=min(self.timeout, 30), retries=1, what="A2A agent card")
            except ToolError as exc:
                problem = str(exc)
                continue
            if isinstance(card, dict) and card.get("name"):
                return card
        if self._endpoint:
            return {"name": self._name or "remote_agent", "description": self._description or ""}
        raise TaskPilotError(f"Could not read the A2A agent card of {self.url}: {problem}. Pass endpoint= if it has none.")

    def _target(self) -> Tuple[str, str]:
        if self._endpoint:
            return self._endpoint, self._version or "1.0"
        card = self.card
        for iface in card.get("supportedInterfaces") or []:  # v1.0
            if isinstance(iface, dict) and str(iface.get("protocolBinding", "")).upper() == "JSONRPC" and iface.get("url"):
                return iface["url"], self._version or str(iface.get("protocolVersion") or "1.0")
        version = self._version or str(card.get("protocolVersion") or "0.3")
        if str(card.get("preferredTransport", "JSONRPC")).upper() == "JSONRPC" and card.get("url"):
            return card["url"], version
        for iface in card.get("additionalInterfaces") or []:
            if isinstance(iface, dict) and str(iface.get("transport", "")).upper() == "JSONRPC" and iface.get("url"):
                return iface["url"], version
        raise TaskPilotError(f"The A2A agent at {self.url} offers no JSON-RPC interface (only {card.get('preferredTransport')})")

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        from .mcp import _safe_name

        return _safe_name(re.sub(r"\W+", "_", str(self.card.get("name", "remote_agent"))).strip("_").lower() or "remote_agent", ())

    @property
    def description(self) -> str:
        if self._description is not None:
            return self._description
        card = self.card
        skills = "; ".join(f"{s.get('name')}: {s.get('description', '')}" for s in card.get("skills") or [] if isinstance(s, dict))
        return " ".join(x for x in (str(card.get("description") or ""), f"Skills: {skills}" if skills else "") if x)

    # ------------------------------------------------------------ calls
    def _rpc(self, method: str, params: Dict[str, Any]) -> Any:
        endpoint, version = self._target()
        self._ids += 1
        hdrs = dict(self.headers)
        if not version.startswith("0."):
            hdrs.setdefault("A2A-Version", version.split(".")[0] + ".0" if version.count(".") > 1 else version)
        reply = _request("POST", endpoint, body={"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params},
                         headers=hdrs, timeout=self.timeout, retries=0, what=f"A2A {self.name}")
        if not isinstance(reply, dict):
            raise ToolError(f"A2A {self.name}: the agent did not answer with JSON-RPC")
        if "error" in reply:
            err = reply["error"] if isinstance(reply["error"], dict) else {"message": str(reply["error"])}
            raise ToolError(f"A2A {self.name}: {err.get('message', err)} (code {err.get('code')})")
        return reply.get("result")

    def _modern(self) -> bool:
        return not self._target()[1].startswith("0.")

    def run(self, task: str, *, context_id: Optional[str] = None, task_id: Optional[str] = None) -> A2AResult:
        """Send a task (a text message) and wait for the answer."""
        modern = self._modern()
        msg: Dict[str, Any] = {"messageId": uuid.uuid4().hex,
                               "role": "ROLE_USER" if modern else "user",
                               "parts": [{"text": task} if modern else {"kind": "text", "text": task}]}
        if not modern:
            msg["kind"] = "message"
        if context_id:
            msg["contextId"] = context_id
        if task_id:
            msg["taskId"] = task_id
        result = self._rpc("SendMessage" if modern else "message/send", {"message": msg})
        out = self._read(result)
        deadline = time.monotonic() + self.timeout
        while out.state not in _STOPS and out.task_id and time.monotonic() < deadline:
            time.sleep(self.poll)
            out = self._read(self._rpc("GetTask" if modern else "tasks/get", {"id": out.task_id}))
        return out

    def reply(self, previous: A2AResult, text: str) -> A2AResult:
        """Answer a question the agent asked (``previous.needs_input``)."""
        return self.run(text, context_id=previous.context_id or None, task_id=previous.task_id or None)

    def _read(self, result: Any) -> A2AResult:
        if not isinstance(result, dict):
            return A2AResult(str(result or ""), "completed", raw=result)
        if "task" in result and isinstance(result["task"], dict):
            result = result["task"]
        elif "message" in result and isinstance(result["message"], dict) and "status" not in result:
            m = result["message"]
            return A2AResult(_parts_text(m.get("parts")), "completed", m.get("taskId", ""), m.get("contextId", ""), result)
        if result.get("kind") == "message" or ("parts" in result and "status" not in result):
            return A2AResult(_parts_text(result.get("parts")), "completed", result.get("taskId", ""),
                             result.get("contextId", ""), result)
        status = result.get("status") or {}
        state = _state(status.get("state"))
        texts = [_parts_text(a.get("parts")) for a in result.get("artifacts") or [] if isinstance(a, dict)]
        said = _parts_text((status.get("message") or {}).get("parts"))
        if not any(texts) or state not in ("completed",):
            texts.append(said)
        output = "\n".join(t for t in texts if t)
        return A2AResult(output, state, result.get("id", ""), result.get("contextId", ""), result)

    def as_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
        """The remote agent as a tool. If it asks a question, the next call answers it."""
        agent = self
        open_task: Dict[str, A2AResult] = {}

        def delegate(task: str) -> str:
            last = open_task.pop("last", None)
            r = agent.reply(last, task) if last is not None else agent.run(task)
            if r.needs_input:
                open_task["last"] = r
                return f"The agent '{agent.name}' asks: {r.output} (call this tool again with the answer)"
            if r.state in ("failed", "rejected", "canceled"):
                raise ToolError(f"The agent '{agent.name}' {r.state} the task: {r.output}")
            return r.output

        about = description or f"Ask the remote agent '{self.name}' to do a task. {self.description}".strip()
        return Tool.from_schema(delegate, name or self.name, about[:1000],
                                {"type": "object", "properties": {"task": {"type": "string",
                                 "description": "a clear, complete instruction"}}, "required": ["task"]})

    def __repr__(self) -> str:
        return f"A2AAgent({self.url!r})"
