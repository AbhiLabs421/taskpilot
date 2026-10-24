"""MCP (Model Context Protocol): use any MCP server's tools, or serve your own.

Use tools from an MCP server (a tool service of any platform)::

    from taskpilot.mcp import MCPClient

    with MCPClient.stdio("npx -y @modelcontextprotocol/server-filesystem /data") as fs:
        agent = Agent(llm, tools=fs.tools())            # every MCP tool is a normal tool

    crm = MCPClient.http("https://tools.example.com/mcp", headers={"Authorization": "Bearer ..."})
    AutoFlow(llm, crm.tools(include=["find_customer", "create_ticket"]))

Serve your tools, flows and agents to any MCP host (IDEs, other platforms)::

    from taskpilot.mcp import MCPServer

    server = MCPServer([add, send_email, report_flow, writer_agent], name="office")
    server.serve_stdio()                                 # or: server.serve_http(port=8765)

    taskpilot mcp tools.py                               # the same from the command line

Standard library only. Speaks the handshake protocol versions 2024-11-05 to
2025-11-25 over stdio and Streamable HTTP.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import itertools
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .errors import TaskPilotError, ToolError
from .llm.tools import Tool, to_text

__all__ = ["MCPClient", "MCPServer", "MCPError", "PROTOCOL_VERSION", "SUPPORTED_VERSIONS"]

PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


class MCPError(TaskPilotError):
    """An MCP server answered with an error, or could not be reached."""

    def __init__(self, message: str, code: Optional[int] = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


def _version() -> str:
    from . import __version__

    return __version__


def _safe_name(name: str, taken: Iterable[str]) -> str:
    """A name every model API accepts (MCP allows dots and longer names)."""
    base = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "tool"
    if not re.match(r"[A-Za-z_]", base[0]):
        base = "t_" + base
    base = base[:64]
    taken = set(taken)
    out, n = base, 2
    while out in taken:
        suffix = f"_{n}"
        out, n = base[: 64 - len(suffix)] + suffix, n + 1
    return out


# ====================================================================== client
class _Stdio:
    """Runs a server process; newline-delimited JSON-RPC on stdin/stdout."""

    def __init__(self, command: Sequence[str], env: Optional[Dict[str, str]], cwd: Optional[str],
                 on_message: Callable[[Dict[str, Any]], None]):
        self.command = list(command)
        full_env = None if env is None else {**os.environ, **env}
        try:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, env=full_env, cwd=cwd, bufsize=0)
        except OSError as exc:
            raise MCPError(f"Could not start MCP server {' '.join(self.command)!r}: {exc}") from None
        self._write = threading.Lock()
        self.stderr: "deque[str]" = deque(maxlen=30)
        self._on_message = on_message
        threading.Thread(target=self._read_out, daemon=True, name="mcp-stdout").start()
        threading.Thread(target=self._read_err, daemon=True, name="mcp-stderr").start()

    def _read_out(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # a server that prints logs to stdout; skip the line
            for one in msg if isinstance(msg, list) else [msg]:
                if isinstance(one, dict):
                    self._on_message(one)
        self._on_message({"_closed": True})

    def _read_err(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr.append(line.decode("utf-8", "replace").rstrip())

    def send(self, msg: Dict[str, Any]) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except (OSError, ValueError):
                raise MCPError(f"MCP server stopped. {self.why()}") from None

    def why(self) -> str:
        code = self.proc.poll()
        tail = " | ".join(list(self.stderr)[-5:])
        return (f"Exit code {code}. " if code is not None else "") + (f"Its last messages: {tail}" if tail else "")

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass


class _SessionExpired(Exception):
    pass


class _HTTP:
    """Streamable HTTP: every message is a POST; replies come as JSON or as an event stream."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]], timeout: float,
                 on_request: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]):
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.session: Optional[str] = None
        self.version: Optional[str] = None
        self._on_request = on_request

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
             "User-Agent": f"taskpilot/{_version()}", **self.headers}
        if self.session:
            h["Mcp-Session-Id"] = self.session
        if self.version:
            h["MCP-Protocol-Version"] = self.version
        return h

    def _post(self, msg: Dict[str, Any], timeout: float) -> Any:
        req = urllib.request.Request(self.url, data=json.dumps(msg).encode("utf-8"), headers=self._headers(), method="POST")
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            if exc.code == 404 and self.session:
                raise _SessionExpired() from None
            detail = body
            try:
                err = json.loads(body).get("error")
                detail = err.get("message", body) if isinstance(err, dict) else (err or body)
            except (ValueError, AttributeError):
                pass
            hint = {401: " Check the token or API key in headers=.", 403: " Not allowed with these credentials."}.get(exc.code, "")
            raise MCPError(f"HTTP {exc.code} from MCP server {self.url}: {detail}.{hint}", code=exc.code) from None
        except (urllib.error.URLError, OSError) as exc:
            raise MCPError(f"Could not reach MCP server {self.url}: {getattr(exc, 'reason', exc)}") from None

    def request(self, msg: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        resp = self._post(msg, timeout)
        with resp:
            sid = resp.headers.get("Mcp-Session-Id")
            if sid:
                self.session = sid
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/event-stream" in ctype:
                for event in _sse_events(resp):
                    found = self._from_stream(event, msg["id"])
                    if found is not None:
                        return found
                raise MCPError(f"MCP server {self.url} closed the stream without answering '{msg['method']}'")
            raw = resp.read()
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise MCPError(f"MCP server {self.url} sent invalid JSON: {raw[:200]!r}") from None
        for one in data if isinstance(data, list) else [data]:
            if isinstance(one, dict) and one.get("id") == msg["id"] and "method" not in one:
                return one
        raise MCPError(f"MCP server {self.url} did not answer '{msg['method']}'")

    def _from_stream(self, data: str, want: Any) -> Optional[Dict[str, Any]]:
        try:
            msg = json.loads(data)
        except ValueError:
            return None
        for one in msg if isinstance(msg, list) else [msg]:
            if not isinstance(one, dict):
                continue
            if "method" in one:
                if "id" in one:  # the server asks us something mid-call (ping, roots...)
                    reply = self._on_request(one)
                    if reply is not None:
                        self.notify(reply)
                continue
            if one.get("id") == want:
                return one
        return None

    def notify(self, msg: Dict[str, Any]) -> None:
        try:
            self._post(msg, self.timeout).close()
        except _SessionExpired:
            pass

    def close(self) -> None:
        if not self.session:
            return
        req = urllib.request.Request(self.url, headers=self._headers(), method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=5).close()
        except (urllib.error.URLError, OSError):
            pass  # servers may not allow DELETE; the session simply expires
        self.session = None


def _sse_events(resp: Any) -> Iterable[str]:
    data: List[str] = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" ") if line[5:6] == " " else line[5:])
    if data:
        yield "\n".join(data)


class MCPClient:
    """Talks to one MCP server. Make it with :meth:`stdio` or :meth:`http`.

    It connects on first use, is safe to share between threads, and turns the
    server's tools into normal taskpilot :class:`~taskpilot.Tool` objects with
    :meth:`tools`.
    """

    def __init__(self, *, timeout: float = 60.0, name: str = "taskpilot"):
        self.timeout = timeout
        self.client_name = name
        self.server_info: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}
        self.instructions = ""
        self.protocol_version = ""
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self._ready = False
        self._closed = False
        self._pending: Dict[Any, Tuple[threading.Event, List[Dict[str, Any]]]] = {}
        self._stdio: Optional[_Stdio] = None
        self._http: Optional[_HTTP] = None
        self._spawn: Optional[Callable[[], _Stdio]] = None
        self._tool_cache: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------ constructors
    @classmethod
    def stdio(cls, command: Union[str, Sequence[str]], *args: str, env: Optional[Dict[str, str]] = None,
              cwd: Optional[str] = None, timeout: float = 60.0) -> "MCPClient":
        """A server started as a local process (``"python server.py"`` or ``["npx", "-y", "pkg"]``)."""
        cmd = (shlex.split(command, posix=os.name != "nt") if isinstance(command, str) else list(command)) + list(args)
        if not cmd:
            raise MCPError("Give the command that starts the MCP server")
        client = cls(timeout=timeout)
        client._spawn = lambda: _Stdio(cmd, env, cwd, client._on_message)
        return client

    @classmethod
    def http(cls, url: str, *, headers: Optional[Dict[str, str]] = None, timeout: float = 60.0) -> "MCPClient":
        """A server reached over Streamable HTTP (its URL usually ends in ``/mcp``)."""
        if not url.startswith(("http://", "https://")):
            raise MCPError(f"MCP URL must start with http:// or https://, got {url!r}")
        client = cls(timeout=timeout)
        client._http = _HTTP(url, headers, timeout, client._server_request)
        return client

    # ------------------------------------------------------------ plumbing
    def _on_message(self, msg: Dict[str, Any]) -> None:
        if msg.get("_closed"):
            for event, slot in list(self._pending.values()):
                slot.append({"_closed": True})
                event.set()
            return
        if "method" in msg:
            if "id" in msg:
                reply = self._server_request(msg)
                if reply is not None and self._stdio is not None:
                    try:
                        self._stdio.send(reply)
                    except MCPError:
                        pass
            elif msg["method"] == "notifications/tools/list_changed":
                self._tool_cache = None
            return
        waiter = self._pending.get(msg.get("id"))
        if waiter is not None:
            waiter[1].append(msg)
            waiter[0].set()

    def _server_request(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = msg.get("method")
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        if method == "roots/list":
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {"roots": []}}
        return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": METHOD_NOT_FOUND, "message": f"{method} is not supported"}}

    def _exchange(self, method: str, params: Optional[Dict[str, Any]], timeout: Optional[float]) -> Any:
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
        if params is not None:
            msg["params"] = params
        wait = timeout or self.timeout
        if self._http is not None:
            reply = self._http.request(msg, wait)
        else:
            assert self._stdio is not None
            event, slot = threading.Event(), []  # type: ignore[var-annotated]
            self._pending[msg["id"]] = (event, slot)
            try:
                self._stdio.send(msg)
                if not event.wait(wait):
                    self._notify("notifications/cancelled", {"requestId": msg["id"], "reason": "timeout"})
                    raise MCPError(f"MCP server did not answer '{method}' within {wait:g}s")
            finally:
                self._pending.pop(msg["id"], None)
            reply = slot[0]
            if reply.get("_closed"):
                raise MCPError(f"MCP server stopped while answering '{method}'. {self._stdio.why()}")
        if "error" in reply:
            err = reply["error"] if isinstance(reply["error"], dict) else {"message": str(reply["error"])}
            raise MCPError(f"MCP '{method}' failed: {err.get('message', err)}", code=err.get("code"), data=err.get("data"))
        return reply.get("result", {})

    def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if self._http is not None:
            self._http.notify(msg)
        elif self._stdio is not None:
            self._stdio.send(msg)

    def connect(self) -> "MCPClient":
        """Start (or reach) the server and do the handshake. Called for you on first use."""
        with self._lock:
            if self._ready:
                return self
            if self._closed:
                raise MCPError("This MCP client is closed")
            if self._spawn is not None and self._stdio is None:
                self._stdio = self._spawn()
            try:
                result = self._exchange("initialize", {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"roots": {"listChanged": False}},
                    "clientInfo": {"name": self.client_name, "version": _version()},
                }, None)
            except BaseException:
                self._drop_process()
                raise
            self.protocol_version = str(result.get("protocolVersion", ""))
            self.server_info = result.get("serverInfo") or {}
            self.capabilities = result.get("capabilities") or {}
            self.instructions = result.get("instructions") or ""
            if self._http is not None:
                self._http.version = self.protocol_version or None
            self._notify("notifications/initialized")
            self._ready = True
        return self

    def _drop_process(self) -> None:
        if self._stdio is not None:
            self._stdio.close()
            self._stdio = None

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, *, timeout: Optional[float] = None) -> Any:
        """Send any MCP request and return its ``result``."""
        self.connect()
        try:
            return self._exchange(method, params, timeout)
        except _SessionExpired:
            # The server forgot our session (restart, timeout): start a new one, once.
            assert self._http is not None
            with self._lock:
                self._http.session = None
                self._http.version = None
                self._ready = False
            self.connect()
            return self._exchange(method, params, timeout)

    def _paged(self, method: str, key: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        cursor = None
        for _ in range(1000):
            result = self.request(method, {"cursor": cursor} if cursor else {})
            items.extend(x for x in result.get(key) or [] if isinstance(x, dict))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return items

    # ------------------------------------------------------------ tools
    def list_tools(self, *, refresh: bool = False) -> List[Dict[str, Any]]:
        """The server's tool definitions (name, description, inputSchema)."""
        if self._tool_cache is None or refresh:
            self._tool_cache = self._paged("tools/list", "tools")
        return list(self._tool_cache)

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None, *, timeout: Optional[float] = None) -> Any:
        """Call a tool and return its value. Raises :class:`ToolError` when the tool reports an error.

        Structured results come back as dicts; text as a string.
        """
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        value = _tool_value(result)
        if result.get("isError"):
            raise ToolError(f"{name}: {to_text(value)}")
        return value

    def tools(self, *, include: Optional[Iterable[str]] = None, exclude: Iterable[str] = (),
              prefix: str = "") -> List[Tool]:
        """The server's tools as taskpilot tools, ready for an Agent, AutoFlow or Workspace.

        Args:
            include: only these tool names (by their MCP name).
            exclude: leave these out.
            prefix: put before every name (``"crm_"``), so tools of two servers never clash.
        """
        wanted = set(include) if include is not None else None
        skip = set(exclude)
        defs = self.list_tools()
        if wanted:
            unknown = wanted - {d.get("name") for d in defs}
            if unknown:
                raise MCPError(f"The MCP server has no tool(s) {sorted(unknown)}. It has: {sorted(d.get('name') for d in defs)}")
        out: List[Tool] = []
        for d in defs:
            mcp_name = str(d.get("name", ""))
            if not mcp_name or mcp_name in skip or (wanted is not None and mcp_name not in wanted):
                continue
            name = _safe_name(prefix + mcp_name, (t.name for t in out))
            schema = d.get("inputSchema") if isinstance(d.get("inputSchema"), dict) else {}
            desc = d.get("description") or d.get("title") or mcp_name

            def call(_n: str = mcp_name, **args: Any) -> Any:
                return self.call(_n, args)

            t = Tool.from_schema(call, name, desc, schema)
            t.mcp_name = mcp_name  # type: ignore[attr-defined]
            t.annotations = d.get("annotations") or {}  # type: ignore[attr-defined]
            out.append(t)
        return out

    # ------------------------------------------------------------ resources and prompts
    def resources(self) -> List[Dict[str, Any]]:
        return self._paged("resources/list", "resources")

    def read(self, uri: str) -> str:
        """Read a resource; text parts are joined, binary parts are described."""
        result = self.request("resources/read", {"uri": uri})
        parts = []
        for c in result.get("contents") or []:
            parts.append(c["text"] if "text" in c else f"[binary {c.get('mimeType', 'data')}: {c.get('uri', uri)}]")
        return "\n".join(parts)

    def prompts(self) -> List[Dict[str, Any]]:
        return self._paged("prompts/list", "prompts")

    def prompt(self, name: str, arguments: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """A prompt from the server as chat messages (``[{"role", "content"}]``)."""
        result = self.request("prompts/get", {"name": name, "arguments": arguments or {}})
        return [{"role": m.get("role", "user"), "content": _content_text([m.get("content") or {}])}
                for m in result.get("messages") or []]

    def ping(self) -> bool:
        self.request("ping")
        return True

    # ------------------------------------------------------------ lifetime
    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._ready = False
            if self._http is not None:
                self._http.close()
            self._drop_process()

    def __enter__(self) -> "MCPClient":
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            if self._stdio is not None:
                self._stdio.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        where = self._http.url if self._http else " ".join(self._stdio.command if self._stdio else ["(not started)"])
        return f"MCPClient({where!r})"


def _content_text(content: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for c in content:
        kind = c.get("type")
        if kind == "text":
            parts.append(str(c.get("text", "")))
        elif kind == "resource":
            r = c.get("resource") or {}
            parts.append(r["text"] if "text" in r else f"[resource {r.get('uri', '')}]")
        elif kind == "resource_link":
            parts.append(f"[resource {c.get('uri', '')}]")
        elif kind:
            parts.append(f"[{kind} {c.get('mimeType', '')}]".replace(" ]", "]"))
    return "\n".join(parts)


def _tool_value(result: Dict[str, Any]) -> Any:
    if isinstance(result.get("structuredContent"), dict):
        sc = result["structuredContent"]
        # Servers wrap plain values as {"result": value}.
        return sc["result"] if list(sc) == ["result"] else sc
    text = _content_text(result.get("content") or [])
    content = result.get("content") or []
    if len(content) == 1 and content[0].get("type") == "text":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list, int, float)) and not isinstance(parsed, bool):
                return parsed
        except ValueError:
            pass
    return text


# ====================================================================== server
class MCPServer:
    """Offers tools, flows and agents to any MCP host.

    ::

        MCPServer([add, report_flow, writer_agent]).serve_stdio()

    Anything you can pass: a :class:`Tool`, a plain function, a
    :class:`~taskpilot.Flow` (its inputs become the tool's arguments), an
    :class:`~taskpilot.Agent` (takes a ``task``) or an :class:`MCPClient`
    (its tools are passed through, which makes a gateway).
    """

    def __init__(self, tools: Iterable[Any] = (), *, name: str = "taskpilot", version: Optional[str] = None,
                 instructions: str = "", workers: int = 8):
        self.name = name
        self.version = version or _version()
        self.instructions = instructions
        self.tools: Dict[str, Tool] = {}
        self.annotations: Dict[str, Dict[str, Any]] = {}
        self.workers = workers
        for t in tools:
            self.add(t)

    def add(self, thing: Any, name: Optional[str] = None, description: Optional[str] = None,
            annotations: Optional[Dict[str, Any]] = None) -> "MCPServer":
        """Offer one more tool, flow, agent or MCP client's tools."""
        from .flow import Flow
        from .llm.agent import Agent

        if isinstance(thing, MCPClient):
            for t in thing.tools():
                self._put(t, None, getattr(t, "annotations", None))
            return self
        if isinstance(thing, Tool):
            t = thing if not (name or description) else Tool.from_schema(
                thing.fn, name or thing.name, description or thing.description, thing.parameters)
        elif isinstance(thing, Flow):
            t = thing.as_tool(name, description)
        elif isinstance(thing, Agent):
            t = thing.as_tool(name, description)
        elif callable(thing):
            t = Tool(thing, name=name, description=description)
        else:
            raise ToolError(f"Can't offer {type(thing).__name__} over MCP; give a tool, function, Flow, Agent or MCPClient")
        self._put(t, None, annotations)
        return self

    def _put(self, t: Tool, name: Optional[str], annotations: Optional[Dict[str, Any]]) -> None:
        key = name or t.name
        old = self.tools.get(key)
        if old is not None and old.fn is not t.fn:
            raise ToolError(f"Two different tools are named '{key}'")
        self.tools[key] = t
        if annotations:
            self.annotations[key] = dict(annotations)

    @classmethod
    def from_workspace(cls, ws: Any, *, agents: bool = True, run: bool = True, approval_tools: bool = False,
                       name: Optional[str] = None) -> "MCPServer":
        """Offer a :class:`~taskpilot.Workspace`: its tools, its agents, and a ``run_request`` tool.

        Tools that need a person's approval are left out unless
        ``approval_tools=True`` (then the MCP host's own confirmation is the
        approval). Agents are offered without those tools, so they can draft
        but not act. ``run_request`` starts a normal workspace run: anything
        that needs approval waits on the web page or the REST API.
        """
        server = cls(name=name or ws.name, instructions=ws.instructions)
        for tname, t in ws.tools.items():
            if tname in ws.approve and not approval_tools:
                continue
            server._put(t, None, {"destructiveHint": True} if tname in ws.approve else None)
        if agents:
            for aname, spec in ws.agents.items():
                def delegate(task: str, _a: str = aname) -> str:
                    result = ws.build(_a, actions=False).run(task)
                    return result.output if result.verified is not False else "UNVERIFIED: " + result.output

                about = (spec.description or spec.instructions or aname)[:300]
                server._put(Tool.from_schema(delegate, _safe_name("agent_" + aname, server.tools),
                                             f"Ask the '{aname}' agent to do a task. {about}",
                                             {"type": "object", "properties": {"task": {"type": "string",
                                              "description": "a clear, complete instruction"}}, "required": ["task"]}),
                            None, {"readOnlyHint": True})
        if run:
            def run_request(request: str, pattern: str = "", agents: Optional[List[str]] = None) -> Dict[str, Any]:
                chosen, team = pattern, list(agents or [])
                if not chosen:
                    tip = ws.suggest(request)
                    chosen, team = tip["pattern"], tip["agents"]
                r = ws.start(request, chosen, team)
                waiting = [a for a in r.actions if a.get("status") == "waiting"]
                if r.result is not None and r.result.paused:
                    waiting += [{"question": p.question or p.key} for p in r.result.pending]
                out = {"run_id": r.id, "status": r.status, "pattern": chosen, "agents": team, "output": r.output}
                if r.error:
                    out["error"] = r.error
                if waiting:
                    out["waiting_for_approval"] = [{k: a[k] for k in ("tool", "args", "question") if k in a} for a in waiting]
                return out

            from .workspace import PATTERNS

            patterns = list(PATTERNS)
            server._put(Tool.from_schema(run_request, "run_request",
                                         "Do a business request with this workspace's agents and tools. "
                                         "Steps that need a person's approval wait for it (they are not done here).",
                                         {"type": "object", "properties": {
                                             "request": {"type": "string", "description": "what should be done, in plain words"},
                                             "pattern": {"type": "string", "enum": [""] + patterns,
                                                         "description": "workflow type; empty lets the AI choose"},
                                             "agents": {"type": "array", "items": {"type": "string"},
                                                        "description": f"agents to use: {', '.join(ws.agents) or 'none'}"}},
                                          "required": ["request"]}),
                        None, None)
        return server

    # ------------------------------------------------------------ protocol
    def _tool_list(self) -> List[Dict[str, Any]]:
        out = []
        for key, t in self.tools.items():
            schema = {k: v for k, v in t.parameters.items()}
            entry = {"name": key, "description": t.description, "inputSchema": schema}
            if key in self.annotations:
                entry["annotations"] = self.annotations[key]
            out.append(entry)
        return out

    def _call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        t = self.tools.get(name) if isinstance(name, str) else None
        if t is None:
            raise _RPCError(INVALID_PARAMS, f"Unknown tool: {name}. Tools: {', '.join(self.tools)}")
        args = params.get("arguments") or {}
        try:
            value = t.fn(**t.prepare(args))
            if inspect.isawaitable(value):
                value = asyncio.run(_await(value))
        except Exception as exc:  # tool errors go back to the model, they are not protocol errors
            return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
        result: Dict[str, Any] = {"content": [{"type": "text", "text": to_text(value)}], "isError": False}
        if isinstance(value, dict):
            result["structuredContent"] = json.loads(json.dumps(value, default=str))
        return result

    def handle(self, message: Any) -> Optional[Any]:
        """Answer one JSON-RPC message (or a batch). Returns ``None`` for notifications."""
        if isinstance(message, list):
            replies = [r for r in (self.handle(m) for m in message) if r is not None]
            return replies or None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            if isinstance(message, dict) and "method" not in message and ("result" in message or "error" in message):
                return None  # a reply to something we asked; we never ask, so ignore it
            return _error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST, "Invalid request")
        method, rid = message["method"], message.get("id")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if "id" not in message:
            return None  # notifications: initialized, cancelled, ...
        try:
            return {"jsonrpc": "2.0", "id": rid, "result": self._dispatch(method, params)}
        except _RPCError as exc:
            return _error(rid, exc.code, str(exc))
        except Exception as exc:  # never crash the server
            return _error(rid, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _dispatch(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "initialize":
            asked = params.get("protocolVersion")
            info: Dict[str, Any] = {
                "protocolVersion": asked if asked in SUPPORTED_VERSIONS else PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": self.version},
            }
            if self.instructions:
                info["instructions"] = self.instructions
            return info
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self._tool_list()}
        if method == "tools/call":
            return self._call(params)
        if method in ("resources/list", "resources/templates/list", "prompts/list"):
            key = {"resources/list": "resources", "resources/templates/list": "resourceTemplates"}.get(method, "prompts")
            return {key: []}
        if method == "logging/setLevel":
            return {}
        raise _RPCError(METHOD_NOT_FOUND, f"Method not found: {method}")

    # ------------------------------------------------------------ stdio
    def serve_stdio(self, stdin: Any = None, stdout: Any = None) -> None:
        """Serve on stdin/stdout until stdin closes (how MCP hosts start local servers)."""
        reader = stdin if stdin is not None else sys.stdin.buffer
        writer = stdout if stdout is not None else sys.stdout.buffer
        text = isinstance(writer, io.TextIOBase)
        lock = threading.Lock()
        saved = sys.stdout
        if stdout is None:
            sys.stdout = sys.stderr  # a stray print() in a tool must not break the protocol

        def write(reply: Any) -> None:
            data = json.dumps(reply, ensure_ascii=False, default=str) + "\n"
            with lock:
                writer.write(data if text else data.encode("utf-8"))
                writer.flush()

        def work(msg: Any) -> None:
            reply = self.handle(msg)
            if reply is not None:
                write(reply)

        try:
            self._stdio_loop(reader, write, work)
        finally:
            sys.stdout = saved

    def _stdio_loop(self, reader: Any, write: Callable[[Any], None], work: Callable[[Any], None]) -> None:
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="mcp") as pool:
            for line in reader:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", "replace")
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    write(_error(None, PARSE_ERROR, "Parse error"))
                    continue
                if isinstance(msg, dict) and msg.get("method") in ("initialize", "ping"):
                    work(msg)  # answer the handshake in order
                else:
                    pool.submit(work, msg)  # slow tools don't block pings or other calls

    # ------------------------------------------------------------ HTTP
    def handle_http(self, method: str, headers: Dict[str, str], body: bytes, sessions: Optional[set] = None,
                    ) -> Tuple[int, Dict[str, str], bytes]:
        """One Streamable HTTP request → (status, headers, body). Used by :meth:`serve_http` and the web app."""
        if method == "GET":
            return 405, {"Allow": "POST, DELETE"}, b""  # we never push messages on our own
        if method == "DELETE":
            sid = headers.get("mcp-session-id")
            if sessions is not None and sid:
                sessions.discard(sid)
            return 200, {}, b""
        if method != "POST":
            return 405, {"Allow": "POST, DELETE"}, b""
        try:
            msg = json.loads(body.decode("utf-8") or "null")
        except ValueError:
            return 400, {"Content-Type": "application/json"}, json.dumps(_error(None, PARSE_ERROR, "Parse error")).encode()
        out_headers = {"Content-Type": "application/json"}
        first = msg[0] if isinstance(msg, list) and msg else msg
        is_init = isinstance(first, dict) and first.get("method") == "initialize"
        if sessions is not None:
            if is_init:
                sid = secrets.token_hex(16)
                sessions.add(sid)
                out_headers["Mcp-Session-Id"] = sid
            else:
                sid = headers.get("mcp-session-id")
                if sid and sid not in sessions:
                    return 404, out_headers, json.dumps(_error(None, INVALID_REQUEST, "Unknown session; initialize again")).encode()
        reply = self.handle(msg)
        if reply is None:
            return 202, {}, b""
        return 200, out_headers, json.dumps(reply, ensure_ascii=False, default=str).encode("utf-8")

    def serve_http(self, host: str = "127.0.0.1", port: int = 8765, path: str = "/mcp", *,
                   token: Optional[str] = None, allowed_origins: Iterable[str] = ()) -> None:
        """Serve Streamable HTTP at ``http://host:port/mcp`` until Ctrl+C."""
        httpd = self.http_server(host, port, path, token=token, allowed_origins=allowed_origins)
        print(f"MCP server '{self.name}' at http://{host}:{httpd.server_address[1]}{path}  (Ctrl+C to stop)", file=sys.stderr)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()

    def http_server(self, host: str = "127.0.0.1", port: int = 8765, path: str = "/mcp", *,
                    token: Optional[str] = None, allowed_origins: Iterable[str] = ()) -> Any:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        server, sessions, origins = self, set(), set(allowed_origins)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _go(self, method: str) -> None:
                hdrs = {k.lower(): v for k, v in self.headers.items()}
                problem = http_guard(hdrs, token, origins)
                if urllib.parse.urlsplit(self.path).path.rstrip("/") != path.rstrip("/"):
                    status, out, body = 404, {}, b""
                elif problem:
                    status, out, body = problem
                else:
                    length = int(hdrs.get("content-length") or 0)
                    if length > 4_000_000:
                        status, out, body = 413, {}, b""
                    else:
                        status, out, body = server.handle_http(method, hdrs, self.rfile.read(length) if length else b"", sessions)
                self.send_response(status)
                for k, v in out.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                self._go("POST")

            def do_GET(self) -> None:
                self._go("GET")

            def do_DELETE(self) -> None:
                self._go("DELETE")

        return ThreadingHTTPServer((host, port), Handler)

    def __repr__(self) -> str:
        return f"MCPServer({self.name!r}, tools={list(self.tools)})"


def http_guard(headers: Dict[str, str], token: Optional[str], origins: Iterable[str] = ()) -> Optional[Tuple[int, Dict[str, str], bytes]]:
    """Reject requests from other web sites (DNS rebinding) and, with ``token``, without the token."""
    origin = headers.get("origin")
    if origin:
        netloc = urllib.parse.urlsplit(origin).netloc
        if netloc != headers.get("host") and origin not in set(origins):
            return 403, {"Content-Type": "application/json"}, b'{"error": "origin not allowed"}'
    if token:
        given = headers.get("authorization", "")
        if not secrets.compare_digest(given.encode(), f"Bearer {token}".encode()):
            return 401, {"Content-Type": "application/json", "WWW-Authenticate": "Bearer"}, b'{"error": "missing or wrong token"}'
    return None


class _RPCError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _error(rid: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


async def _await(x: Any) -> Any:
    return await x
