"""The business-user web page, a REST API and an MCP endpoint, on one server.

::

    taskpilot serve tools.py                    # then open http://127.0.0.1:8000
    taskpilot serve tools.py --token SECRET     # API and page need "Authorization: Bearer SECRET"

* ``/``      the web page (request, workflow type, agents, tools, approvals)
* ``/v1/...`` a REST API for other programs and platforms (see ``/v1/openapi.json``)
* ``/mcp``   the same workspace for MCP hosts (Streamable HTTP)

Built on the standard library only. It listens on 127.0.0.1 by default and
refuses requests sent by other web sites. Use ``token=`` (or your own login in
front of it) before exposing it to a network.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from .auto import ASK_HUMAN, AutoFlow, AutoPlan, _show
from .errors import TaskPilotError
from .workspace import PATTERNS, Workspace, WorkspaceRun


def _short(value: Any, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Forbidden(Exception):
    pass


class WebApp:
    """Serves a :class:`Workspace` (an :class:`AutoFlow` is wrapped into one).

    Args:
        token: when set, everything except the page itself needs
            ``Authorization: Bearer <token>`` (the page asks for it once).
            ``TASKPILOT_TOKEN`` is used when not given.
        mcp: also serve the workspace at ``/mcp`` for MCP hosts.
        allowed_origins: other web sites (``https://app.example.com``) whose
            pages may call the API from a browser.
    """

    def __init__(self, workspace: Union[Workspace, AutoFlow], title: str = "taskpilot", *,
                 token: Optional[str] = None, mcp: bool = True, allowed_origins: Iterable[str] = ()):
        self.ws = workspace if isinstance(workspace, Workspace) else Workspace.from_auto(workspace)
        self.title = title
        self.token = token if token is not None else (os.environ.get("TASKPILOT_TOKEN") or None)
        self.allowed_origins = set(allowed_origins)
        self.mcp = mcp
        self._mcp_sessions: set = set()

    def mcp_server(self) -> Any:
        """The workspace as an MCP server, as it is now (agents made on the page included)."""
        from .mcp import MCPServer

        return MCPServer.from_workspace(self.ws)

    # ------------------------------------------------------------ views
    @staticmethod
    def plan_view(plan: AutoPlan) -> Dict[str, Any]:
        steps = []
        for st in plan.steps:
            steps.append({"id": st.id, "tool": st.tool, "note": st.note, "after": st.refs(),
                          "args": {k: _show(v) for k, v in st.args.items()},
                          "approval": st.tool in plan.approve, "question": st.tool == ASK_HUMAN})
        return {"request": plan.request, "answer": plan.answer, "steps": steps, "text": plan.describe()}

    def view(self, run: WorkspaceRun) -> Dict[str, Any]:
        out = run.to_dict()
        if run.plan is not None:
            out["plan"] = self.plan_view(run.plan)
        r = run.result
        if r is not None:
            out["steps"] = [{"id": n, "status": r.steps[n].status, "ms": round(r.steps[n].duration * 1000, 1),
                             "value": _short(r.steps[n].value) if r.steps[n].status == "success" else "",
                             "note": r.steps[n].reason or (f"{type(r.steps[n].error).__name__}: {r.steps[n].error}" if r.steps[n].error else "")}
                            for n in r.order]
            out["pending"] = [{"key": p.key, "question": p.question or p.key,
                               "kind": "approval" if p.key.startswith("approve_") else "question"} for p in r.pending]
            out["ms"] = round(r.duration * 1000, 1)
        else:
            out.setdefault("steps", [])
            out["pending"] = []
        if out.get("output") is not None:
            out["output"] = _short(out["output"], 4000)
        return out

    # ------------------------------------------------------------ api
    def api(self, method: str, path: str, body: Dict[str, Any]) -> Any:
        ws = self.ws
        if method == "GET":
            if path == "/api/workspace":
                return ws.overview()
            if path == "/api/tools":
                return {"tools": ws.overview()["tools"]}
            if path == "/api/history":
                return {"runs": [{"id": r.id, "request": r.request[:120], "pattern": r.pattern, "agents": r.agents,
                                  "status": r.status, "started": r.started} for r in reversed(list(ws.history.values()))]}
            raise LookupError(path)
        s = lambda key: str(body.get(key, ""))  # noqa: E731
        if path == "/api/suggest":
            return ws.suggest(s("request"))
        if path == "/api/plan":
            run = ws.plan(s("request"))
            return {"id": run.id, **self.plan_view(run.plan)}  # type: ignore[arg-type]
        if path == "/api/run":
            return self.view(ws.go(s("id")))
        if path == "/api/answer":
            answers = body.get("answers") or {}
            if not isinstance(answers, dict):
                raise ValueError("answers must be an object")
            return self.view(ws.answer(s("id"), answers))
        if path == "/api/start":
            agents = body.get("agents") or []
            items = body.get("items") or []
            if not isinstance(agents, list) or not isinstance(items, list):
                raise ValueError("agents and items must be lists")
            rounds = body.get("rounds")
            return self.view(ws.start(s("request"), s("pattern"), [str(a) for a in agents], items=[str(i) for i in items],
                                      rounds=int(rounds) if rounds else None))
        if path == "/api/get":
            return self.view(ws.get(s("id")))
        if path == "/api/action":
            return self.view(ws.approve_action(s("id"), s("action"), bool(body.get("approve"))))
        if path == "/api/agents":
            tools = body.get("tools") or []
            if not isinstance(tools, list):
                raise ValueError("tools must be a list")
            if body.get("update"):
                ws.update_agent(s("name"), s("instructions"), [str(t) for t in tools], s("description"))
            else:
                ws.create_agent(s("name"), s("instructions"), [str(t) for t in tools], s("description"))
            return ws.overview()
        if path == "/api/agents/delete":
            ws.delete_agent(s("name"))
            return ws.overview()
        if path == "/api/workflows":
            ws.save_workflow(s("name"), s("id"))
            return ws.overview()
        if path == "/api/workflows/run":
            return self.view(ws.run_workflow(s("name")))
        if path == "/api/workflows/delete":
            ws.delete_workflow(s("name"))
            return ws.overview()
        raise LookupError(path)

    # ------------------------------------------------------------ REST
    def rest(self, method: str, path: str, body: Any) -> Tuple[int, Any]:
        """The ``/v1`` API: ``(status, json)``. Errors come back as ``{"error": ...}``."""
        ws = self.ws
        parts = [urllib.parse.unquote(p) for p in path.strip("/").split("/")[1:]]
        b: Dict[str, Any] = body if isinstance(body, dict) else {}
        s = lambda key: str(b.get(key) or "")  # noqa: E731
        route = (method, parts[0] if parts else "", len(parts))
        if route == ("GET", "health", 1):
            from . import __version__

            return 200, {"ok": True, "version": __version__, "name": ws.name}
        if route == ("GET", "openapi.json", 1):
            return 200, openapi_doc(self)
        if method == "GET" and len(parts) == 1 and parts[0] in ("tools", "agents", "patterns", "workflows"):
            return 200, {parts[0]: ws.overview()[parts[0]]}
        if route == ("POST", "tools", 2):
            tool = ws.tools.get(parts[1])
            if tool is None:
                raise LookupError(parts[1])
            if parts[1] in ws.approve:
                raise Forbidden(f"'{parts[1]}' needs a person's approval; start a run instead (POST /v1/runs)")
            try:
                return 200, {"output": tool.fn(**tool.prepare(b))}
            except ValueError as exc:
                return 400, {"error": str(exc)}
            except Exception as exc:
                return 422, {"error": f"{type(exc).__name__}: {exc}"}
        if route == ("POST", "suggest", 1):
            return 200, ws.suggest(s("request"))
        if route == ("GET", "runs", 1):
            return 200, {"runs": [r.to_dict() for r in reversed(list(ws.history.values()))]}
        if route == ("POST", "runs", 1):
            request, pattern = s("request"), s("pattern")
            agents = [str(a) for a in b.get("agents") or []]
            why = None
            if pattern in ("", "auto"):
                tip = ws.suggest(request)
                pattern, agents, why = tip["pattern"], tip["agents"], tip.get("why")
                b.setdefault("items", tip.get("items") or [])
            if pattern == "plan" and b.get("plan_only"):
                run = ws.plan(request)
            else:
                rounds = b.get("rounds")
                run = ws.start(request, pattern, agents, items=[str(i) for i in b.get("items") or []],
                               rounds=int(rounds) if rounds else None)
            out = self.view(run)
            if why:
                out["why"] = why
            return 201, out
        if parts[:1] == ["runs"] and len(parts) >= 2:
            rid = parts[1]
            if method == "GET" and len(parts) == 2:
                return 200, self.view(ws.get(rid))
            if method == "POST" and parts[2:] == ["go"]:
                return 200, self.view(ws.go(rid))
            if method == "POST" and parts[2:] == ["answer"]:
                answers = b.get("answers") if "answers" in b else b
                if not isinstance(answers, dict):
                    raise ValueError("answers must be an object")
                return 200, self.view(ws.answer(rid, answers))
            if method == "POST" and len(parts) == 4 and parts[2] == "actions":
                if not isinstance(b.get("approve"), bool):
                    raise ValueError('send {"approve": true} or {"approve": false}')
                return 200, self.view(ws.approve_action(rid, parts[3], b["approve"]))
        if method == "POST" and len(parts) == 3 and parts[0] == "workflows" and parts[2] == "run":
            return 200, self.view(ws.run_workflow(parts[1]))
        raise LookupError(path)

    # ------------------------------------------------------------ http
    def handler(self) -> type:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: Any, ctype: str = "application/json") -> None:
                data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", f"{ctype}; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _raw(self, status: int, headers: Dict[str, str], data: bytes) -> None:
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method: str) -> None:
                from .mcp import http_guard

                path = urllib.parse.urlsplit(self.path).path
                try:
                    if method == "GET" and path in ("/", "/index.html"):
                        self._send(200, PAGE.replace("{{TITLE}}", _html(app.title)), "text/html")
                        return
                    hdrs = {k.lower(): v for k, v in self.headers.items()}
                    problem = http_guard(hdrs, app.token, app.allowed_origins)
                    if problem:
                        self._raw(*problem)
                        return
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > 4_000_000:
                        raise ValueError("request too large")
                    raw = self.rfile.read(length) if length else b""
                    if path.rstrip("/") == "/mcp" and app.mcp:
                        self._raw(*app.mcp_server().handle_http(method, hdrs, raw, app._mcp_sessions))
                        return
                    body: Any = json.loads(raw or b"{}") if method == "POST" else {}
                    if path.startswith("/v1/"):
                        status, out = app.rest(method, path, body)
                        self._send(status, out)
                        return
                    if not isinstance(body, dict):
                        raise ValueError("send a JSON object")
                    self._send(200, app.api(method, path, body))
                except LookupError:
                    self._send(404, {"error": "not found"})
                except Forbidden as exc:
                    self._send(403, {"error": str(exc)})
                except (TaskPilotError, ValueError, TypeError) as exc:
                    self._send(400, {"error": str(exc)})
                except Exception as exc:  # never crash the server
                    self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

            def do_DELETE(self) -> None:
                self._handle("DELETE")

        return Handler

    def server(self, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
        return ThreadingHTTPServer((host, port), self.handler())


def openapi_doc(app: "WebApp") -> Dict[str, Any]:
    """An OpenAPI 3 description of ``/v1``, so other platforms can import it (even with ``openapi_tools``)."""
    from . import __version__

    obj = {"type": "object"}
    run_id = {"name": "run_id", "in": "path", "required": True, "schema": {"type": "string"}}

    def op(op_id: str, summary: str, body: Optional[Dict[str, Any]] = None, params: Any = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"operationId": op_id, "summary": summary, "responses": {"200": {"description": "OK"}}}
        if body is not None:
            out["requestBody"] = {"required": True, "content": {"application/json": {"schema": body}}}
        if params:
            out["parameters"] = params
        return out

    run_body = {"type": "object", "required": ["request"], "properties": {
        "request": {"type": "string", "description": "what should be done, in plain words"},
        "pattern": {"type": "string", "description": "workflow type (empty or 'auto' lets the AI choose)",
                    "enum": ["", "auto"] + list(PATTERNS)},
        "agents": {"type": "array", "items": {"type": "string"}},
        "items": {"type": "array", "items": {"type": "string"}},
        "plan_only": {"type": "boolean", "description": "for 'plan': only make the plan; run it with /go"}}}
    paths: Dict[str, Any] = {
        "/v1/health": {"get": op("health", "Is the server up")},
        "/v1/tools": {"get": op("list_tools", "Tools of this workspace")},
        "/v1/agents": {"get": op("list_agents", "Agents of this workspace")},
        "/v1/patterns": {"get": op("list_patterns", "Workflow types")},
        "/v1/workflows": {"get": op("list_workflows", "Saved workflows")},
        "/v1/suggest": {"post": op("suggest", "Let the AI pick a workflow type and agents",
                                   {"type": "object", "required": ["request"], "properties": {"request": {"type": "string"}}})},
        "/v1/runs": {"get": op("list_runs", "Recent runs"), "post": op("start_run", "Do a request", run_body)},
        "/v1/runs/{run_id}": {"get": op("get_run", "A run's status, output and waiting approvals", params=[run_id])},
        "/v1/runs/{run_id}/go": {"post": op("go_run", "Run a planned step plan", params=[run_id])},
        "/v1/runs/{run_id}/answer": {"post": op("answer_run", "Answer the questions or approvals a plan waits on",
                                                {"type": "object", "properties": {"answers": obj}}, [run_id])},
        "/v1/runs/{run_id}/actions/{action_id}": {"post": op(
            "decide_action", "Approve or reject an action an agent is holding back",
            {"type": "object", "required": ["approve"], "properties": {"approve": {"type": "boolean"}}},
            [run_id, {"name": "action_id", "in": "path", "required": True, "schema": {"type": "string"}}])},
        "/v1/workflows/{name}/run": {"post": op("run_workflow", "Run a saved workflow",
                                                params=[{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}])},
    }
    for name, t in app.ws.tools.items():
        if name in app.ws.approve:
            continue
        paths[f"/v1/tools/{name}"] = {"post": op(f"tool_{name}", t.description[:300], t.parameters)}
    doc: Dict[str, Any] = {"openapi": "3.0.3", "info": {"title": f"{app.title} API", "version": __version__},
                           "servers": [{"url": "/"}], "paths": paths}
    if app.token:
        doc["components"] = {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}}}
        doc["security"] = [{"bearer": []}]
    return doc


def _html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def serve(workspace: Union[Workspace, AutoFlow], host: str = "127.0.0.1", port: int = 8000, title: str = "taskpilot",
          *, token: Optional[str] = None, mcp: bool = True, allowed_origins: Iterable[str] = ()) -> None:
    """Start the web page, the REST API and the MCP endpoint; block until Ctrl+C."""
    app = WebApp(workspace, title, token=token, mcp=mcp, allowed_origins=allowed_origins)
    httpd = app.server(host, port)
    base = f"http://{host}:{httpd.server_address[1]}"
    print(f"taskpilot is ready: open {base}  (Ctrl+C to stop)")
    print(f"  REST API: {base}/v1  (spec: {base}/v1/openapi.json)" + ("" if not mcp else f"   MCP: {base}/mcp"))
    if app.token:
        print("  every API call needs: Authorization: Bearer <token>")
    elif host not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: no token set and the server is open to the network; use --token")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1f2328;--dim:#656d76;--line:#d0d7de;--accent:#0969da;--ok:#1a7f37;--wait:#8250df;--bad:#cf222e;--warn:#9a6700;--chip:#eef1f4}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--text:#e6edf3;--dim:#8b949e;--line:#30363d;--accent:#2f81f7;--ok:#3fb950;--wait:#bc8cff;--bad:#f85149;--warn:#d29922;--chip:#21262d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:22px 16px 60px}h1{font-size:24px;margin:0}.sub{color:var(--dim);margin:2px 0 14px}
nav{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px}nav button{margin:0}nav button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0}
textarea,input[type=text],select{width:100%;border:1px solid var(--line);border-radius:10px;padding:10px 12px;font:inherit;background:var(--bg);color:var(--text)}
textarea{min-height:84px;resize:vertical}label.lab{display:block;font-weight:600;margin:14px 0 6px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:9px;padding:8px 15px;cursor:pointer;margin:8px 8px 0 0}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.good{background:var(--ok);border-color:var(--ok);color:#fff}
button.small{padding:3px 10px;font-size:13px;margin:4px 6px 0 0}button:disabled{opacity:.5;cursor:wait}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{border:1px solid var(--line);background:var(--chip);border-radius:999px;padding:5px 12px;cursor:pointer;font-size:14px;user-select:none}
.chip.on{background:var(--accent);border-color:var(--accent);color:#fff}.chip .n{font-weight:700;margin-right:4px}
.dim{color:var(--dim);font-size:14px}.step{border-left:3px solid var(--line);padding:5px 0 5px 12px;margin:7px 0}
.t{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px}.badge{display:inline-block;font-size:12px;border-radius:999px;padding:1px 9px;margin-left:6px;border:1px solid currentColor}
.success,.done{border-color:var(--ok)}.paused,.waiting{border-color:var(--wait)}.failed,.rejected{border-color:var(--bad)}.skipped{border-color:var(--dim)}
.b-ok{color:var(--ok)}.b-wait{color:var(--wait)}.b-bad{color:var(--bad)}.b-warn{color:var(--warn)}.b-dim{color:var(--dim)}
.answer{font-size:17px;white-space:pre-wrap;margin:8px 0}.err{color:var(--bad);white-space:pre-wrap}.pre{white-space:pre-wrap}
.tip{border:1px dashed var(--accent);border-radius:10px;padding:10px 14px;margin-top:12px}.hidden{display:none}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:7px 6px;text-align:left;vertical-align:top}
.row{display:flex;gap:8px;align-items:center}.row input{flex:1}
</style></head><body><main>
<h1>{{TITLE}}</h1><p class="sub">Say what you need. Pick how it should be done, or let the AI suggest. Risky actions always ask you first.</p>
<nav><button data-tab="run" class="on">Run</button><button data-tab="agents">Agents</button><button data-tab="tools">Tools</button><button data-tab="saved">Saved</button><button data-tab="history">History</button></nav>

<section id="tab-run">
 <div class="card">
  <label class="lab" for="req">What should be done?</label>
  <textarea id="req" placeholder="e.g. Write a thank-you email to ravi@example.com for his order and send it"></textarea>
  <label class="lab" for="pattern">How should it be done?</label>
  <select id="pattern"></select><div class="dim" id="patternAbout"></div>
  <div id="agentBox"><label class="lab">Agents <span class="dim" id="agentHint"></span></label><div class="chips" id="agentChips"></div></div>
  <div id="itemsBox" class="hidden"><label class="lab" for="items">Items (one per line)</label><textarea id="items"></textarea></div>
  <button class="primary" id="goBtn">Run</button><span class="dim" id="goNote"></span>
  <div id="tip" class="tip hidden"></div>
 </div>
 <div class="card hidden" id="planCard"><b>Plan</b> <span class="dim" id="planNote"></span><div id="plan"></div>
  <button class="good" id="runPlanBtn">Run this plan</button><button id="cancelBtn">Change request</button></div>
 <div class="card hidden" id="resCard"><b>Result</b> <span id="status"></span>
  <div id="pending"></div><div class="answer" id="answer"></div><div id="trace"></div>
  <div class="row" style="margin-top:10px"><input type="text" id="saveName" placeholder="Name this workflow to reuse it (e.g. Monthly net sales)"><button id="saveBtn" style="margin-top:0">Save</button></div></div>
</section>

<section id="tab-agents" class="hidden">
 <div class="card"><b>Agents</b> <span class="dim">AI helpers. Each has instructions and the tools it may use.</span><div id="agentList"></div></div>
 <div class="card"><b id="agentFormTitle">New agent</b>
  <label class="lab" for="aName">Name</label><input type="text" id="aName" placeholder="e.g. writer">
  <label class="lab" for="aInstr">Instructions (how it should behave)</label><textarea id="aInstr" placeholder="e.g. You write short, friendly customer emails in simple English."></textarea>
  <label class="lab">Tools it may use</label><div class="chips" id="aTools"></div>
  <button class="primary" id="aSave">Save agent</button><button id="aReset">Clear</button></div>
</section>

<section id="tab-tools" class="hidden"><div class="card"><b>Tools</b> <span class="dim">What the system can do. Added by your developer.</span>
 <table id="toolTable"></table></div></section>
<section id="tab-saved" class="hidden"><div class="card"><b>Saved workflows</b> <span class="dim">Run again with one click.</span><div id="savedList"></div></div></section>
<section id="tab-history" class="hidden"><div class="card"><b>Recent runs</b><div id="historyList"></div></div></section>
<div class="err" id="err"></div>
</main><script>
let W={tools:[],agents:[],patterns:[],workflows:[]},picked=[],currentRun=null,editing=null;
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let TOKEN="";try{TOKEN=sessionStorage.getItem("tp_token")||""}catch(e){}
async function api(path,body,again){const h={"Content-Type":"application/json"};if(TOKEN)h.Authorization="Bearer "+TOKEN;
 const r=await fetch(path,body===undefined?{headers:h}:{method:"POST",headers:h,body:JSON.stringify(body)});
 if(r.status===401&&!again){const t=prompt("Access token");if(t){TOKEN=t.trim();try{sessionStorage.setItem("tp_token",TOKEN)}catch(e){}return api(path,body,true)}}
 const d=await r.json();if(!r.ok)throw new Error(d.error||r.statusText);return d}
function err(e){$("err").textContent=e?e.message||e:""}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on",x===b));
 document.querySelectorAll("section").forEach(s=>s.classList.toggle("hidden",s.id!=="tab-"+b.dataset.tab));if(b.dataset.tab==="history")loadHistory()});
async function load(){W=await api("/api/workspace");drawAll()}
function pat(k){return W.patterns.find(p=>p.key===k)}
function drawAll(){
 const sel=$("pattern"),keep=sel.value||"auto";
 sel.innerHTML='<option value="auto">AI decides (recommended)</option>'+W.patterns.map(p=>`<option value="${p.key}">${esc(p.label)}</option>`).join("");
 sel.value=keep;onPattern();drawChips();
 $("agentList").innerHTML=W.agents.length?W.agents.map(a=>`<div class="step"><b>${esc(a.name)}</b><span class="badge b-dim">${a.source==="code"?"from code":"made here"}</span>
  <div class="dim">${esc(a.instructions)}</div><div class="dim">tools: ${a.tools.length?a.tools.map(esc).join(", "):"none"}</div>
  ${a.source==="web"?`<button class="small" data-edit="${esc(a.name)}">Edit</button><button class="small" data-del="${esc(a.name)}">Delete</button>`:""}</div>`).join(""):'<div class="dim">No agents yet. Make one below.</div>';
 document.querySelectorAll("[data-del]").forEach(b=>b.onclick=async()=>{if(!confirm("Delete agent "+b.dataset.del+"?"))return;try{W=await api("/api/agents/delete",{name:b.dataset.del});drawAll()}catch(e){err(e)}});
 document.querySelectorAll("[data-edit]").forEach(b=>b.onclick=()=>{const a=W.agents.find(x=>x.name===b.dataset.edit);editing=a.name;$("aName").value=a.name;$("aName").disabled=true;
  $("aInstr").value=a.instructions;$("agentFormTitle").textContent="Edit agent "+a.name;drawToolChips(a.tools)});
 drawToolChips(editing?(W.agents.find(x=>x.name===editing)||{tools:[]}).tools:[]);
 $("toolTable").innerHTML="<tr><th>Tool</th><th>What it does</th><th>Needs</th></tr>"+W.tools.map(t=>`<tr><td class="t">${esc(t.name)}${t.approval?'<span class="badge b-warn">asks first</span>':""}</td>
  <td>${esc(t.description)}</td><td class="dim">${Object.entries(t.args).map(([k,v])=>esc(k)+(t.required.includes(k)?"":"?")+": "+esc(v)).join(", ")}</td></tr>`).join("");
 $("savedList").innerHTML=W.workflows.length?W.workflows.map(w=>`<div class="step"><b>${esc(w.name)}</b> <span class="badge b-dim">${esc((pat(w.pattern)||{label:w.pattern}).label)}</span>${w.saved_plan?'<span class="badge b-ok">no AI needed</span>':""}
  <div class="dim">${esc(w.request)}${w.agents.length?" | agents: "+w.agents.map(esc).join(" -> "):""}</div>
  <button class="small good" data-run="${esc(w.name)}">Run</button><button class="small" data-wdel="${esc(w.name)}">Delete</button></div>`).join(""):'<div class="dim">Nothing saved yet. Run something, then use "Save".</div>';
 document.querySelectorAll("[data-run]").forEach(b=>b.onclick=async()=>{err();b.disabled=true;try{tab("run");show(await api("/api/workflows/run",{name:b.dataset.run}))}catch(e){err(e)}finally{b.disabled=false}});
 document.querySelectorAll("[data-wdel]").forEach(b=>b.onclick=async()=>{try{W=await api("/api/workflows/delete",{name:b.dataset.wdel});drawAll()}catch(e){err(e)}});
}
function tab(name){document.querySelector(`nav button[data-tab="${name}"]`).click()}
function drawChips(){$("agentChips").innerHTML=W.agents.map(a=>{const i=picked.indexOf(a.name);return `<span class="chip ${i>=0?"on":""}" data-a="${esc(a.name)}">${i>=0?`<span class="n">${i+1}</span>`:""}${esc(a.name)}</span>`}).join("")||'<span class="dim">No agents yet (see the Agents tab).</span>';
 document.querySelectorAll("#agentChips .chip").forEach(c=>c.onclick=()=>{const n=c.dataset.a,i=picked.indexOf(n);if(i>=0)picked.splice(i,1);else picked.push(n);drawChips();onPattern(true)})}
function drawToolChips(sel){$("aTools").innerHTML=W.tools.map(t=>`<span class="chip ${sel.includes(t.name)?"on":""}" data-t="${esc(t.name)}">${esc(t.name)}${t.approval?" (asks first)":""}</span>`).join("");
 document.querySelectorAll("#aTools .chip").forEach(c=>c.onclick=()=>c.classList.toggle("on"))}
function onPattern(keepTip){const k=$("pattern").value,p=pat(k);
 $("patternAbout").textContent=k==="auto"?"The AI looks at your request, the agents and the tools, and suggests the best way. You confirm before it runs.":p.about;
 $("agentBox").classList.toggle("hidden",k==="auto"||k==="plan");$("itemsBox").classList.toggle("hidden",k!=="map");
 $("agentHint").textContent=p&&k!=="plan"?`(${p.min}${p.max?(p.max===p.min?"":"-"+p.max):"+"} - click in the order they should work; picked: ${picked.length})`:"";
 $("goBtn").textContent=k==="auto"?"Ask AI how to do it":k==="plan"?"Make a plan":"Run";if(!keepTip)$("tip").classList.add("hidden")}
$("pattern").onchange=()=>onPattern();
$("goBtn").onclick=async()=>{err();const k=$("pattern").value,b=$("goBtn"),label=b.textContent;b.disabled=true;b.textContent="Working...";
 try{if(k==="auto"){const s=await api("/api/suggest",{request:$("req").value});const p=pat(s.pattern);
   $("tip").innerHTML=`<b>AI suggests:</b> ${esc(p.label)}${s.agents.length?" with "+s.agents.map(esc).join(" -> "):""}<div class="dim">${esc(s.why)}</div>
   <button class="good" id="useTip">Use this</button><button id="tipRun" class="primary">Use and run</button>`;$("tip").classList.remove("hidden");
   const use=()=>{$("pattern").value=s.pattern;picked=[...s.agents];if(s.items&&s.items.length)$("items").value=s.items.join("\n");drawChips();onPattern(true)};
   $("useTip").onclick=use;$("tipRun").onclick=()=>{use();$("goBtn").click()}}
  else if(k==="plan"){const p=await api("/api/plan",{request:$("req").value});currentRun=p.id;drawPlan(p)}
  else{$("planCard").classList.add("hidden");show(await api("/api/start",{request:$("req").value,pattern:k,agents:picked,items:$("items").value.split("\n").filter(x=>x.trim())}))}}
 catch(e){err(e)}finally{b.disabled=false;b.textContent=label}};
function drawPlan(p){$("plan").innerHTML=p.steps.map((s,i)=>`<div class="step"><div class="t">${i+1}. ${esc(s.tool)}(${Object.entries(s.args).map(([k,v])=>esc(k)+"="+esc(v)).join(", ")})
 ${s.approval?'<span class="badge b-warn">asks you first</span>':""}${s.question?'<span class="badge b-wait">asks you</span>':""}
 ${s.after.length?`<span class="badge b-dim">after ${esc(s.after.join(", "))}</span>`:'<span class="badge b-dim">starts right away</span>'}</div>
 <div class="dim">${esc(s.id)}${s.note?" - "+esc(s.note):""}</div></div>`).join("");
 $("planNote").textContent=`${p.steps.length} steps`;$("planCard").classList.remove("hidden");$("runPlanBtn").disabled=false;$("resCard").classList.add("hidden")}
$("runPlanBtn").onclick=async()=>{err();const b=$("runPlanBtn");b.disabled=true;try{show(await api("/api/run",{id:currentRun}))}catch(e){err(e);b.disabled=false}};
$("cancelBtn").onclick=()=>{$("planCard").classList.add("hidden");$("req").focus()};
function show(st){currentRun=st.id;$("resCard").classList.remove("hidden");
 const cls={ok:"b-ok",paused:"b-wait",waiting:"b-wait",failed:"b-bad"}[st.status]||"b-dim";
 const label={ok:"done",paused:"waiting for you",waiting:"waiting for your approval",failed:"failed"}[st.status]||st.status;
 $("status").innerHTML=`<span class="badge ${cls}">${esc(label)}</span> <span class="dim">${esc((pat(st.pattern)||{label:st.pattern}).label)}${st.agents.length?" · "+st.agents.map(esc).join(" -> "):""}</span>`;
 $("answer").textContent=st.error?"":(st.output??"");if(st.error)err(st.error);
 const steps=(st.steps||[]).map(s=>`<div class="step ${esc(s.status)}"><span class="t">${esc(s.id)}</span> <span class="dim">${esc(s.status)} · ${s.ms} ms</span><div class="dim pre">${esc(s.value||s.note)}</div></div>`);
 const trace=st.steps&&st.steps.length?[]:(st.trace||[]).map(t=>`<div class="step"><b>${esc(t.who)}</b><div class="dim pre">${esc(t.text)}</div></div>`);
 $("trace").innerHTML=[...steps,...trace].join("");
 const acts=(st.actions||[]).map(a=>`<div class="card"><b>${a.status==="waiting"?"Approval needed":"Action "+esc(a.status)}</b> <span class="t">${esc(a.tool)}</span>
  <div class="dim pre">${esc(JSON.stringify(a.args,null,1))}</div>${a.result?`<div class="pre">${esc(a.result)}</div>`:""}
  ${a.status==="waiting"?`<button class="good" data-act="${esc(a.id)}" data-yes="1">Approve</button><button data-act="${esc(a.id)}">Reject</button>`:""}</div>`);
 const qs=(st.pending||[]).map(p=>p.kind==="approval"
  ?`<div class="card"><b>Approval needed</b><div>${esc(p.question)}</div><button class="good" data-k="${esc(p.key)}" data-v="yes">Approve</button><button data-k="${esc(p.key)}" data-v="no">Reject</button></div>`
  :`<div class="card"><b>Question</b><div>${esc(p.question)}</div><input type="text" data-q="${esc(p.key)}" placeholder="Your answer"><button class="good" data-send="${esc(p.key)}">Send</button></div>`);
 $("pending").innerHTML=[...qs,...acts].join("");
 document.querySelectorAll("#pending [data-act]").forEach(b=>b.onclick=async()=>{try{show(await api("/api/action",{id:st.id,action:b.dataset.act,approve:!!b.dataset.yes}))}catch(e){err(e)}});
 document.querySelectorAll("#pending [data-v]").forEach(b=>b.onclick=()=>answer({[b.dataset.k]:b.dataset.v}));
 document.querySelectorAll("#pending [data-send]").forEach(b=>b.onclick=()=>{const i=document.querySelector(`#pending input[data-q="${b.dataset.send}"]`);answer({[b.dataset.send]:i.value})})}
async function answer(a){err();try{show(await api("/api/answer",{id:currentRun,answers:a}))}catch(e){err(e)}}
$("saveBtn").onclick=async()=>{err();try{W=await api("/api/workflows",{name:$("saveName").value,id:currentRun});$("saveName").value="";drawAll();$("saveBtn").textContent="Saved";setTimeout(()=>$("saveBtn").textContent="Save",1500)}catch(e){err(e)}};
$("aSave").onclick=async()=>{err();const tools=[...document.querySelectorAll("#aTools .chip.on")].map(c=>c.dataset.t);
 try{W=await api("/api/agents",{name:$("aName").value,instructions:$("aInstr").value,tools,update:!!editing});reset();drawAll()}catch(e){err(e)}};
function reset(){editing=null;$("aName").value="";$("aName").disabled=false;$("aInstr").value="";$("agentFormTitle").textContent="New agent";drawToolChips([])}
$("aReset").onclick=reset;
async function loadHistory(){const h=await api("/api/history");$("historyList").innerHTML=h.runs.length?h.runs.map(r=>`<div class="step"><span class="badge ${r.status==="ok"?"b-ok":r.status==="failed"?"b-bad":"b-wait"}">${esc(r.status)}</span>
 <span class="dim">${esc((pat(r.pattern)||{label:r.pattern}).label)} · ${new Date(r.started*1000).toLocaleTimeString()}</span><div>${esc(r.request)}</div><button class="small" data-open="${esc(r.id)}">Open</button></div>`).join(""):'<div class="dim">No runs yet.</div>';
 document.querySelectorAll("[data-open]").forEach(b=>b.onclick=async()=>{try{const st=await api("/api/get",{id:b.dataset.open});tab("run");show(st)}catch(e){err(e)}})}
load().catch(err);
</script></body></html>"""
