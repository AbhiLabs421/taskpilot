"""Connecting to other platforms: MCP (client + server), REST/OpenAPI tools, A2A agents, the REST API."""

import json
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from taskpilot import Agent, FakeLLM, Flow, Supervisor, ToolError, Workspace, tool
from taskpilot.connect import A2AAgent, http_tool, openapi_tools
from taskpilot.errors import TaskPilotError
from taskpilot.mcp import MCPClient, MCPError, MCPServer, _safe_name, _tool_value
from taskpilot.web import WebApp

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def lookup(order_id: str) -> dict:
    """Look up an order."""
    if order_id != "A7":
        raise KeyError(f"no order {order_id}")
    return {"id": "A7", "customer": "Ravi"}


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


def running(httpd):
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}"


def rpc(server, method, params=None, rid=1):
    msg = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        msg["params"] = params
    return server.handle(msg)


# ============================================================ MCP server (protocol only)
class TestMCPServerProtocol(unittest.TestCase):
    def setUp(self):
        self.server = MCPServer([add, lookup], name="office", instructions="Office tools")

    def test_easy_handshake_and_list(self):
        r = rpc(self.server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(r["result"]["serverInfo"]["name"], "office")
        self.assertEqual(r["result"]["instructions"], "Office tools")
        future = rpc(self.server, "initialize", {"protocolVersion": "2099-01-01"})
        self.assertEqual(future["result"]["protocolVersion"], "2025-11-25")
        tools = rpc(self.server, "tools/list")["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["add", "lookup"])
        self.assertEqual(tools[0]["inputSchema"]["properties"]["a"], {"type": "integer"})
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(rpc(self.server, "ping")["result"], {})

    def test_medium_calls_and_errors(self):
        ok = rpc(self.server, "tools/call", {"name": "add", "arguments": {"a": "2", "b": 3}})["result"]
        self.assertEqual((ok["content"][0]["text"], ok["isError"]), ("5", False))
        data = rpc(self.server, "tools/call", {"name": "lookup", "arguments": {"order_id": "A7"}})["result"]
        self.assertEqual(data["structuredContent"], {"id": "A7", "customer": "Ravi"})
        bad = rpc(self.server, "tools/call", {"name": "lookup", "arguments": {"order_id": "Z9"}})["result"]
        self.assertTrue(bad["isError"])
        self.assertIn("no order Z9", bad["content"][0]["text"])
        missing = rpc(self.server, "tools/call", {"name": "add", "arguments": {"a": 1}})["result"]
        self.assertTrue(missing["isError"])
        self.assertEqual(rpc(self.server, "tools/call", {"name": "nope"})["error"]["code"], -32602)
        self.assertEqual(rpc(self.server, "server/discover")["error"]["code"], -32601)
        self.assertEqual(self.server.handle({"id": 3})["error"]["code"], -32600)
        batch = self.server.handle([{"jsonrpc": "2.0", "id": 1, "method": "ping"},
                                    {"jsonrpc": "2.0", "method": "notifications/cancelled"}])
        self.assertEqual(len(batch), 1)

    def test_high_flows_agents_and_names(self):
        flow = Flow("net_sales")

        @flow.step
        def gross(a: float, b: float):
            return a + b

        @flow.step
        def net(gross, returns: float):
            return gross - returns

        helper = Agent(FakeLLM(["agent answer"]), name="helper", system="Helps.")
        server = MCPServer([flow, helper])
        server.add(add, name="plus", description="Adds")
        names = [t["name"] for t in rpc(server, "tools/list")["result"]["tools"]]
        self.assertEqual(names, ["net_sales", "helper", "plus"])
        out = rpc(server, "tools/call", {"name": "net_sales", "arguments": {"a": 125000, "b": 75000, "returns": 40000}})
        self.assertEqual(out["result"]["content"][0]["text"], "160000")
        self.assertEqual(rpc(server, "tools/call", {"name": "helper", "arguments": {"task": "hi"}})["result"]["content"][0]["text"],
                         "agent answer")
        with self.assertRaises(ToolError):
            server.add(lambda x: x, name="plus")
        self.assertEqual(_safe_name("crm.find customer", ()), "crm_find_customer")
        self.assertEqual(_safe_name("9lives", ()), "t_9lives")
        self.assertEqual(_safe_name("a", ["a"]), "a_2")
        self.assertEqual(len(_safe_name("x" * 200, ())), 64)

    def test_values_from_results(self):
        self.assertEqual(_tool_value({"structuredContent": {"result": 5}}), 5)
        self.assertEqual(_tool_value({"structuredContent": {"a": 1}}), {"a": 1})
        self.assertEqual(_tool_value({"content": [{"type": "text", "text": "[1, 2]"}]}), [1, 2])
        self.assertEqual(_tool_value({"content": [{"type": "text", "text": "true"}]}), "true")
        self.assertEqual(_tool_value({"content": [{"type": "text", "text": "a"}, {"type": "image", "mimeType": "image/png"}]}),
                         "a\n[image image/png]")


# ============================================================ MCP client <-> server over stdio
SERVER_SCRIPT = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {src!r})
    from taskpilot import tool
    from taskpilot.mcp import MCPServer

    @tool
    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        print("a stray print goes to stderr, not into the protocol")
        return a + b

    def slow(seconds: float) -> str:
        '''Sleep.'''
        time.sleep(seconds)
        return "awake"

    def boom() -> str:
        '''Always fails.'''
        raise RuntimeError("database is down")

    def crash() -> str:
        '''Kills the server.'''
        sys.stderr.write("fatal: out of memory\\n"); sys.stderr.flush()
        import os; os._exit(3)

    MCPServer([add, slow, boom, crash], name="script", instructions="Test server").serve_stdio()
""")


class TestMCPStdio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.script = os.path.join(cls.dir, "server.py")
        with open(cls.script, "w") as fh:
            fh.write(SERVER_SCRIPT.format(src=os.path.abspath(SRC)))

    def client(self, **kw):
        return MCPClient.stdio([sys.executable, self.script], **kw)

    def test_easy_tools_from_a_server(self):
        with self.client() as c:
            self.assertEqual(c.server_info["name"], "script")
            self.assertEqual(c.instructions, "Test server")
            self.assertEqual(c.protocol_version, "2025-11-25")
            tools = c.tools()
            self.assertEqual([t.name for t in tools], ["add", "slow", "boom", "crash"])
            self.assertEqual(tools[0].invoke({"a": "40", "b": 2}), "42")
            self.assertEqual(c.call("add", {"a": 1, "b": 2}), 3)
            self.assertTrue(c.ping())

    def test_medium_errors_come_back_readable(self):
        with self.client() as c:
            with self.assertRaises(ToolError) as cm:
                c.call("boom")
            self.assertIn("database is down", str(cm.exception))
            boom = c.tools(include=["boom"])[0]
            self.assertIn("database is down", boom.invoke({}))  # the model sees the error and can recover
            with self.assertRaises(MCPError):
                c.tools(include=["ghost"])
            self.assertEqual([t.name for t in c.tools(prefix="s_", exclude=["crash"])], ["s_add", "s_slow", "s_boom"])

    def test_high_parallel_calls_and_timeout(self):
        with self.client(timeout=5) as c:
            results, threads = [], []
            t0 = time.perf_counter()
            for _ in range(4):
                th = threading.Thread(target=lambda: results.append(c.call("slow", {"seconds": 0.4})))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()
            self.assertEqual(results, ["awake"] * 4)
            self.assertLess(time.perf_counter() - t0, 1.5)  # calls run side by side, not one by one
            with self.assertRaises(MCPError) as cm:
                c.call("slow", {"seconds": 3}, timeout=0.3)
            self.assertIn("did not answer", str(cm.exception))
            self.assertEqual(c.call("add", {"a": 2, "b": 2}), 4)  # still usable afterwards

    def test_complex_server_crash_and_agent_use(self):
        c = self.client()
        llm = FakeLLM([{"tool": "add", "args": {"a": 20, "b": 22}}, "The answer is 42."])
        result = Agent(llm, tools=c.tools(include=["add"])).run("What is 20 + 22?")
        self.assertEqual(result.output, "The answer is 42.")
        self.assertEqual(result.steps[0].output, "42")
        with self.assertRaises(MCPError) as cm:
            c.call("crash")
        self.assertIn("out of memory", str(cm.exception))
        c.close()
        with self.assertRaises(MCPError):
            c.call("add", {"a": 1, "b": 1})
        with self.assertRaises(MCPError):
            MCPClient.stdio("definitely-not-a-real-command-xyz").connect()


class TestMCPCommandLine(unittest.TestCase):
    def test_high_taskpilot_mcp_command(self):
        folder = tempfile.mkdtemp()
        tools_file = os.path.join(folder, "tools.py")
        with open(tools_file, "w") as fh:
            fh.write(textwrap.dedent("""
                from taskpilot import FakeLLM, tool
                print("loading tools...")          # must not reach the MCP protocol

                @tool
                def add(a: int, b: int) -> int:
                    '''Add two numbers.'''
                    return a + b

                @tool
                def send_email(to: str, body: str) -> str:
                    '''Send an email.'''
                    return "sent"

                llm = FakeLLM(["ok"])
                tools = [add, send_email]
                approve = ["send_email"]
                agents = [{"name": "helper", "instructions": "Helps.", "tools": ["add"]}]
            """))
        env = {"PYTHONPATH": os.path.abspath(SRC)}
        cmd = [sys.executable, "-m", "taskpilot", "mcp", tools_file, "--store", os.path.join(folder, "ws.json")]
        with MCPClient.stdio(cmd, env=env, cwd=folder) as c:
            self.assertEqual(sorted(t["name"] for t in c.list_tools()), ["add", "agent_helper", "run_request"])
            self.assertEqual(c.call("add", {"a": 20, "b": 22}), 42)


# ============================================================ MCP over HTTP
class TestMCPHttp(unittest.TestCase):
    def setUp(self):
        self.httpd = MCPServer([add, lookup], name="web").http_server("127.0.0.1", 0, token="s3cret")
        self.url = running(self.httpd) + "/mcp"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_medium_token_session_and_recovery(self):
        with self.assertRaises(MCPError) as cm:
            MCPClient.http(self.url).connect()
        self.assertEqual(cm.exception.code, 401)
        c = MCPClient.http(self.url, headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(c.call("add", {"a": 2, "b": 5}), 7)
        self.assertTrue(c._http.session)
        c._http.session = "forgotten-by-the-server"  # e.g. the server restarted
        self.assertEqual(c.call("lookup", {"order_id": "A7"})["customer"], "Ravi")
        c.close()

    def test_high_rejects_other_websites(self):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer s3cret", "Origin": "https://evil.example"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 403)
        with self.assertRaises(MCPError):
            MCPClient.http("ftp://nope")


# ============================================================ MCP from a workspace
class TestMCPWorkspace(unittest.TestCase):
    def test_complex_workspace_over_mcp(self):
        def brain(messages, tools):
            names = [t["function"]["name"] for t in tools or []]
            if "send_email" in names and not any(m.get("role") == "tool" for m in messages):
                return {"tool": "send_email", "args": {"to": "ravi@example.com", "body": "Thanks"}}
            return "done: " + messages[-1]["content"][:30]

        ws = Workspace(FakeLLM(fn=brain), [lookup, send_email], approve=["send_email"], review=False, verify=False)
        ws.create_agent("writer", "Writes and sends emails.", ["lookup", "send_email"])
        server = MCPServer.from_workspace(ws)
        listed = {t["name"]: t for t in rpc(server, "tools/list")["result"]["tools"]}
        self.assertEqual(sorted(listed), ["agent_writer", "lookup", "run_request"])  # send_email needs a person
        drafted = rpc(server, "tools/call", {"name": "agent_writer", "arguments": {"task": "thank Ravi"}})["result"]
        self.assertIn("done", drafted["content"][0]["text"])  # it drafted; it could not send
        run = rpc(server, "tools/call", {"name": "run_request",
                                         "arguments": {"request": "Thank Ravi by email", "pattern": "single", "agents": ["writer"]}})
        info = run["result"]["structuredContent"]
        self.assertEqual(info["status"], "waiting")
        self.assertEqual(info["waiting_for_approval"][0]["tool"], "send_email")
        exposed = MCPServer.from_workspace(ws, approval_tools=True)
        tools = {t["name"]: t for t in rpc(exposed, "tools/list")["result"]["tools"]}
        self.assertEqual(tools["send_email"]["annotations"], {"destructiveHint": True})


# ============================================================ REST tools
class _API(BaseHTTPRequestHandler):
    hits = []

    def log_message(self, *a):
        pass

    def _reply(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        _API.hits.append(("GET", u.path))
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/flaky":
            if len([h for h in _API.hits if h[1] == "/flaky"]) < 2:
                return self._reply(503, {"error": "busy"})
            return self._reply(200, {"ok": True})
        if u.path.startswith("/customers/"):
            cid = urllib.parse.unquote(u.path.split("/")[2])
            if cid == "missing":
                return self._reply(404, {"error": "no such customer"})
            return self._reply(200, {"data": {"id": cid, "name": "Ravi", "tier": q.get("fields", "all"),
                                              "key": self.headers.get("X-API-Key")}})
        if u.path == "/openapi.json":
            return self._reply(200, SPEC)
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        u = urllib.parse.urlsplit(self.path)
        _API.hits.append(("POST", u.path))
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if u.path == "/busy":
            return self._reply(503, {"error": "busy"})
        self._reply(201, {"created": body, "path": u.path})


SPEC = {
    "openapi": "3.0.0",
    "servers": [{"url": "/"}],
    "paths": {
        "/customers/{customer_id}": {
            "parameters": [{"name": "customer_id", "in": "path", "required": True, "schema": {"type": "string"}}],
            "get": {"operationId": "getCustomer", "summary": "Get a customer",
                    "parameters": [{"name": "fields", "in": "query", "schema": {"type": "string"}},
                                   {"name": "X-API-Key", "in": "header", "schema": {"type": "string"}}]},
        },
        "/tickets": {"post": {"operationId": "createTicket", "summary": "Open a support ticket",
                              "requestBody": {"required": True, "content": {"application/json": {
                                  "schema": {"$ref": "#/components/schemas/Ticket"}}}}}},
        "/notes": {"post": {"summary": "Add a note", "requestBody": {"content": {"application/json": {
            "schema": {"type": "array", "items": {"type": "string"}}}}}}},
    },
    "components": {"schemas": {"Ticket": {"type": "object", "required": ["title"], "properties": {
        "title": {"type": "string"}, "priority": {"type": "integer"}, "parent": {"$ref": "#/components/schemas/Ticket"}}}}},
}


class TestRestTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _API)
        cls.base = running(cls.httpd)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        _API.hits.clear()

    def test_easy_http_tool(self):
        find = http_tool(self.base + "/customers/{customer_id}", method="GET", name="find_customer",
                         description="Find a customer", params={"fields": {"type": "string", "default": "all"}},
                         headers={"X-API-Key": "k1"}, pick="data")
        self.assertEqual(find.parameters["required"], ["customer_id"])
        self.assertEqual(find(customer_id="C 1", fields="name"), {"id": "C 1", "name": "Ravi", "tier": "name", "key": "k1"})
        self.assertIn("HTTP 404", find.invoke({"customer_id": "missing"}))
        create = http_tool(self.base + "/tickets", name="create_ticket", params={"title": str, "priority": int})
        self.assertEqual(create.invoke({"title": "Late order", "priority": "2"}),
                         json.dumps({"created": {"title": "Late order", "priority": 2}, "path": "/tickets"}))

    def test_medium_retries_only_when_safe(self):
        flaky = http_tool(self.base + "/flaky", method="GET", name="flaky")
        self.assertEqual(flaky(), {"ok": True})  # the 503 was retried
        busy = http_tool(self.base + "/busy", name="busy")
        with self.assertRaises(ToolError):
            busy()
        self.assertEqual(len([h for h in _API.hits if h[1] == "/busy"]), 1)  # a POST is never sent twice
        down = http_tool("http://127.0.0.1:9/x", method="GET", name="down", retries=0)
        self.assertIn("could not reach", down.invoke({}))

    def test_high_openapi_tools(self):
        tools = {t.name: t for t in openapi_tools(self.base + "/openapi.json", headers={"X-API-Key": "k2"})}
        self.assertEqual(sorted(tools), ["createTicket", "getCustomer", "post__notes"])
        get = tools["getCustomer"]
        self.assertEqual(get.http_method, "GET")
        self.assertEqual(get.parameters["required"], ["customer_id"])
        self.assertEqual(get(customer_id="A1", fields="name")["data"]["key"], "k2")
        ticket = tools["createTicket"]
        self.assertEqual(ticket.parameters["required"], ["title"])
        self.assertEqual(ticket.parameters["properties"]["parent"], {"type": "object"})  # recursive $ref stops
        self.assertEqual(ticket(title="Refund", priority=1)["created"], {"title": "Refund", "priority": 1})
        self.assertEqual(tools["post__notes"](body=["a", "b"])["created"], ["a", "b"])
        only = openapi_tools(SPEC, base_url=self.base, include=["getCustomer"], prefix="crm_")
        self.assertEqual([t.name for t in only], ["crm_getCustomer"])
        with self.assertRaises(ToolError):
            openapi_tools(SPEC, base_url=self.base, include=["nope"])
        with self.assertRaises(ToolError):
            openapi_tools({"openapi": "3.0.0"})

    def test_complex_swagger2_and_agent(self):
        swagger = {"swagger": "2.0", "host": self.httpd.server_address[0] + f":{self.httpd.server_address[1]}",
                   "schemes": ["http"], "basePath": "", "paths": {"/tickets": {"post": {
                       "operationId": "open", "parameters": [{"name": "t", "in": "body", "schema": {
                           "type": "object", "properties": {"title": {"type": "string"}}}}]}}}}
        open_ticket = openapi_tools(swagger)[0]
        llm = FakeLLM([{"tool": "open", "args": {"title": "Broken lamp"}}, "Ticket opened."])
        result = Agent(llm, tools=[open_ticket]).run("Open a ticket for the broken lamp")
        self.assertEqual(result.output, "Ticket opened.")
        self.assertIn("Broken lamp", result.steps[0].output)


# ============================================================ A2A
class _A2A(BaseHTTPRequestHandler):
    """A small A2A agent: v1.0 or v0.3 by its card; asks for an order id; slow tasks need polling."""
    version = "1.0"
    tasks = {}

    def log_message(self, *a):
        pass

    def _reply(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = f"http://127.0.0.1:{self.server.server_address[1]}/rpc"
        if self.path == "/.well-known/agent-card.json":
            if _A2A.version == "1.0":
                return self._reply({"name": "Report Agent", "description": "Writes reports",
                                    "supportedInterfaces": [{"url": url, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
                                    "skills": [{"id": "r", "name": "Reports", "description": "Short reports"}]})
            return self._reply({"name": "Report Agent", "description": "Writes reports", "url": url,
                                "protocolVersion": "0.3.0", "preferredTransport": "JSONRPC"})
        self._reply({"error": "not found"}, 404)

    def _task(self, tid, ctx, state, text, artifact=False):
        modern = _A2A.version == "1.0"
        part = {"text": text} if modern else {"kind": "text", "text": text}
        st = ("TASK_STATE_" + state.upper().replace("-", "_")) if modern else state
        task = {"id": tid, "contextId": ctx, "status": {"state": st}}
        if artifact:
            task["artifacts"] = [{"artifactId": "a1", "parts": [part]}]
        else:
            task["status"]["message"] = {"role": "ROLE_AGENT" if modern else "agent", "parts": [part]}
        if not modern:
            task["kind"] = "task"
        return {"task": task} if modern else task

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        modern = _A2A.version == "1.0"
        send, get = ("SendMessage", "GetTask") if modern else ("message/send", "tasks/get")
        if modern and self.headers.get("A2A-Version") != "1.0":
            return self._reply({"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32009, "message": "version"}})
        if req["method"] == get:
            tid = req["params"]["id"]
            _A2A.tasks[tid] += 1
            done = _A2A.tasks[tid] >= 2
            return self._reply({"jsonrpc": "2.0", "id": req["id"],
                                "result": self._task(tid, "c1", "completed" if done else "working",
                                                     "slow report ready" if done else "", artifact=done)})
        if req["method"] != send:
            return self._reply({"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32601, "message": "no method"}})
        msg = req["params"]["message"]
        text = msg["parts"][0]["text"]
        if text == "hello":
            part = {"text": "hi there"} if modern else {"kind": "text", "text": "hi there"}
            reply = {"message": {"role": "ROLE_AGENT", "parts": [part]}} if modern else {"kind": "message", "role": "agent", "parts": [part]}
            return self._reply({"jsonrpc": "2.0", "id": req["id"], "result": reply})
        if text == "slow":
            _A2A.tasks["t-slow"] = 0
            return self._reply({"jsonrpc": "2.0", "id": req["id"], "result": self._task("t-slow", "c1", "working", "")})
        if text == "fail":
            return self._reply({"jsonrpc": "2.0", "id": req["id"], "result": self._task("t-f", "c1", "failed", "no data")})
        if msg.get("taskId") == "t-refund":
            return self._reply({"jsonrpc": "2.0", "id": req["id"],
                                "result": self._task("t-refund", "c9", "completed", f"refunded order {text}", artifact=True)})
        if "refund" in text:
            return self._reply({"jsonrpc": "2.0", "id": req["id"],
                                "result": self._task("t-refund", "c9", "input-required", "Which order id?")})
        self._reply({"jsonrpc": "2.0", "id": req["id"], "result": self._task("t1", "c1", "completed", f"report: {text}", True)})


class TestA2A(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _A2A)
        cls.base = running(cls.httpd)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def check(self, version):
        _A2A.version = version
        agent = A2AAgent(self.base, poll=0.01)
        self.assertEqual(agent.name, "report_agent")
        self.assertIn("Reports" if version == "1.0" else "Writes reports", agent.description)
        self.assertEqual(agent.run("hello").output, "hi there")
        r = agent.run("Q3 sales")
        self.assertEqual((r.output, r.state, r.ok), ("report: Q3 sales", "completed", True))
        self.assertEqual(agent.run("slow").output, "slow report ready")  # polled until done
        failed = agent.run("fail")
        self.assertEqual((failed.state, failed.output, failed.ok), ("failed", "no data", False))
        ask = agent.run("refund please")
        self.assertTrue(ask.needs_input)
        self.assertEqual(agent.reply(ask, "A7").output, "refunded order A7")
        t = agent.as_tool()
        self.assertIn("asks: Which order id?", t.invoke({"task": "refund please"}))
        self.assertEqual(t.invoke({"task": "A7"}), "refunded order A7")
        self.assertIn("failed the task", t.invoke({"task": "fail"}))

    def test_medium_a2a_v1(self):
        self.check("1.0")

    def test_medium_a2a_v03(self):
        self.check("0.3")

    def test_high_remote_agent_in_a_team(self):
        _A2A.version = "1.0"
        remote = A2AAgent(self.base, name="reporter")
        boss = Supervisor(FakeLLM([{"tool": "reporter", "args": {"task": "Q3 sales"}}, "Final: report done"]), [remote])
        result = boss.run("Get me the Q3 report")
        self.assertEqual(result.output, "Final: report done")
        self.assertEqual(result.steps[0].output, "report: Q3 sales")
        with self.assertRaises(TaskPilotError):
            A2AAgent("http://127.0.0.1:9").run("x")
        no_card = A2AAgent("http://127.0.0.1:9", endpoint=self.base + "/rpc", name="direct")
        self.assertEqual(no_card.run("hello").output, "hi there")


# ============================================================ REST API of a workspace
class TestRestAPI(unittest.TestCase):
    def setUp(self):
        def brain(messages, tools):
            names = [t["function"]["name"] for t in tools or []]
            if "send_email" in names and not any(m.get("role") == "tool" for m in messages):
                return {"tool": "send_email", "args": {"to": "ravi@example.com", "body": "Thanks"}}
            if "Choose how to do" in messages[-1]["content"]:
                return '{"pattern": "single", "agents": ["writer"], "why": "one writer is enough"}'
            return "Email is ready."

        self.ws = Workspace(FakeLLM(fn=brain), [add, send_email], approve=["send_email"], review=False, verify=False)
        self.ws.create_agent("writer", "Writes and sends emails.", ["send_email"])
        self.httpd = WebApp(self.ws, "Desk", token="t0k").server("127.0.0.1", 0)
        self.base = running(self.httpd)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, method, path, body=None, token="t0k"):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def test_easy_health_tools_and_token(self):
        self.assertEqual(self.call("GET", "/v1/health")[1]["ok"], True)
        self.assertEqual(self.call("GET", "/v1/health", token=None)[0], 401)
        self.assertEqual(self.call("GET", "/v1/health", token="wrong")[0], 401)
        self.assertEqual(self.call("GET", "/api/workspace", token=None)[0], 401)  # the page's API too
        with urllib.request.urlopen(self.base + "/") as resp:
            self.assertIn(b"Access token", resp.read())  # the page itself loads and asks once
        self.assertEqual([t["name"] for t in self.call("GET", "/v1/tools")[1]["tools"]], ["add", "send_email"])
        self.assertEqual(self.call("POST", "/v1/tools/add", {"a": 2, "b": "3"}), (200, {"output": 5}))
        self.assertEqual(self.call("POST", "/v1/tools/add", {"a": 2})[0], 400)
        self.assertEqual(self.call("POST", "/v1/tools/send_email", {"to": "x", "body": "y"})[0], 403)
        self.assertEqual(self.call("POST", "/v1/tools/ghost", {})[0], 404)

    def test_high_runs_with_approval(self):
        status, run = self.call("POST", "/v1/runs", {"request": "Thank Ravi by email"})
        self.assertEqual(status, 201)
        self.assertEqual((run["pattern"], run["why"], run["status"]), ("single", "one writer is enough", "waiting"))
        action = run["actions"][0]
        self.assertEqual(action["tool"], "send_email")
        status, done = self.call("POST", f"/v1/runs/{run['id']}/actions/{action['id']}", {"approve": True})
        self.assertEqual((status, done["status"], done["actions"][0]["status"]), (200, "ok", "done"))
        self.assertEqual(self.call("GET", f"/v1/runs/{run['id']}")[1]["status"], "ok")
        self.assertEqual(self.call("POST", f"/v1/runs/{run['id']}/actions/{action['id']}", {"approve": "yes"})[0], 400)
        self.assertEqual(self.call("GET", "/v1/runs/nope")[0], 400)
        self.assertEqual(len(self.call("GET", "/v1/runs")[1]["runs"]), 1)

    def test_complex_other_platforms_plug_in(self):
        # 1) another platform imports this workspace from its OpenAPI spec
        spec = self.call("GET", "/v1/openapi.json")[1]
        self.assertIn("/v1/tools/add", spec["paths"])
        self.assertNotIn("/v1/tools/send_email", spec["paths"])  # approval tools are not callable directly
        tools = {t.name: t for t in openapi_tools(spec, base_url=self.base, headers={"Authorization": "Bearer t0k"})}
        self.assertEqual(tools["tool_add"](a=4, b=5), {"output": 9})
        self.assertEqual(tools["health"]()["ok"], True)
        # 2) an MCP host uses the same server at /mcp
        with MCPClient.http(self.base + "/mcp", headers={"Authorization": "Bearer t0k"}) as c:
            self.assertEqual(sorted(t["name"] for t in c.list_tools()), ["add", "agent_writer", "run_request"])
            self.assertEqual(c.call("add", {"a": 1, "b": 1}), 2)
            info = c.call("run_request", {"request": "Thank Ravi", "pattern": "single", "agents": ["writer"]})
            self.assertEqual(info["status"], "waiting")
            self.ws.create_agent("notes", "Takes notes.")  # made on the web page while connected
            self.assertIn("agent_notes", [t["name"] for t in c.list_tools(refresh=True)])
        # ... and the person approves it through the REST API
        action = self.ws.get(info["run_id"]).actions[0]
        self.assertEqual(self.call("POST", f"/v1/runs/{info['run_id']}/actions/{action['id']}", {"approve": False})[1]
                         ["actions"][0]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
