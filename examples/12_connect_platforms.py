"""Plug taskpilot into a platform that already has its own tool and agent services.

The "platform" here is simulated on this machine, so the example runs offline:

* a tool service that speaks MCP          -> MCPClient(...).tools()
* a REST API with an OpenAPI spec         -> openapi_tools(...)
* a remote agent that speaks A2A          -> A2AAgent(...)
* and the other way round: taskpilot's own tools offered as an MCP server

    python examples/12_connect_platforms.py
    OLLAMA_API_KEY=... python examples/12_connect_platforms.py    # a real model drives the team
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from taskpilot import Agent, FakeLLM, Supervisor, tool  # noqa: E402
from taskpilot.connect import A2AAgent, openapi_tools  # noqa: E402
from taskpilot.mcp import MCPClient, MCPServer  # noqa: E402


# ---------------------------------------------------------------- 1. the platform's MCP tool service
@tool
def find_customer(email: str) -> dict:
    """Find a customer by email: name and tier."""
    return {"ravi@example.com": {"name": "Ravi", "tier": "gold"}}.get(email, {"error": "no such customer"})


if "--mcp-server" in sys.argv:  # this same file, started as the platform's MCP server
    MCPServer([find_customer], name="crm").serve_stdio()
    sys.exit(0)


# ---------------------------------------------------------------- 2. its REST API and 3. its A2A agent
SPEC = {"openapi": "3.0.0", "servers": [{"url": "/"}], "paths": {"/tickets": {"post": {
    "operationId": "create_ticket", "summary": "Open a support ticket",
    "requestBody": {"content": {"application/json": {"schema": {
        "type": "object", "required": ["title"], "properties": {"title": {"type": "string"}}}}}}}}}}


class Platform(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        port = self.server.server_address[1]
        if self.path == "/openapi.json":
            return self.reply(SPEC)
        if self.path == "/.well-known/agent-card.json":
            return self.reply({"name": "reporter", "description": "Writes one-line customer reports",
                               "supportedInterfaces": [{"url": f"http://127.0.0.1:{port}/a2a",
                                                        "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tickets":
            return self.reply({"ticket": "T-1001", "title": body["title"]})
        text = body["params"]["message"]["parts"][0]["text"]  # an A2A SendMessage
        return self.reply({"jsonrpc": "2.0", "id": body["id"], "result": {"message": {
            "role": "ROLE_AGENT", "parts": [{"text": f"Report: {text[:60]} (all orders delivered)"}]}}})


httpd = ThreadingHTTPServer(("127.0.0.1", 0), Platform)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{httpd.server_address[1]}"

# ---------------------------------------------------------------- taskpilot uses all three
with MCPClient.stdio([sys.executable, __file__, "--mcp-server"]) as crm:
    tools = crm.tools() + openapi_tools(base + "/openapi.json")
    reporter = A2AAgent(base)
    print("tools from the platform:", [t.name for t in tools])
    print("remote agent:", reporter.name, "-", reporter.description)

    print("MCP tool:", crm.call("find_customer", {"email": "ravi@example.com"}))
    print("REST tool:", tools[1](title="Late delivery"))
    print("A2A agent:", reporter.run("Ravi's orders").output)

    if os.environ.get("OLLAMA_API_KEY"):
        from taskpilot import Ollama

        llm = Ollama("gpt-oss:120b", options={"temperature": 0})
    else:  # scripted, so the example runs offline
        llm = FakeLLM([{"tool": "support", "args": {"task": "Find ravi@example.com and open a ticket 'Late delivery'"}},
                       {"tool": "find_customer", "args": {"email": "ravi@example.com"}},
                       {"tool": "create_ticket", "args": {"title": "Late delivery"}},
                       "Ravi (gold) has ticket T-1001.",
                       {"tool": "reporter", "args": {"task": "Ravi's orders"}},
                       "Ravi (gold): ticket T-1001 opened. Report: all orders delivered."])
    support = Agent(llm, tools=tools, name="support", system="Support agent. Use the tools.")
    team = Supervisor(llm, [support, reporter], system="Use support for customers and tickets, reporter for reports.")
    result = team.run("Ravi (ravi@example.com) says his delivery is late. Open a ticket and get a report on his orders.")
    print("team answer:", result.output)
    print("used:", result.tools_used)

httpd.shutdown()

# ---------------------------------------------------------------- the other way round
# Any MCP host (an IDE, another platform) can use taskpilot's tools, flows and agents:
#     MCPServer([find_customer, my_flow, my_agent]).serve_stdio()     or    taskpilot mcp tools.py
#     taskpilot serve tools.py --token SECRET    ->  web page + REST API (/v1) + MCP (/mcp) on one port
