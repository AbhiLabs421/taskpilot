# Using taskpilot inside another platform

Say you run an AI platform that already has its own **tool service** and
**agent service**. taskpilot connects to both. It can also be offered back to
the platform as a tool server or a REST service. Everything here uses the
standard library only.

| the platform has | in taskpilot |
|---|---|
| tools behind **MCP** (Model Context Protocol) | `MCPClient.stdio(...)` / `MCPClient.http(url)` and then `.tools()` |
| a REST API with an **OpenAPI / Swagger** spec | `openapi_tools(spec_url, headers=...)` |
| one REST endpoint | `http_tool(url, name=..., params={...})` |
| agents that speak **A2A** (Agent2Agent 1.0 or 0.3) | `A2AAgent(url)`: use `.run()`, `.as_tool()`, or put it in `Supervisor` |
| its own LLM gateway | `OpenAICompatible(base_url=..., api_key=..., headers=...)`, or subclass `LLM` |
| other agents that want taskpilot's tools | `MCPServer([...]).serve_stdio()`, or `taskpilot mcp tools.py` |
| apps that want to start workflows | `taskpilot serve tools.py --token ...` gives a REST API at `/v1` and MCP at `/mcp` |

Every imported tool works like a local one: in an `Agent`, an `AutoFlow` plan,
a `Workspace` (approval rules included) or a `Flow` step.

## MCP: use a platform's tools

```python
from taskpilot import Agent, AutoFlow, Workspace
from taskpilot.mcp import MCPClient

fs  = MCPClient.stdio("npx -y @modelcontextprotocol/server-filesystem /data")      # a local server process
crm = MCPClient.http("https://tools.example.com/mcp", headers={"Authorization": "Bearer ..."})

tools = fs.tools(prefix="fs_") + crm.tools(include=["find_customer", "create_ticket"])
agent = Agent(llm, tools=tools)
ws = Workspace(llm, tools, approve=["create_ticket"])        # approvals still apply to remote tools
```

* It connects on first use. It is thread-safe, and parallel calls really run side by side.
* Tool names that models refuse (`crm.find`, names over 64 characters) are made safe. The original name stays in `tool.mcp_name`.
* When a remote tool fails, the model sees the error and can recover. `client.call()` raises `ToolError`.
* A crashed server, a timeout or an expired HTTP session gives a clear error message. An expired session reconnects once by itself.
* Also: `client.resources()`, `client.read(uri)`, `client.prompts()`, `client.prompt(name, args)`, `client.ping()`.

## MCP: offer taskpilot to MCP hosts

```python
from taskpilot.mcp import MCPServer

server = MCPServer([add, report_flow, writer_agent, crm_client], name="office")
server.serve_stdio()               # how desktop apps, IDEs and most hosts start local servers
server.serve_http(port=8765, token="SECRET")   # Streamable HTTP at /mcp
```

You can pass tools, plain functions, a `Flow` (its inputs become the
arguments), an `Agent` (it takes a `task`), or an `MCPClient` (its tools pass
straight through, so taskpilot works as a gateway).

From the command line, any tools file works (the same file `taskpilot serve` uses):

```bash
taskpilot mcp tools.py                     # stdio
taskpilot mcp tools.py --http --port 8765 --token SECRET
```

```json
{"mcpServers": {"office": {"command": "taskpilot", "args": ["mcp", "/path/to/tools.py"]}}}
```

A workspace offers its tools, one `agent_<name>` tool per agent, and
`run_request` (a full workspace run that picks the workflow type itself).
**Tools that need approval are not offered.** Agents run without them, so
they can draft but not act, and anything `run_request` would do that needs
approval waits for a person on the web page or the REST API. Pass
`--approval-tools` to offer those tools anyway. The MCP host's own
confirmation then counts as the approval.

## REST and OpenAPI tools

```python
from taskpilot.connect import http_tool, openapi_tools

find = http_tool("https://crm.example.com/customers/{customer_id}", method="GET", name="find_customer",
                 description="Look up a customer", headers={"X-API-Key": "..."}, pick="data")

tools = openapi_tools("https://api.example.com/openapi.json", headers={"Authorization": "Bearer ..."})
ws = Workspace(llm, tools, approve=[t.name for t in tools if t.http_method != "GET"])   # writes need a person
```

* Path, query and header parameters and the fields of the JSON body become the tool's arguments. Local `$ref`s are resolved, including recursive ones.
* OpenAPI 3 and Swagger 2, as a dict, a JSON file or a URL. YAML needs a converter, because there are no dependencies.
* Outages and 5xx/429 errors are retried **only for requests that are safe to repeat** (GET, PUT, DELETE). A POST that might send an email is never sent twice.

## A2A: remote agents

```python
from taskpilot.connect import A2AAgent

reporter = A2AAgent("https://agents.example.com/reporter", headers={"Authorization": "Bearer ..."})
r = reporter.run("Q3 sales summary")
if r.needs_input:                      # the agent asked a question
    r = reporter.reply(r, "Only the North region")
team = Supervisor(llm, [reporter, local_writer])      # mix remote and local agents
```

The agent card (`/.well-known/agent-card.json`) gives the endpoint, protocol
version, name and skills. Tasks that are still working are polled until they
finish. As a tool, a remote agent's question goes back to the calling model,
and its next call answers it.

**Check what a team says about remote answers.** In a live test the remote
agent replied only "all good". The supervisor (gpt-oss:120b) then wrote a
report with revenue and growth figures it had invented. With
`Supervisor(..., verify=Ollama("nemotron-3-super"))` the made-up numbers were
caught and removed in 3 of 3 runs. The Workspace turns `verify` on by default.

## The REST API (`taskpilot serve`)

One server, three doors: the web page at `/`, the REST API at `/v1`, and MCP at `/mcp`.

```bash
taskpilot serve tools.py --token SECRET            # or set TASKPILOT_TOKEN
curl -H "Authorization: Bearer SECRET" localhost:8000/v1/openapi.json
```

| call | does |
|---|---|
| `GET /v1/health`, `/v1/tools`, `/v1/agents`, `/v1/patterns`, `/v1/workflows` | what is there |
| `POST /v1/tools/{name}` (body: arguments) | call one tool (403 for tools that need approval) |
| `POST /v1/suggest` `{"request"}` | the AI's choice of workflow type and agents, with a reason |
| `POST /v1/runs` `{"request", "pattern"?, "agents"?, "items"?, "plan_only"?}` | do a request; no `pattern` lets the AI choose |
| `GET /v1/runs`, `GET /v1/runs/{id}` | status, output, trace, actions waiting for approval |
| `POST /v1/runs/{id}/actions/{action_id}` `{"approve": true}` | approve or reject an agent's held-back action |
| `POST /v1/runs/{id}/go`, `/v1/runs/{id}/answer` `{"answers"}` | run a planned step plan; answer its questions and approvals |
| `POST /v1/workflows/{name}/run` | run a saved workflow |

`/v1/openapi.json` describes all of it, so another platform can import
taskpilot the same way taskpilot imports others (`openapi_tools(...)` works on it).

**Security:** the server listens on 127.0.0.1 by default. Requests sent by other
web sites are refused (DNS-rebinding protection; allow your own front end with
`--allow-origin`). With a token, every API call and the page's own calls need
it. The page asks for the token once. Not built in yet: user accounts, roles
and per-tenant data. Put your own login in front for multi-user setups.

## Tested against the official SDKs

Before release, the MCP code was checked in all four directions against the
official `mcp` Python SDK 2.3: its server with our client and our server with
its client, over both stdio and Streamable HTTP, including error results and
dotted tool names. The A2A client was checked against the official `a2a-sdk`
1.2 server, in its 1.0 and 0.3 modes, including multi-turn input-required
tasks. These SDKs are *not* dependencies. The unit tests (`tests/test_connect.py`)
use small stdlib servers, and `tests/test_live_ollama.py` has a real model
using MCP tools.

Not covered: MCP sampling and elicitation (a server asking the client's
model or user something), MCP OAuth login flows (pass a token in `headers=`),
A2A streaming and push notifications, and gRPC.
