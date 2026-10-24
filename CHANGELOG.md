# Changelog

## 0.1.0 (first release)

* Workflows from plain functions: parameters named after steps get their results; parallel by default.
* Step options: retry, timeout, cache, conditions, map, fallback, checks, checkpoints, `lean` memory release.
* Loops, human-in-the-loop pause/resume, dynamic steps at runtime, tracing (JSONL, console, OpenTelemetry).
* Patterns: Sequential, Parallel, Router, MapReduce, ReviewLoop, Vote, Supervisor, Handoff, GroupChat.
* LLM clients on the standard library: Ollama, OpenAI-compatible (incl. Azure), fallback; tools and agents.
* AutoFlow (plain-language request → checked plan → run), Workspace and a web page for business users.
* Reliability guards: plan review panels, answer verification, consensus, escalation, evals, chaos testing.
* Memory: auto-compacting conversation memory and a long-term fact store.
* Integrations: MCP client and server, REST/OpenAPI tools, A2A remote agents, REST API (`taskpilot serve`).
