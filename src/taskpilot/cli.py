"""Command line: run, inspect and draw flows defined in a Python file.

::

    taskpilot run pipeline.py:flow subject=AI count=3 --verbose
    taskpilot show pipeline.py:flow
    taskpilot graph pipeline.py:flow > flow.mmd
    taskpilot resume pipeline.py paused-run.json approve=true
    taskpilot ask tools.py "add 125 and 75, then take away 40"     # plan + run from plain words
    taskpilot serve tools.py --port 8000                           # the same, as a web page + REST API + MCP
    taskpilot mcp tools.py                                         # the tools and agents as an MCP server (stdio)
    taskpilot mcp tools.py --http --port 8765                      # ... over HTTP
    taskpilot version

A *tools file* for ``ask``/``serve`` defines a ``Workspace`` or ``AutoFlow`` object,
or a list called ``tools`` (plus optional ``agents``, ``approve``, ``llm``,
``instructions``, and the reliability settings ``review``, ``verify`` and
``escalate``). Agents made on the web page are kept in ``--store``.
Without ``llm`` the model comes from the environment (see ``taskpilot.llm.from_env``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .errors import TaskPilotError
from .flow import Flow


def _import(path: str) -> Any:
    if not os.path.exists(path):
        raise SystemExit(f"error: file not found: {path}")
    spec = importlib.util.spec_from_file_location("_taskpilot_target", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(module)
    return module


def _load_auto(target: str) -> Any:
    from .auto import AutoFlow
    from .llm import from_env

    path, _, attr = target.partition(":")
    module = _import(path)
    if attr:
        found = getattr(module, attr, None)
        if not isinstance(found, AutoFlow):
            raise SystemExit(f"error: '{attr}' in {path} is not an AutoFlow")
        return found
    from .workspace import Workspace

    spaces = [v for v in vars(module).values() if isinstance(v, Workspace)]
    if len(spaces) == 1:
        return spaces[0].auto_flow()
    autos = [v for v in vars(module).values() if isinstance(v, AutoFlow)]
    if len(autos) == 1:
        return autos[0]
    if len(autos) > 1:
        raise SystemExit(f"error: {path} defines {len(autos)} AutoFlows; name one with {path}:<name>")
    tools = getattr(module, "tools", None)
    if not tools:
        raise SystemExit(f"error: {path} needs a list called 'tools' (or an AutoFlow object)")
    return AutoFlow(getattr(module, "llm", None) or from_env(), tools, approve=getattr(module, "approve", ()),
                    instructions=getattr(module, "instructions", ""))


def _load_workspace(target: str, store: Optional[str]) -> Any:
    from .auto import AutoFlow
    from .llm import from_env
    from .workspace import Workspace

    path, _, attr = target.partition(":")
    module = _import(path)
    found = [getattr(module, attr, None)] if attr else list(vars(module).values())
    for kind in (Workspace, AutoFlow):
        objs = [v for v in found if isinstance(v, kind)]
        if len(objs) > 1:
            raise SystemExit(f"error: {path} defines several {kind.__name__}s; name one with {path}:<name>")
        if objs:
            return objs[0] if kind is Workspace else Workspace.from_auto(objs[0], store=store)
    tools = getattr(module, "tools", None)
    if not tools and not getattr(module, "agents", None):
        raise SystemExit(f"error: {path} needs a list called 'tools' (and/or 'agents'), or a Workspace object")
    return Workspace(getattr(module, "llm", None) or from_env(), tools or [], agents=getattr(module, "agents", ()),
                     approve=getattr(module, "approve", ()), instructions=getattr(module, "instructions", ""), store=store,
                     review=getattr(module, "review", True), verify=getattr(module, "verify", True),
                     escalate=getattr(module, "escalate", None))


def _ask(args: Any) -> int:
    auto = _load_auto(args.tools)
    try:
        plan = auto.load(args.plan) if args.plan else auto.plan(args.request)
    except (TaskPilotError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("Plan:\n" + plan.describe(), file=sys.stderr)
    if args.save_plan:
        plan.save(args.save_plan)
        print(f"(plan saved to {args.save_plan}; rerun it with --plan {args.save_plan})", file=sys.stderr)
    if not args.yes and input("\nRun it? [y/N] ").strip().lower() not in ("y", "yes", "haan", "ha"):
        print("Not run.", file=sys.stderr)
        return 0
    result = plan.run(verbose=args.verbose, raise_errors=False)
    while result.paused:
        answers = {}
        for req in result.pending:
            hint = " [y/N]" if req.key.startswith("approve_") else ""
            answers[req.key] = input(f"{req.question}{hint} ")
        result = plan.resume(result, answers, verbose=args.verbose, raise_errors=False)
    out = plan.output(result)
    if out is not None:
        print(out if isinstance(out, str) else json.dumps(out, indent=2, ensure_ascii=False, default=str))
    if not result.ok:
        print(result.summary(), file=sys.stderr)
    return 0 if result.ok else 1


def _load_flow(target: str) -> Flow:
    path, _, attr = target.partition(":")
    module = _import(path)
    if attr:
        flow = getattr(module, attr, None)
        if not isinstance(flow, Flow):
            raise SystemExit(f"error: '{attr}' in {path} is not a taskpilot Flow")
        return flow
    flows = [v for v in vars(module).values() if isinstance(v, Flow)]
    if len(flows) != 1:
        raise SystemExit(f"error: {path} defines {len(flows)} flows; name one with {path}:<name>")
    return flows[0]


def _parse_inputs(pairs: List[str]) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key.isidentifier():
            raise SystemExit(f"error: inputs look like name=value, got {pair!r}")
        try:
            inputs[key] = json.loads(raw)  # numbers, lists, true/false, null...
        except ValueError:
            inputs[key] = raw  # anything else is a plain string
    return inputs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="taskpilot", description="Run and inspect taskpilot flows.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a flow")
    run.add_argument("target", help="file.py or file.py:flow_variable")
    run.add_argument("inputs", nargs="*", help="name=value (values are parsed as JSON when possible)")
    run.add_argument("-v", "--verbose", action="store_true", help="print a live trace")
    run.add_argument("--json", action="store_true", help="print the full result as JSON")
    run.add_argument("--checkpoint", help="checkpoint file for resuming")
    run.add_argument("--save", default="paused-run.json", help="where to save the run if it pauses for input")

    res = sub.add_parser("resume", help="answer a paused run and continue it")
    res.add_argument("target", help="file.py or file.py:flow_variable")
    res.add_argument("paused", help="file written when the run paused")
    res.add_argument("answers", nargs="*", help="key=value answers (JSON when possible)")
    res.add_argument("-v", "--verbose", action="store_true")
    res.add_argument("--json", action="store_true")
    res.add_argument("--save", default=None, help="where to save the run if it pauses again (default: same file)")

    show = sub.add_parser("show", help="describe a flow")
    show.add_argument("target")
    graph = sub.add_parser("graph", help="print a Mermaid diagram")
    graph.add_argument("target")
    sub.add_parser("version", help="print the version")

    ask = sub.add_parser("ask", help="plan and run a workflow from a plain-language request")
    ask.add_argument("tools", help="tools file (defines 'tools' or an AutoFlow)")
    ask.add_argument("request", nargs="?", default="", help="what to do, in plain words")
    ask.add_argument("-y", "--yes", action="store_true", help="run without asking to confirm the plan")
    ask.add_argument("-v", "--verbose", action="store_true")
    ask.add_argument("--save-plan", help="save the plan as JSON to reuse it")
    ask.add_argument("--plan", help="run a saved plan instead of asking the model")

    srv = sub.add_parser("serve", help="web page where people type requests, check plans and approve steps")
    srv.add_argument("tools", help="tools file (defines 'tools' or an AutoFlow)")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--title", default="taskpilot")
    srv.add_argument("--store", default="taskpilot_workspace.json",
                     help="file that keeps agents and workflows made on the page")
    srv.add_argument("--token", default=None, help="require 'Authorization: Bearer TOKEN' (default: $TASKPILOT_TOKEN)")
    srv.add_argument("--no-mcp", action="store_true", help="don't serve /mcp")
    srv.add_argument("--allow-origin", action="append", default=[], help="another web site allowed to call the API")

    mcp = sub.add_parser("mcp", help="offer the tools, agents and workspace runs to MCP hosts")
    mcp.add_argument("tools", help="tools file (defines 'tools', a Workspace or an AutoFlow)")
    mcp.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    mcp.add_argument("--host", default="127.0.0.1")
    mcp.add_argument("--port", type=int, default=8765)
    mcp.add_argument("--token", default=None, help="with --http: require 'Authorization: Bearer TOKEN'")
    mcp.add_argument("--store", default="taskpilot_workspace.json", help="file with agents made on the web page")
    mcp.add_argument("--approval-tools", action="store_true",
                     help="also offer tools that need approval (the MCP host's confirmation is then the approval)")

    args = parser.parse_args(argv)
    if args.command == "version":
        print(f"taskpilot {__version__}")
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "ask":
        if not args.request and not args.plan:
            parser.error("ask needs a request (in quotes) or --plan FILE")
        return _ask(args)
    if args.command == "serve":
        from .web import serve

        serve(_load_workspace(args.tools, args.store), args.host, args.port, args.title,
              token=args.token, mcp=not args.no_mcp, allowed_origins=args.allow_origin)
        return 0
    if args.command == "mcp":
        from .mcp import MCPServer

        real_stdout = sys.stdout
        sys.stdout = sys.stderr  # with stdio, stdout belongs to the protocol
        try:
            server = MCPServer.from_workspace(_load_workspace(args.tools, args.store), approval_tools=args.approval_tools)
        finally:
            sys.stdout = real_stdout
        if args.http:
            server.serve_http(args.host, args.port, token=args.token or os.environ.get("TASKPILOT_TOKEN") or None)
        else:
            server.serve_stdio()
        return 0
    flow = _load_flow(args.target)
    if args.command == "show":
        print(flow.describe())
        return 0
    if args.command == "graph":
        print(flow.to_mermaid())
        return 0
    try:
        if args.command == "resume":
            result = flow.resume(args.paused, _parse_inputs(args.answers), verbose=args.verbose, raise_errors=False)
            save_to = args.save or args.paused
        else:
            result = flow.run(_parse_inputs(args.inputs), verbose=args.verbose, checkpoint=args.checkpoint, raise_errors=False)
            save_to = args.save
    except TaskPilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result.paused:
        result.save(save_to)
        print(f"Paused, waiting for input. Saved to {save_to}", file=sys.stderr)
        for req in result.pending:
            q = f" - {req.question}" if req.question else ""
            print(f"  {req.key} (step {req.step}){q}", file=sys.stderr)
        keys = " ".join(f"{r.key}=..." for r in result.pending)
        print(f"Continue with: taskpilot resume {args.target} {save_to} {keys}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        out = result.output
        print(out if isinstance(out, str) else json.dumps(out, indent=2, default=str))
        if not result.ok:
            print(result.summary(), file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
