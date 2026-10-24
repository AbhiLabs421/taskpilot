"""Workspace: everything a business user needs in one place.

* **tools** – what the system can do (written once by a developer, in code)
* **agents** – AI helpers with instructions and a set of tools; made in code
  or created by business users on the web page
* **workflow types** – how agents work together (one after another, all at
  once, pick one, do and check, team discussion...), chosen by the user or
  suggested by the AI
* **approvals** – risky tools never run without a person saying yes, even
  when an agent calls them

::

    ws = Workspace(llm, tools=[lookup_order, send_email], approve=["send_email"], store="workspace.json")
    ws.create_agent("writer", "You write short, friendly emails.", tools=["send_email"])
    tip = ws.suggest("Write and send a thank-you email to ravi@example.com")
    run = ws.start("Write and send a thank-you email to ravi@example.com", tip["pattern"], tip["agents"])
    run.actions        # the email waits here until ws.approve_action(run.id, ..., True)
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from .auto import AutoFlow, AutoPlan
from .errors import FlowDefinitionError, LLMError, TaskPilotError
from .flow import Flow
from .llm.agent import Agent
from .llm.base import LLM
from .llm.tools import Tool
from .patterns import GroupChat, Handoff, ReviewLoop, Supervisor

__all__ = ["Workspace", "AgentSpec", "WorkspaceRun", "PATTERNS"]

#: Workflow types a person can pick. ``min``/``max`` = how many agents it takes.
PATTERNS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("plan", {"label": "Step plan", "min": 0, "max": None,
              "about": "The AI plans exact tool steps (with parallel steps and approvals). Best for clear, multi-step tasks."}),
    ("single", {"label": "One agent", "min": 1, "max": 1,
                "about": "One agent does the whole job with its tools."}),
    ("sequential", {"label": "One after another", "min": 2, "max": None,
                    "about": "Agents work in order; each gets the previous agent's result (e.g. research -> write -> polish)."}),
    ("parallel", {"label": "All at once", "min": 2, "max": None,
                  "about": "Every agent works on the same request at the same time; the answers are combined."}),
    ("router", {"label": "Pick the right agent", "min": 2, "max": None,
                "about": "The AI picks the single best agent for the request."}),
    ("review", {"label": "Do and check", "min": 2, "max": 2,
                "about": "First agent does the work, second agent checks it; repeats with feedback until it is OK."}),
    ("map", {"label": "Each item separately", "min": 1, "max": 1,
             "about": "The agent handles every item of a list in parallel (one item per line)."}),
    ("supervisor", {"label": "Boss delegates", "min": 1, "max": None,
                    "about": "A boss AI splits the job and hands parts to the agents, then combines the results."}),
    ("handoff", {"label": "Pass to specialist", "min": 2, "max": None,
                 "about": "The first agent talks to the user and hands over to the right specialist."}),
    ("group_chat", {"label": "Team discussion", "min": 2, "max": None,
                    "about": "Agents discuss in turns on one conversation; the last agent decides when it is done."}),
])

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


class AgentSpec:
    """An agent described by data: name, instructions and tool names."""

    __slots__ = ("name", "instructions", "tools", "source", "description", "llm")

    def __init__(self, name: str, instructions: str, tools: Sequence[str] = (), *, source: str = "web",
                 description: str = "", llm: Optional[LLM] = None):
        self.name = name
        self.instructions = instructions
        self.tools = list(tools)
        self.source = source
        self.description = description
        self.llm = llm

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "instructions": self.instructions, "tools": self.tools,
                "source": self.source, "description": self.description}


class WorkspaceRun:
    """One job: what was asked, how it ran, the result and any approvals."""

    def __init__(self, request: str, pattern: str, agents: List[str]):
        self.id = os.urandom(6).hex()
        self.request = request
        self.pattern = pattern
        self.agents = agents
        self.status = "running"
        self.output: Any = None
        self.trace: List[Dict[str, str]] = []
        self.actions: List[Dict[str, Any]] = []
        self.error = ""
        self.started = time.time()
        self.seconds = 0.0
        self.items: List[str] = []
        #: for "plan" runs: the plan and its paused/finished RunResult
        self.plan: Optional[AutoPlan] = None
        self.result: Any = None
        self.lock = threading.Lock()

    @property
    def waiting(self) -> bool:
        return any(a["status"] == "waiting" for a in self.actions)

    def to_dict(self) -> Dict[str, Any]:
        out = self.output if isinstance(self.output, str) or self.output is None else json.dumps(self.output, ensure_ascii=False, default=str)
        return {"id": self.id, "request": self.request, "pattern": self.pattern, "agents": self.agents,
                "status": self.status, "output": out, "trace": self.trace, "actions": self.actions,
                "error": self.error, "seconds": round(self.seconds, 2), "started": self.started}


class _Gate:
    """Holds back risky tool calls made by agents until a person approves them."""

    def __init__(self, run: WorkspaceRun, tools: Dict[str, Tool]):
        self.run = run
        self.tools = tools
        self.lock = threading.Lock()  # agents may call tools from several threads at once

    def wrap(self, tool: Tool) -> Tool:
        gate = self
        held = copy.copy(tool)

        def hold(**args: Any) -> str:
            with gate.lock:
                action = {"id": f"a{len(gate.run.actions) + 1}", "tool": tool.name, "args": args, "status": "waiting", "result": ""}
                gate.run.actions.append(action)
            return (f"NOT DONE YET: '{tool.name}' is waiting for a person to approve it on the page "
                    f"(request {action['id']}). Say that it is waiting for approval; do not call it again.")

        held.fn = hold
        held.is_async = False
        return held


class Workspace:
    """Tools, agents, workflow types and approvals, ready for a web page or a script.

    Args:
        llm: the default model (agents can bring their own).
        tools: what agents and plans may use.
        agents: :class:`Agent` objects or dicts ``{"name", "instructions", "tools"}``
            defined in code (read-only on the web page).
        approve: tool names that always wait for a person.
        store: JSON file where agents and workflows created on the web are kept.
        instructions: house rules added to every planner prompt.
        review: a second model checks every step plan before it runs (default on).
            Pass a different model, or a list of models, for stronger checks.
        verify: agents' answers are checked against their tool results before
            they are shown (default on). Same options as ``review``.
        escalate: a stronger model that takes over when checks keep failing.
    """

    def __init__(self, llm: LLM, tools: Iterable[Union[Tool, Callable]] = (), *, agents: Iterable[Any] = (),
                 approve: Iterable[str] = (), store: Optional[str] = None, instructions: str = "",
                 max_rounds: int = 3, name: str = "workspace", review: Any = True, verify: Any = True,
                 escalate: Optional[LLM] = None):
        self.llm = llm
        #: a second model checks every step plan before it runs (see AutoFlow(review=...))
        self.review = review
        #: every agent answer is checked against its tool results (see Agent(verify=...))
        self.verify = verify
        #: stronger model for plans and answers that keep failing their checks
        self.escalate = escalate
        self.name = name
        self.instructions = instructions
        self.max_rounds = max_rounds
        self.tools: Dict[str, Tool] = {}
        for t in tools:
            self._add_tool(t if isinstance(t, Tool) else Tool(t))
        self.agents: "OrderedDict[str, AgentSpec]" = OrderedDict()
        for a in agents:
            self._add_code_agent(a)
        self.approve = set(approve)
        unknown = [a for a in self.approve if a not in self.tools]
        if unknown:
            raise FlowDefinitionError(f"approve= names unknown tools: {unknown}. Tools: {sorted(self.tools)}")
        self.store = os.fspath(store) if store else None
        self.workflows: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.history: "OrderedDict[str, WorkspaceRun]" = OrderedDict()
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------ setup
    def _add_tool(self, tool: Tool) -> None:
        old = self.tools.get(tool.name)
        if old is not None and old.fn is not tool.fn:
            raise FlowDefinitionError(f"Two different tools are named '{tool.name}'")
        self.tools[tool.name] = tool

    def _add_code_agent(self, a: Any) -> None:
        if isinstance(a, Agent):
            for t in a.tools.values():
                self._add_tool(t)
            spec = AgentSpec(a.name, a.system or "", list(a.tools), source="code", llm=a.llm)
        elif isinstance(a, dict):
            spec = AgentSpec(a["name"], a.get("instructions", ""), a.get("tools", []), source="code",
                             description=a.get("description", ""))
        else:
            raise FlowDefinitionError(f"agents must be Agent objects or dicts, got {type(a).__name__}")
        self._check_agent(spec, new=True)
        self.agents[spec.name] = spec

    def _check_agent(self, spec: AgentSpec, new: bool) -> None:
        if not isinstance(spec.name, str) or not _NAME.match(spec.name):
            raise FlowDefinitionError("Agent name: 2-31 lowercase letters, digits or _, starting with a letter (e.g. writer)")
        if new and spec.name in self.agents:
            raise FlowDefinitionError(f"An agent named '{spec.name}' already exists")
        if spec.name in self.tools or spec.name in ("ask_ai", "ask_human"):
            raise FlowDefinitionError(f"'{spec.name}' is already a tool name; pick another agent name")
        if not isinstance(spec.instructions, str) or len(spec.instructions) > 4000:
            raise FlowDefinitionError("Instructions must be text up to 4000 characters")
        missing = [t for t in spec.tools if t not in self.tools]
        if missing:
            raise FlowDefinitionError(f"Unknown tools {missing}. Available: {sorted(self.tools)}")

    # ------------------------------------------------------------ agents
    def create_agent(self, name: str, instructions: str, tools: Sequence[str] = (), description: str = "") -> AgentSpec:
        """Add an agent (as a business user would on the web page) and save it."""
        spec = AgentSpec(str(name).strip(), instructions, list(tools), source="web", description=description)
        with self._lock:
            self._check_agent(spec, new=True)
            self.agents[spec.name] = spec
            self._save()
        return spec

    def update_agent(self, name: str, instructions: Optional[str] = None, tools: Optional[Sequence[str]] = None,
                     description: Optional[str] = None) -> AgentSpec:
        with self._lock:
            spec = self._agent(name)
            if spec.source == "code":
                raise FlowDefinitionError(f"'{name}' is defined in code; change it there")
            new = AgentSpec(spec.name, spec.instructions if instructions is None else instructions,
                            spec.tools if tools is None else list(tools), source="web",
                            description=spec.description if description is None else description)
            self._check_agent(new, new=False)
            self.agents[name] = new
            self._save()
            return new

    def delete_agent(self, name: str) -> None:
        with self._lock:
            spec = self._agent(name)
            if spec.source == "code":
                raise FlowDefinitionError(f"'{name}' is defined in code and can't be deleted here")
            del self.agents[name]
            self._save()

    def _agent(self, name: str) -> AgentSpec:
        spec = self.agents.get(name)
        if spec is None:
            from .errors import did_you_mean

            raise FlowDefinitionError(f"No agent named '{name}'.{did_you_mean(str(name), self.agents)}")
        return spec

    def build(self, name: str, gate: Optional[_Gate] = None, *, actions: bool = True) -> Agent:
        """A ready :class:`Agent` for one run; risky tools go through ``gate``.

        ``actions=False`` leaves out the tools that need approval (for drafting).
        """
        spec = self._agent(name)
        tools = []
        for t in spec.tools:
            tool = self.tools[t]
            if t in self.approve and not actions:
                continue
            if t in self.approve:
                if gate is None:
                    raise TaskPilotError(f"Tool '{t}' needs approval; build the agent inside a run")
                tool = gate.wrap(tool)
            tools.append(tool)
        about = f"\n\n{spec.description}" if spec.description else ""
        return Agent(spec.llm or self.llm, tools=tools, name=spec.name, system=(spec.instructions + about).strip() or None,
                     verify=self.verify, escalate=self.escalate)

    # ------------------------------------------------------------ suggest
    def _catalogue(self) -> str:
        types = "\n".join(f"- {k}: {v['about']} (agents: {v['min']}" + (f"-{v['max']}" if v["max"] else "+") + ")"
                          for k, v in PATTERNS.items())
        agents = "\n".join(f"- {a.name}: {a.instructions[:200]} | tools: {', '.join(a.tools) or 'none'}"
                           for a in self.agents.values()) or "- (no agents yet)"
        tools = "\n".join(f"- {n}: {t.description}" for n, t in self.tools.items()) or "- (no tools)"
        return f"Workflow types:\n{types}\n\nAgents:\n{agents}\n\nTools:\n{tools}"

    def check_choice(self, pattern: str, agents: Sequence[str]) -> List[str]:
        """Problems with a workflow type + agent choice (empty = fine)."""
        meta = PATTERNS.get(pattern)
        if meta is None:
            return [f"unknown workflow type '{pattern}'. Types: {list(PATTERNS)}"]
        problems = [f"unknown agent '{a}'" for a in agents if a not in self.agents]
        if len(set(agents)) != len(agents):
            problems.append("an agent is listed twice")
        n = len(agents)
        if n < meta["min"]:
            problems.append(f"'{pattern}' needs at least {meta['min']} agent(s), got {n}")
        if meta["max"] is not None and n > meta["max"]:
            problems.append(f"'{pattern}' takes at most {meta['max']} agent(s), got {n}")
        return problems

    def suggest(self, request: str) -> Dict[str, Any]:
        """Ask the AI which workflow type and agents fit this request (with a reason)."""
        if not isinstance(request, str) or not request.strip():
            raise ValueError("Write a request first")
        schema = {"type": "object", "required": ["pattern", "agents", "why"],
                  "properties": {"pattern": {"type": "string", "enum": list(PATTERNS)},
                                 "agents": {"type": "array", "items": {"type": "string"}},
                                 "why": {"type": "string"},
                                 "items": {"type": "array", "items": {"type": "string"}}}}
        prompt = (
            "Choose how to do the user's request with the workflow types, agents and tools below.\n"
            "Pick the simplest type that does the job well. Use 'plan' for exact tool steps, especially "
            "calculations, data lookups and actions. Use agent types when writing, judging or discussing is needed. "
            "List agents in the order they should work (for 'review': the doer first, then the checker). "
            "For 'map', put the list items in 'items'. Explain 'why' in one short sentence for a business user.\n\n"
            f"{self._catalogue()}\n\nRequest: {request}"
        )
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        problems: List[str] = []
        for _ in range(2):
            try:
                tip = self.llm.json(messages, schema=schema)
            except LLMError as exc:
                problems = [str(exc)]
                break
            agents = [a for a in tip.get("agents", []) if isinstance(a, str)]
            problems = self.check_choice(tip.get("pattern", ""), agents if tip.get("pattern") != "plan" else [])
            if not problems:
                return {"pattern": tip["pattern"], "agents": agents if tip["pattern"] != "plan" else [],
                        "why": tip.get("why", ""), "items": [str(i) for i in tip.get("items", []) or []]}
            messages = messages + [{"role": "assistant", "content": json.dumps(tip)},
                                   {"role": "user", "content": "Problems: " + "; ".join(problems) + ". Choose again."}]
        return {"pattern": "plan", "agents": [], "items": [],
                "why": "Using a step plan (the suggestion did not fit: " + "; ".join(problems[:2]) + ")."}

    # ------------------------------------------------------------ run
    def auto_flow(self, gate: Optional[_Gate] = None) -> AutoFlow:
        """An :class:`AutoFlow` over the tools and the agents (agents become tools)."""
        tools: List[Any] = list(self.tools.values())
        if gate is not None:
            for name in self.agents:
                spec = self.agents[name]
                agent = self.build(name, gate)
                tools.append(agent.as_tool(description=f"Agent '{name}': {(spec.description or spec.instructions)[:200]}"))
        return AutoFlow(self.llm, tools, approve=self.approve, instructions=self.instructions, review=self.review,
                        escalate=self.escalate)

    def plan(self, request: str) -> WorkspaceRun:
        """Make a step plan (shown to the user before it runs)."""
        run = WorkspaceRun(request, "plan", [])
        run.plan = self.auto_flow(_Gate(run, self.tools)).plan(request)
        run.status = "planned"
        self._remember(run)
        return run

    def start(self, request: str, pattern: str, agents: Sequence[str] = (), *, items: Optional[Sequence[str]] = None,
              rounds: Optional[int] = None) -> WorkspaceRun:
        """Run a request with a chosen workflow type and agents. Never raises for agent errors."""
        if not isinstance(request, str) or not request.strip():
            raise ValueError("Write a request first")
        agents = list(agents)
        problems = self.check_choice(pattern, agents)
        if problems:
            raise FlowDefinitionError("; ".join(problems))
        if pattern == "plan":
            run = self.plan(request)
            return self.go(run.id)
        run = WorkspaceRun(request, pattern, agents)
        run.items = [str(i).strip() for i in (items or []) if str(i).strip()]
        self._remember(run)
        t0 = time.perf_counter()
        try:
            gate = _Gate(run, self.tools)
            built = [self.build(a, gate) for a in agents]
            if pattern == "review":
                # Drafting rounds can't take actions; the doer acts only after the checker says OK.
                built = [self.build(agents[0], gate, actions=False), built[1], built[0]]
            run.output = getattr(self, f"_run_{pattern}")(run, built, rounds or self.max_rounds)
            run.status = "waiting" if run.waiting else "ok"
        except Exception as exc:  # report, never crash the caller (or the web server)
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
        run.seconds = time.perf_counter() - t0
        return run

    def go(self, run_id: str) -> WorkspaceRun:
        """Run a planned step plan."""
        run = self.get(run_id)
        with run.lock:
            if run.plan is None:
                raise TaskPilotError("This run has no step plan")
            if run.result is None:
                t0 = time.perf_counter()
                run.result = run.plan.run(raise_errors=False)
                run.seconds = time.perf_counter() - t0
                self._plan_state(run)
        return run

    def answer(self, run_id: str, answers: Dict[str, Any]) -> WorkspaceRun:
        """Answer the questions a paused step plan is waiting on."""
        run = self.get(run_id)
        with run.lock:
            if run.plan is None or run.result is None or not run.result.paused:
                raise TaskPilotError("Nothing is waiting for an answer.")
            run.result = run.plan.resume(run.result, answers, raise_errors=False)
            self._plan_state(run)
        return run

    def _plan_state(self, run: WorkspaceRun) -> None:
        r = run.result
        run.output = run.plan.output(r) if run.plan else None
        run.trace = [{"who": n, "text": _short(r.steps[n].value if r.steps[n].status == "success" else
                                               (r.steps[n].reason or str(r.steps[n].error or "")))}
                     for n in r.order]
        run.status = "paused" if r.paused else ("waiting" if run.waiting else ("ok" if r.ok else "failed"))
        if r.error is not None:
            run.error = f"{type(r.error).__name__}: {r.error}"

    def approve_action(self, run_id: str, action_id: str, yes: bool) -> WorkspaceRun:
        """Approve (run now) or reject a tool call an agent was holding back."""
        run = self.get(run_id)
        with run.lock:
            action = next((a for a in run.actions if a["id"] == action_id), None)
            if action is None or action["status"] != "waiting":
                raise TaskPilotError("No such action waiting for approval.")
            if yes:
                tool = self.tools[action["tool"]]
                try:
                    action["result"] = _short(tool.fn(**tool.prepare(action["args"])))
                    action["status"] = "done"
                except Exception as exc:
                    action["result"] = f"{type(exc).__name__}: {exc}"
                    action["status"] = "failed"
            else:
                action["status"] = "rejected"
            if run.status == "waiting" and not run.waiting:
                run.status = "ok"
        return run

    def get(self, run_id: str) -> WorkspaceRun:
        run = self.history.get(run_id)
        if run is None:
            raise TaskPilotError("Unknown or expired run.")
        return run

    def _remember(self, run: WorkspaceRun) -> None:
        with self._lock:
            self.history[run.id] = run
            while len(self.history) > 100:
                self.history.popitem(last=False)

    # ------------------------------------------------------------ patterns
    def _run_single(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        out = agents[0].run(run.request).output
        run.trace.append({"who": agents[0].name, "text": out})
        return out

    def _run_sequential(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        text = ""
        for i, agent in enumerate(agents):
            task = run.request if i == 0 else (f"Original request: {run.request}\n\nWork so far (from {agents[i - 1].name}):\n"
                                              f"{text}\n\nDo your part and give the improved result.")
            text = agent.run(task).output
            run.trace.append({"who": agent.name, "text": text})
        return text

    def _run_parallel(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        flow = Flow("parallel")
        for agent in agents:
            flow.add(agent.as_step(agent.name, "{request}"))
        result = flow.run({"request": run.request})
        answers = {a.name: result[a.name] for a in agents}
        for name, text in answers.items():
            run.trace.append({"who": name, "text": text})
        joined = "\n\n".join(f"[{n}]\n{t}" for n, t in answers.items())
        combined = self.llm.ask(f"Request: {run.request}\n\nAnswers from several helpers:\n{joined}\n\n"
                                f"Combine them into one clear final answer.")
        run.trace.append({"who": "combine", "text": combined})
        return combined

    def _run_router(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        names = [a.name for a in agents]
        about = "\n".join(f"- {a.name}: {(a.system or '')[:200]}" for a in agents)
        try:
            pick = self.llm.json(f"Which one helper should handle this request?\n{about}\n\nRequest: {run.request}",
                                 schema={"type": "object", "required": ["agent"],
                                         "properties": {"agent": {"type": "string", "enum": names}, "why": {"type": "string"}}})
            chosen = pick["agent"] if pick.get("agent") in names else names[0]
            why = pick.get("why", "")
        except LLMError:
            chosen, why = names[0], "could not decide; using the first agent"
        run.trace.append({"who": "router", "text": f"chose {chosen}" + (f": {why}" if why else "")})
        agent = agents[names.index(chosen)]
        out = agent.run(run.request).output
        run.trace.append({"who": chosen, "text": out})
        return out

    def _run_review(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        doer, checker = agents[0], agents[1]
        finisher = agents[2] if len(agents) > 2 else None

        def work(request: str, feedback: Optional[str] = None) -> str:
            base = (f"{request}\n\nFor now, only write the content itself (for example the email text), "
                    f"with no notes about reviewing. Do not send anything yet.")
            task = base if not feedback else f"{base}\n\nThe reviewer said: {feedback}\nImprove your work."
            out = doer.run(task).output
            run.trace.append({"who": doer.name, "text": out})
            return out

        def check(draft: str, request: str) -> str:
            verdict = checker.run(f"Request: {request}\n\nWork to check:\n{draft}\n\nIf it fully meets the request reply "
                                  f"exactly 'OK'. Otherwise reply with short, specific feedback.").output
            run.trace.append({"who": checker.name, "text": verdict})
            return verdict

        def good(verdict: str) -> bool:
            return verdict.strip().upper().rstrip(".!").startswith("OK")

        result = ReviewLoop(work, check, until=good, max_iterations=max(1, rounds)).run(request=run.request)
        approved = good(result["review"])
        if not approved:
            run.trace.append({"who": "review", "text": f"Not approved after {rounds} round(s); nothing was done."})
            return result.output
        if finisher is not None and finisher.tools:
            done = finisher.run(
                f"{run.request}\n\nThis final version was approved by the reviewer:\n{result.output}\n\n"
                f"Now complete the request with your tools if an action is needed (for example sending it). "
                f"Use the approved version exactly. Then say briefly what you did."
            ).output
            run.trace.append({"who": f"{finisher.name} (after the check)", "text": done})
        return result.output

    def _run_map(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        agent = agents[0]
        items = run.items or [line.strip("-* ").strip() for line in run.request.splitlines()[1:] if line.strip()]
        if not items:
            raise TaskPilotError("'Each item separately' needs a list: put one item per line in the items box")
        flow = Flow("map")
        flow.add(lambda item: agent.run(f"{run.request}\n\nItem: {item}").output, name="one", dynamic=True)
        flow.add(lambda items, ctx: ctx.map("one", items), name="all")
        outs = flow.run(items=items).output
        for item, out in zip(items, outs):
            run.trace.append({"who": f"{agent.name} - {item[:40]}", "text": out})
        return "\n".join(f"- {item}: {out}" for item, out in zip(items, outs))

    def _run_supervisor(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        boss = Supervisor(self.llm, agents)
        result = boss.run(run.request)
        for step in result.steps:
            run.trace.append({"who": step.call.name, "text": f"task: {step.call.arguments.get('task', '')}\n-> {step.output}"})
        run.trace.append({"who": "supervisor", "text": result.output})
        return result.output

    def _run_handoff(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        result = Handoff(agents).run(run.request)
        run.trace.append({"who": "path", "text": " -> ".join(result.path)})
        for step in result.steps:
            run.trace.append({"who": step.call.name, "text": step.output})
        run.trace.append({"who": result.agent, "text": result.output})
        return result.output

    def _run_group_chat(self, run: WorkspaceRun, agents: List[Agent], rounds: int) -> str:
        chat = GroupChat(agents, rounds=max(2, rounds * len(agents)), finishers=[agents[-1].name])
        result = chat.run(run.request)
        for m in result.messages[1:]:
            run.trace.append({"who": m["speaker"], "text": m["content"]})
        return result.output

    # ------------------------------------------------------------ saved workflows
    def save_workflow(self, name: str, run_id: str) -> Dict[str, Any]:
        """Keep a run's setup under a name so it can be run again with one click."""
        name = str(name).strip()
        if not name or len(name) > 60:
            raise FlowDefinitionError("Give the workflow a name (up to 60 characters)")
        run = self.get(run_id)
        entry: Dict[str, Any] = {"request": run.request, "pattern": run.pattern, "agents": run.agents, "items": run.items}
        if run.plan is not None:
            entry["plan"] = run.plan.to_dict()
        with self._lock:
            self.workflows[name] = entry
            self._save()
        return entry

    def delete_workflow(self, name: str) -> None:
        with self._lock:
            if name not in self.workflows:
                raise FlowDefinitionError(f"No saved workflow '{name}'")
            del self.workflows[name]
            self._save()

    def run_workflow(self, name: str) -> WorkspaceRun:
        """Run a saved workflow. Saved step plans run without asking the AI again."""
        entry = self.workflows.get(name)
        if entry is None:
            raise FlowDefinitionError(f"No saved workflow '{name}'")
        if entry.get("plan"):
            run = WorkspaceRun(entry["request"], "plan", [])
            run.plan = self.auto_flow(_Gate(run, self.tools)).load(entry["plan"])
            run.status = "planned"
            self._remember(run)
            return self.go(run.id)
        return self.start(entry["request"], entry["pattern"], entry.get("agents", []), items=entry.get("items"))

    # ------------------------------------------------------------ storage
    def _load(self) -> None:
        if not self.store or not os.path.exists(self.store):
            return
        try:
            with open(self.store, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise TaskPilotError(f"Cannot read workspace file {self.store}: {exc}") from None
        for a in data.get("agents", []):
            spec = AgentSpec(a["name"], a.get("instructions", ""), a.get("tools", []), source="web",
                             description=a.get("description", ""))
            spec.tools = [t for t in spec.tools if t in self.tools]  # a tool removed from code is dropped, not fatal
            if spec.name not in self.agents:
                self._check_agent(spec, new=True)
                self.agents[spec.name] = spec
        for name, entry in (data.get("workflows") or {}).items():
            self.workflows[name] = entry

    def _save(self) -> None:
        if not self.store:
            return
        data = {"taskpilot_workspace": 1,
                "agents": [a.to_dict() for a in self.agents.values() if a.source == "web"],
                "workflows": self.workflows}
        folder = os.path.dirname(os.path.abspath(self.store))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".taskpilot-", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self.store)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------ views
    def overview(self) -> Dict[str, Any]:
        """Tools, agents, workflow types and saved workflows, for a UI."""
        return {
            "tools": [{"name": n, "description": t.description, "approval": n in self.approve,
                       "args": {k: v.get("type", "any") for k, v in t.parameters["properties"].items()},
                       "required": t.parameters["required"]} for n, t in self.tools.items()],
            "agents": [a.to_dict() for a in self.agents.values()],
            "patterns": [{"key": k, **v} for k, v in PATTERNS.items()],
            "workflows": [{"name": n, **{k: v for k, v in e.items() if k != "plan"}, "saved_plan": bool(e.get("plan"))}
                          for n, e in self.workflows.items()],
        }

    @classmethod
    def from_auto(cls, auto: AutoFlow, **kw: Any) -> "Workspace":
        tools = [t for n, t in auto.tools.items() if n not in ("ask_ai", "ask_human")]
        kw.setdefault("review", auto.review)
        kw.setdefault("escalate", auto.escalate)
        return cls(auto.llm, tools, approve=auto.approve, instructions=auto.instructions, **kw)

    def __repr__(self) -> str:
        return f"Workspace(tools={list(self.tools)}, agents={list(self.agents)})"


def _short(value: Any, limit: int = 2000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."
