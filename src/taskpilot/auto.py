"""AutoFlow: describe the job in plain words; the workflow is planned and run for you.

A developer registers tools once. After that anyone can ask::

    auto = AutoFlow(llm, tools=[add_numbers, subtract_numbers, send_email], approve=["send_email"])
    result = auto.run("Add 125 and 75, take away 40, and email the answer to ravi@example.com")
    print(result.plan.describe())     # what it decided to do, step by step
    print(result.output)

The model only *plans*: it returns a list of tool calls. The plan is checked
(real tools, required arguments, no loops) and then executed by the normal
taskpilot engine, so independent steps run in parallel, failed steps are
retried, and tools listed in ``approve`` always wait for a person first, no
matter what the model wrote. Plans can be saved and re-run without the model.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from .engine import Skip
from .errors import FlowDefinitionError, LLMError, TaskPilotError
from .flow import Flow
from .llm.base import LLM, check_schema
from .llm.tools import Tool
from .result import RunResult

__all__ = ["AutoFlow", "AutoPlan", "PlanStep", "AutoResult", "Unclear", "Disagreement"]

_REF = re.compile(r"\{\{\s*([A-Za-z_]\w*)((?:\.[A-Za-z_0-9]\w*)*)\s*\}\}")
_ID = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_RESERVED = {"ctx", "context", "inputs"}

ASK_AI = "ask_ai"
ASK_HUMAN = "ask_human"

PLAN_SCHEMA = {
    "type": "object",
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "tool", "args"],
                "properties": {
                    "id": {"type": "string"},
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                    "note": {"type": "string"},
                },
            },
        },
        "answer": {"type": "string"},
        "reason": {"type": "string"},
    },
}


REVIEW_SCHEMA = {
    "type": "object",
    "required": ["ok", "problems"],
    "properties": {
        "ok": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "ambiguous": {"type": "boolean"},
        "question": {"type": "string"},
    },
}

REVIEW_PROMPT = (
    "You review a plan made by another AI, before it runs. The plan must do exactly what the request asks.\n\n"
    "Request: {request}\n\nTools:\n{tools}\n\nPlan:\n{plan}\n\n{hints}"
    "Check carefully:\n"
    "1. Every number, name, id and email in the plan comes from the request, or is correctly derived from it "
    "(for example 10% -> 0.10, a 10% discount -> multiply by 0.90).\n"
    "2. Working through the steps gives exactly the result the request asks for (right operations, right order, "
    "right inputs).\n"
    "3. Nothing the request asks for is missing, and nothing extra is done.\n"
    "4. For every amount, decide whether the request makes it clear if it is per unit (per day, per item, each) "
    "or a total. If the words leave this open and the two readings give different results, the request is "
    "ambiguous. Do not pick a reading yourself.\n"
    "5. If the request can reasonably mean two different things that give different results, set ambiguous=true "
    "and write the one short question to ask the user.\n"
    "Return JSON: {{\"ok\": true|false, \"problems\": [\"...\"], \"ambiguous\": true|false, \"question\": \"...\"}}."
)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers_in_text(text: str) -> set:
    found = set()
    for n in _NUM.findall(text.replace(",", "")):
        v = float(n)
        found.update({v, v / 100, 1 - v / 100, 1 + v / 100})
    return found


def _numbers_in_args(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, str) and not _REF.search(value):
        try:
            out.append(float(value.replace(",", "")))
        except ValueError:
            pass
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_numbers_in_args(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_numbers_in_args(v))
    return out


def _reviewers(review: Any, default: LLM) -> List[LLM]:
    if isinstance(review, LLM):
        return [review]
    if isinstance(review, (list, tuple)):
        found = [r for r in review if isinstance(r, LLM)]
        if not found:
            raise FlowDefinitionError("review=[...] must contain LLM objects")
        return found
    return [default]


class Unclear(TaskPilotError):
    """The request can mean different things; ask the person this question first."""

    def __init__(self, question: str, problems: Optional[List[str]] = None):
        self.question = question
        self.problems = list(problems or [])
        super().__init__(f"Please clarify: {question}")


class Disagreement(Unclear):
    """Several planners made different plans, so none is trusted."""

    def __init__(self, groups: int, failures: List[str]):
        self.groups = groups
        self.failures = failures
        super().__init__("The planners made different plans for this request. Please say more exactly what you want.",
                         failures)


def _canonical(plan: "AutoPlan") -> str:
    """A form of the plan that ignores step ids and order, for comparing plans."""
    steps = {s.id: s for s in plan.steps}
    memo: Dict[str, str] = {}

    def value(v: Any) -> Any:
        if isinstance(v, str):
            whole = _REF.fullmatch(v.strip())
            if whole:
                return f"<{expr(whole.group(1))}{whole.group(2)}>"
            return _REF.sub(lambda m: f"<{expr(m.group(1))}{m.group(2)}>", v.strip())
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return f"{round(float(v), 9):g}"
        if isinstance(v, dict):
            return {k: value(x) for k, x in sorted(v.items())}
        if isinstance(v, list):
            return [value(x) for x in v]
        return str(v)

    def expr(sid: str) -> str:
        if sid not in memo:
            s = steps[sid]
            memo[sid] = f"{s.tool}{json.dumps(value(s.args), sort_keys=True, ensure_ascii=False)}"
        return memo[sid]

    return "|".join(sorted(expr(s.id) for s in plan.steps))


class PlanStep:
    """One tool call in a plan."""

    __slots__ = ("id", "tool", "args", "note")

    def __init__(self, id: str, tool: str, args: Dict[str, Any], note: str = ""):
        self.id = id
        self.tool = tool
        self.args = args
        self.note = note

    def refs(self) -> List[str]:
        """Ids of the steps whose results this step uses."""
        found: List[str] = []

        def walk(v: Any) -> None:
            if isinstance(v, str):
                for m in _REF.finditer(v):
                    if m.group(1) not in found:
                        found.append(m.group(1))
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)

        walk(self.args)
        return found

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "tool": self.tool, "args": self.args}
        if self.note:
            d["note"] = self.note
        return d

    def __repr__(self) -> str:
        return f"PlanStep({self.id!r}, {self.tool!r}, {self.args!r})"


def _lookup(value: Any, path: str) -> Any:
    for part in [p for p in path.split(".") if p]:
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.isdigit():
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return value


def _fill(value: Any, results: Dict[str, Any]) -> Any:
    """Replace ``{{step}}`` / ``{{step.field}}`` with real results."""
    if isinstance(value, str):
        whole = _REF.fullmatch(value.strip())
        if whole:  # the whole value is a reference: keep its type (number, list...)
            return _lookup(results[whole.group(1)], whole.group(2))

        def text(m: "re.Match[str]") -> str:
            v = _lookup(results[m.group(1)], m.group(2))
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)

        return _REF.sub(text, value)
    if isinstance(value, dict):
        return {k: _fill(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, results) for v in value]
    return value


def _show(value: Any) -> str:
    if isinstance(value, str):
        m = _REF.fullmatch(value.strip())
        if m:
            return f"<result of {m.group(1)}{m.group(2)}>"
        return json.dumps(_REF.sub(lambda m: f"<{m.group(1)}{m.group(2)}>", value), ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, default=str)


class AutoPlan:
    """A checked plan: steps, the tools they use, and how to run them."""

    def __init__(self, request: str, steps: List[PlanStep], answer: Optional[str], tools: Dict[str, Tool],
                 approve: Iterable[str] = (), retry: int = 2, name: str = "auto"):
        self.request = request
        self.steps = steps
        self.answer = answer or (steps[-1].id if steps else None)
        self.tools = tools
        self.approve = set(approve)
        self.retry = retry
        self.name = name
        self._flow: Optional[Flow] = None

    # ------------------------------------------------------------ reading
    def describe(self) -> str:
        """The plan in plain words, one line per step."""
        if not self.steps:
            return "Nothing to do."
        lines = []
        for i, s in enumerate(self.steps, 1):
            args = ", ".join(f"{k}={_show(v)}" for k, v in s.args.items())
            gate = "  [needs approval]" if s.tool in self.approve else ""
            after = f"  (after {', '.join(s.refs())})" if s.refs() else "  (can start right away)"
            note = f" - {s.note}" if s.note else ""
            lines.append(f"{i}. {s.id}: {s.tool}({args}){gate}{after}{note}")
        lines.append(f"Answer: result of {self.answer}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"taskpilot_plan": 1, "request": self.request, "answer": self.answer,
                "steps": [s.to_dict() for s in self.steps]}

    def save(self, path: str) -> str:
        """Save the plan as JSON (readable, editable, reusable without the model)."""
        with open(os.fspath(path), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return os.fspath(path)

    # ------------------------------------------------------------ running
    def flow(self) -> Flow:
        """The plan as an ordinary :class:`~taskpilot.Flow`."""
        if self._flow is None:
            self._flow = _build_flow(self)
        return self._flow

    def to_mermaid(self) -> str:
        return self.flow().to_mermaid()

    def run(self, **run_options: Any) -> RunResult:
        """Run the plan. Same options as :meth:`Flow.run` (``verbose``, ``checkpoint``...)."""
        return self.flow().run(**run_options)

    async def arun(self, **run_options: Any) -> RunResult:
        return await self.flow().arun(**run_options)

    def resume(self, paused: Any, answers: Optional[Dict[str, Any]] = None, **kw: Any) -> RunResult:
        """Continue after a person answered (``approve_<step>=True`` or an ``ask_human`` step id)."""
        return self.flow().resume(paused, answers, **kw)

    def output(self, result: RunResult) -> Any:
        return result.get(self.answer) if self.answer else None

    def __repr__(self) -> str:
        return f"AutoPlan({len(self.steps)} steps, answer={self.answer!r})"


def _build_flow(plan: AutoPlan) -> Flow:
    import inspect

    flow = Flow(plan.name, description=plan.request[:200])
    for step in plan.steps:
        deps = step.refs()
        tool = plan.tools[step.tool]
        gated = step.tool in plan.approve

        def run_step(ctx: Any, __step: PlanStep = step, __tool: Tool = tool, __gated: bool = gated, **done: Any) -> Any:
            args = _fill(__step.args, done)
            if __gated:
                ok = ctx.wait_for(f"approve_{__step.id}", question=f"Allow {__tool.name}({json.dumps(args, ensure_ascii=False, default=str)[:300]})?",
                                  tool=__tool.name, args=args)
                if not _yes(ok):
                    raise Skip(f"{__tool.name} was not approved")
            if __tool.name == ASK_HUMAN:
                return ctx.wait_for(__step.id, question=str(args.get("question", "")))
            return __tool.fn(**__tool.prepare(args))

        run_step.__name__ = step.id
        run_step.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY)]
            + [inspect.Parameter(d, inspect.Parameter.KEYWORD_ONLY) for d in deps]
        )
        flow.add(run_step, name=step.id, retry=plan.retry if step.tool != ASK_HUMAN else None,
                 description=step.note or f"{step.tool}")
    return flow


def _yes(answer: Any) -> bool:
    if isinstance(answer, str):
        return answer.strip().lower() in ("y", "yes", "true", "ok", "approve", "approved", "haan", "ha", "1")
    return bool(answer)


class AutoResult:
    """What :meth:`AutoFlow.run` returns: the plan and the run."""

    def __init__(self, plan: AutoPlan, result: Optional[RunResult]):
        self.plan = plan
        self.result = result

    @property
    def output(self) -> Any:
        return self.plan.output(self.result) if self.result is not None else None

    @property
    def paused(self) -> bool:
        return bool(self.result is not None and self.result.paused)

    @property
    def pending(self) -> list:
        return list(self.result.pending) if self.result is not None else []

    @property
    def ok(self) -> bool:
        return bool(self.result is not None and self.result.ok)

    def resume(self, answers: Optional[Dict[str, Any]] = None, **kw: Any) -> "AutoResult":
        """Answer the pending questions and continue."""
        if self.result is None:
            raise TaskPilotError("This plan has not run yet.")
        return AutoResult(self.plan, self.plan.resume(self.result, answers, **kw))

    def __repr__(self) -> str:
        status = self.result.status if self.result is not None else "planned"
        return f"AutoResult({status}, output={self.output!r})"


class AutoFlow:
    """Turn plain-language requests into workflows and run them.

    Args:
        llm: the model that plans (any :class:`~taskpilot.LLM`).
        tools: functions, :class:`Tool` s, agents (``agent.as_tool()``),
            teams or whole flows (``flow.as_tool()``) the plan may use.
        approve: tool names that must be approved by a person before they run.
            Enforced by the engine, whatever the model planned.
        allow_ai: add the built-in ``ask_ai`` tool (write, summarise, classify text).
        allow_human: add the built-in ``ask_human`` tool (ask a person a question).
        max_steps: refuse plans longer than this.
        repair: how many times the model may fix an invalid plan.
        retry: attempts for every step when it fails.
        instructions: extra rules for the planner (house rules, tone...).
        review: ``True`` (the planner checks its own plan), an :class:`LLM`, or a
            list of LLMs (a review panel) that check each plan against the request
            before it runs: numbers and names copied correctly, the steps compute
            what was asked, nothing missing or extra, no unclear per-unit/total
            amounts. Every reviewer must agree. Problems go back to the planner to
            fix; if any reviewer finds the request ambiguous, :class:`Unclear` is
            raised with a question to ask. A *different* model catches far more
            mistakes than the planner checking itself.
        consensus: an int (plan that many times) or a list of planner LLMs; a
            plan runs only when more than half of them produce the same plan,
            otherwise :class:`Disagreement` (an :class:`Unclear`) is raised.
        escalate: a stronger LLM that plans again when the plan can't be made
            to pass its checks.
    """

    def __init__(self, llm: LLM, tools: Iterable[Union[Tool, Callable]] = (), *, approve: Iterable[str] = (),
                 allow_ai: bool = True, allow_human: bool = True, max_steps: int = 20, repair: int = 2,
                 retry: int = 2, instructions: str = "", name: str = "auto", review: Any = False,
                 consensus: Any = None, escalate: Optional[LLM] = None):
        self.llm = llm
        self.review = review
        self.consensus = consensus
        self.escalate = escalate
        self.tools: Dict[str, Tool] = {}
        for t in tools:
            if not isinstance(t, Tool):
                # agents, teams and flows know how to become a tool; plain functions get wrapped
                t = t.as_tool() if hasattr(t, "as_tool") else Tool(t)
            if t.name in self.tools:
                raise FlowDefinitionError(f"Two tools are named '{t.name}'")
            if t.name in (ASK_AI, ASK_HUMAN):
                raise FlowDefinitionError(f"'{t.name}' is a built-in tool name; rename your tool")
            self.tools[t.name] = t
        if allow_ai:
            self.tools[ASK_AI] = Tool(self._ask_ai, name=ASK_AI,
                                      description="Use the AI for text work: write, summarise, translate, classify or explain. "
                                                  "Put everything it needs into the prompt.")
        if allow_human:
            def ask_human(question: str) -> str:
                """Ask a person a question and wait for the answer (approvals, missing details)."""
                return question

            self.tools[ASK_HUMAN] = Tool(ask_human, name=ASK_HUMAN)
        self.approve = set(approve)
        unknown = [a for a in self.approve if a not in self.tools]
        if unknown:
            raise FlowDefinitionError(f"approve= names unknown tools: {unknown}. Tools: {sorted(self.tools)}")
        self.max_steps = max_steps
        self.repair = repair
        self.retry = retry
        self.instructions = instructions
        self.name = name

    def _ask_ai(self, prompt: str) -> str:
        """Use the AI for text work."""
        return self.llm.ask(prompt)

    # ------------------------------------------------------------ planning
    def _system(self) -> str:
        specs = json.dumps([t.spec()["function"] for t in self.tools.values()], ensure_ascii=False)
        rules = (
            "You are a workflow planner. Turn the user's request into a plan of tool calls.\n"
            f"Tools you may use (JSON schema of their arguments):\n{specs}\n\n"
            "Return JSON: {\"steps\": [{\"id\": \"s1\", \"tool\": \"<tool name>\", \"args\": {...}, \"note\": \"short reason\"}], "
            "\"answer\": \"<id of the step whose result answers the user>\"}.\n"
            "Rules:\n"
            "- Use only the tools listed. Fill every required argument with a concrete value from the request.\n"
            "- Ids are s1, s2, s3... To use a step's result, write \"{{s1}}\" as the whole argument value "
            "(keeps numbers as numbers), or put {{s1}} inside a text. {{s1.field}} reads a field of a result.\n"
            "- Only refer to earlier steps. Steps that do not need each other's results run at the same time.\n"
            "- Do not calculate or invent results yourself; let the tools do it.\n"
            f"- Use at most {self.max_steps} steps.\n"
            "- If the request cannot be done with these tools, return {\"steps\": [], \"reason\": \"why\"}."
        )
        if self.instructions:
            rules += "\nExtra rules:\n" + self.instructions
        return rules

    def check(self, raw: Any) -> List[str]:
        """Problems with a plan (empty list = valid)."""
        problems = check_schema(raw, PLAN_SCHEMA)
        if problems:
            return problems
        seen: List[str] = []
        for i, s in enumerate(raw["steps"], 1):
            sid, tname, args = s.get("id"), s.get("tool"), s.get("args")
            where = f"step {i} ({sid})"
            if not isinstance(sid, str) or not _ID.match(sid) or sid in _RESERVED:
                problems.append(f"{where}: id must be short lowercase like s1")
            elif sid in seen:
                problems.append(f"{where}: id '{sid}' is used twice")
            tool = self.tools.get(tname) if isinstance(tname, str) else None
            if tool is None:
                problems.append(f"{where}: unknown tool '{tname}'. Use one of {sorted(self.tools)}")
            elif isinstance(args, dict):
                props = tool.parameters["properties"]
                for k in args:
                    if k not in props and not getattr(tool, "_accepts_any", False):
                        problems.append(f"{where}: tool '{tname}' has no argument '{k}' (it has {sorted(props)})")
                for k in tool.parameters["required"]:
                    if k not in args:
                        problems.append(f"{where}: tool '{tname}' needs argument '{k}'")
            for ref in PlanStep(str(sid), str(tname), args if isinstance(args, dict) else {}).refs():
                if ref not in seen:
                    problems.append(f"{where}: '{{{{{ref}}}}}' must be the id of an earlier step")
            if isinstance(sid, str):
                seen.append(sid)
        if len(raw["steps"]) > self.max_steps:
            problems.append(f"the plan has {len(raw['steps'])} steps; the limit is {self.max_steps}")
        answer = raw.get("answer")
        if answer and answer not in seen:
            problems.append(f"answer '{answer}' is not a step id")
        return problems

    def plan(self, request: str) -> AutoPlan:
        """Ask the model for a plan, check it (letting the model fix mistakes), and return it.

        With ``consensus``, several planners must agree; with ``escalate``, a
        stronger model takes over when the plan can't be made to pass its checks.
        """
        if not isinstance(request, str) or not request.strip():
            raise ValueError("Tell AutoFlow what to do, e.g. auto.plan('add 2 and 3')")
        if self.consensus:
            return self._consensus(request)
        try:
            return self._plan_with(request, self.llm)
        except Unclear:
            raise
        except TaskPilotError:
            if self.escalate is None:
                raise
            return self._plan_with(request, self.escalate)

    def _consensus(self, request: str) -> AutoPlan:
        planners = list(self.consensus) if isinstance(self.consensus, (list, tuple)) else [self.llm] * int(self.consensus)
        from concurrent.futures import ThreadPoolExecutor

        def attempt(planner: LLM) -> Any:
            try:
                return self._plan_with(request, planner)
            except (Unclear, TaskPilotError) as exc:
                return exc

        with ThreadPoolExecutor(len(planners)) as pool:
            outcomes = list(pool.map(attempt, planners))
        for o in outcomes:
            if isinstance(o, Unclear):
                raise o
        groups: Dict[str, List[AutoPlan]] = {}
        for o in outcomes:
            if isinstance(o, AutoPlan):
                groups.setdefault(_canonical(o), []).append(o)
        need = len(planners) // 2 + 1
        best = max(groups.values(), key=len) if groups else []
        if len(best) >= need:
            return best[0]
        failed = [str(o) for o in outcomes if isinstance(o, Exception)]
        raise Disagreement(len(groups), failed)

    def _plan_with(self, request: str, planner: LLM) -> AutoPlan:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system()},
                                          {"role": "user", "content": request}]
        problems: List[str] = []
        for _ in range(self.repair + 1):
            try:
                raw = planner.json(messages, schema=PLAN_SCHEMA)
            except LLMError as exc:
                problems = [str(exc)]
                raw = None
            if raw is not None:
                problems = self.check(raw)
                if not problems:
                    if not raw["steps"]:
                        raise TaskPilotError(f"Can't plan this with the available tools: {raw.get('reason', 'no reason given')}")
                    plan = self._make(request, raw)
                    if not self.review:
                        return plan
                    problems = self.review_plan(plan)
                    if not problems:
                        return plan
            messages = messages + [
                {"role": "assistant", "content": json.dumps(raw) if raw is not None else ""},
                {"role": "user", "content": "That plan has problems:\n- " + "\n- ".join(problems[:10]) + "\nReturn a corrected plan."},
            ]
        raise TaskPilotError("Could not make a plan that passes its checks: " + "; ".join(problems[:5]))

    def review_plan(self, plan: AutoPlan) -> List[str]:
        """Problems a second model finds in ``plan`` (empty = looks right).

        Raises :class:`Unclear` when the request itself is ambiguous.
        """
        known = _numbers_in_text(plan.request)
        odd = sorted({n for s in plan.steps for n in _numbers_in_args(s.args)
                      if not any(abs(n - k) < 1e-9 for k in known)})
        hints = ""
        if odd:
            shown = ", ".join(f"{n:g}" for n in odd[:10])
            hints = f"Numbers in the plan that are not written in the request (check each is correctly derived): {shown}\n\n"
        tools = "\n".join(f"- {n}: {t.description}" for n, t in self.tools.items())
        prompt = REVIEW_PROMPT.format(request=plan.request, tools=tools, plan=plan.describe(), hints=hints)
        reviewers = _reviewers(self.review, self.llm)

        def ask(reviewer: LLM) -> Dict[str, Any]:
            try:
                return reviewer.json(prompt, schema=REVIEW_SCHEMA)
            except Exception as exc:  # a reviewer that fails never counts as approval
                return {"ok": False, "problems": [f"the plan could not be reviewed ({type(exc).__name__}: {exc})"]}

        if len(reviewers) == 1:
            verdicts = [ask(reviewers[0])]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(len(reviewers)) as pool:
                verdicts = list(pool.map(ask, reviewers))
        for res in verdicts:
            if res.get("ambiguous") and res.get("question"):
                raise Unclear(str(res["question"]), [str(p) for p in res.get("problems") or []])
        problems: List[str] = []
        for res in verdicts:
            if not res.get("ok"):
                problems.extend(str(p) for p in res.get("problems") or ["a reviewer did not accept the plan"])
        return problems

    def _make(self, request: str, raw: Dict[str, Any]) -> AutoPlan:
        steps = [PlanStep(s["id"], s["tool"], s["args"], s.get("note", "")) for s in raw["steps"]]
        return AutoPlan(request, steps, raw.get("answer"), self.tools, self.approve, self.retry, self.name)

    def load(self, source: Union[str, Dict[str, Any]]) -> AutoPlan:
        """Load a saved plan (path or dict) and check it against today's tools. No model call."""
        if isinstance(source, dict):
            raw = source
        else:
            with open(os.fspath(source), "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        problems = self.check(raw)
        if problems:
            raise FlowDefinitionError("Saved plan is not valid any more: " + "; ".join(problems[:5]))
        return self._make(raw.get("request", ""), raw)

    # ------------------------------------------------------------ running
    def run(self, request: str, *, confirm: Optional[Callable[[AutoPlan], bool]] = None, **run_options: Any) -> AutoResult:
        """Plan and run in one go.

        ``confirm(plan) -> bool`` is called before anything runs (show the plan
        to the user here); returning ``False`` stops without running.
        """
        plan = self.plan(request)
        if confirm is not None and not confirm(plan):
            return AutoResult(plan, None)
        return AutoResult(plan, plan.run(**run_options))

    def tool_list(self) -> str:
        """The tools in plain words (for help screens)."""
        return "\n".join(f"- {n}: {t.description}" for n, t in self.tools.items())

    def __repr__(self) -> str:
        return f"AutoFlow(tools={sorted(self.tools)}, approve={sorted(self.approve)})"
