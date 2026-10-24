"""A tool-using agent: the model thinks, calls your functions, and answers."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from ..errors import ToolError
from .base import LLM, ChatResponse, ToolCall
from .tools import Tool, to_text

EMPTY_NUDGE = "Please give your answer."
FIX_NUDGE = ("Your answer was checked and has problems: {problems}\n"
             "Answer again. Use only facts from the task and the tool results (call tools if you need more facts). "
             "If something is not known, say clearly that it is not available instead of guessing.")
VERIFY_PROMPT = (
    "You check an AI assistant's answer for mistakes and made-up facts.\n\n"
    "Task:\n{task}\n\nEvidence (tool calls and their results):\n{evidence}\n\nAnswer to check:\n{answer}\n\n"
    "Rules: every specific claim in the answer (names, numbers, amounts, dates, times, statuses, emails, ids) must be "
    "stated in the task or the evidence. Saying that something is unknown or not available is fine. "
    "The answer must also actually answer the task. Return JSON: {{\"supported\": true|false, \"problems\": [\"...\"]}}."
)
VERIFY_SCHEMA = {"type": "object", "required": ["supported", "problems"],
                 "properties": {"supported": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}}}}
FINAL_NUDGE = "You have used the maximum number of tool calls. Do not call tools again; give your best final answer now."


class AgentStep:
    """One tool call the agent made."""

    __slots__ = ("call", "output", "duration")

    def __init__(self, call: ToolCall, output: str, duration: float):
        self.call = call
        self.output = output
        self.duration = duration

    def __repr__(self) -> str:
        return f"AgentStep({self.call.name}({self.call.arguments}) -> {self.output[:40]!r})"


class AgentResult:
    """What :meth:`Agent.run` returns. ``str(result)`` is the answer."""

    def __init__(self, output: str, messages: List[Dict[str, Any]], steps: List[AgentStep], turns: int, usage: Dict[str, int], stopped: bool,
                 verified: Optional[bool] = None, problems: Optional[List[str]] = None, fixes: int = 0):
        #: True = passed check/verify, False = still failing after fixes, None = not checked
        self.verified = verified
        #: what the checks found, in order (empty when everything passed first time)
        self.problems = problems or []
        #: how many times the agent was asked to fix its answer
        self.fixes = fixes
        #: the task was handed to the stronger ``escalate`` model
        self.escalated = False
        self.output = output
        self.messages = messages
        self.steps = steps
        self.turns = turns
        self.usage = usage
        #: True if ``max_steps`` was reached before the model finished on its own.
        self.stopped = stopped

    @property
    def tools_used(self) -> List[str]:
        return [s.call.name for s in self.steps]

    def __str__(self) -> str:
        return self.output

    def __repr__(self) -> str:
        flag = "" if self.verified is None else f", verified={self.verified}"
        return f"AgentResult({self.output[:60]!r}, tools={self.tools_used}, turns={self.turns}{flag})"


class Agent:
    """A model plus tools, running the *think → call tools → answer* loop.

    ::

        @tool
        def add(a: int, b: int) -> int:
            "Add two numbers."
            return a + b

        agent = Agent(Ollama("gpt-oss:120b"), tools=[add], system="Be brief.")
        agent("What is 2+40?")        # '42'

    Args:
        llm: any :class:`~taskpilot.LLM`.
        tools: functions or :class:`Tool` objects.
        system: system prompt.
        max_steps: model turns allowed before it must answer.
        memory: ``True`` keeps the conversation between ``run`` calls (oldest
            turns are dropped after ``max_history``); a
            :class:`~taskpilot.memory.ConversationMemory` summarises old turns instead.
        max_history: messages kept with ``memory=True``.
        parallel_tools: run several tool calls from one turn at the same time.
        on_tool: ``fn(AgentStep)`` called after each tool call.
        check: ``fn(answer) -> True | False | "what is wrong"``; a rejected answer
            is sent back to the model to fix.
        verify: ``True``, an :class:`LLM`, or a list of LLMs (all must agree): a
            checker confirms every fact in the answer comes from the task or the
            tool results; unsupported answers are sent back to fix. A different
            model than ``llm`` catches more.
        max_fixes: how many times to ask for a fix before giving up
            (``result.verified`` is then ``False``).
        strict: raise :class:`~taskpilot.errors.VerificationError` instead of
            returning an answer that failed its checks.
        max_tool_chars: longest tool result shown to the model (longer ones are cut).
        escalate: a stronger :class:`LLM`. If the answer still fails its checks
            after ``max_fixes``, the whole task is retried once with this model
            (``result.escalated`` is then ``True``).
    """

    def __init__(
        self,
        llm: LLM,
        tools: Iterable[Union[Tool, Callable]] = (),
        *,
        system: Optional[str] = None,
        name: str = "agent",
        max_steps: int = 10,
        memory: Any = False,
        max_history: int = 50,
        parallel_tools: bool = True,
        on_tool: Optional[Callable[[AgentStep], Any]] = None,
        check: Optional[Callable[[str], Any]] = None,
        verify: Any = False,
        max_fixes: int = 2,
        strict: bool = False,
        max_tool_chars: int = 6000,
        escalate: Optional[LLM] = None,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm = llm
        self.name = name
        self.system = system
        self.max_steps = max_steps
        self.memory = memory
        self.max_history = max(2, max_history)
        self.parallel_tools = parallel_tools
        self.on_tool = on_tool
        self.check = check
        self.verify = verify
        self.max_fixes = max(0, max_fixes)
        self.strict = strict
        self.max_tool_chars = max(200, max_tool_chars)
        self.escalate = escalate
        self.tools: Dict[str, Tool] = {}
        for t in tools:
            t = t if isinstance(t, Tool) else Tool(t)
            if t.name in self.tools:
                raise ToolError(f"Two tools are named '{t.name}'")
            self.tools[t.name] = t
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    # -------------------------------------------------------------- run
    def run(self, task: str, **chat_options: Any) -> AgentResult:
        """Solve ``task`` and return an :class:`AgentResult`."""
        if self.escalate is None:
            return self._run_once(task, chat_options)
        result = self._run_once(task, chat_options, strict=False)
        if result.verified is not False:
            return self._strict(result)
        better = self._escalated()._run_once(task, chat_options, strict=False)
        better.escalated = True
        better.problems = result.problems + better.problems
        return self._strict(better)

    def _escalated(self) -> "Agent":
        twin = Agent(self.escalate, tools=list(self.tools.values()), system=self.system, name=self.name,  # type: ignore[arg-type]
                     max_steps=self.max_steps, parallel_tools=self.parallel_tools, on_tool=self.on_tool,
                     check=self.check, verify=self.verify, max_fixes=self.max_fixes, max_tool_chars=self.max_tool_chars)
        return twin

    def _strict(self, result: "AgentResult") -> "AgentResult":
        if result.verified is False and self.strict:
            from ..errors import VerificationError

            raise VerificationError(result.output, result.problems)
        return result

    def _run_once(self, task: str, chat_options: Dict[str, Any], strict: Optional[bool] = None) -> AgentResult:
        messages = self._start(task)
        first = len(messages) - 1
        specs = [t.spec() for t in self.tools.values()] or None
        steps: List[AgentStep] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        problems: List[str] = []
        for turn in range(1, self.max_steps + 1):
            resp = self.llm.chat(messages, tools=specs, options=chat_options or None)
            if not resp.tool_calls and not resp.content.strip():
                # Reasoning models sometimes end a turn with no text; ask once more.
                _add_usage(usage, resp)
                resp = self.llm.chat(messages + [{"role": "user", "content": EMPTY_NUDGE}], tools=specs, options=chat_options or None)
            _add_usage(usage, resp)
            messages.append(resp.message)
            if not resp.tool_calls:
                problem = self._review(task, steps, resp.content)
                if problem is None or len(problems) >= self.max_fixes:
                    if problem is not None:
                        problems.append(problem)
                    return self._done(resp.content, messages, steps, turn, usage, False, first, problem, problems, strict)
                problems.append(problem)
                messages.append({"role": "user", "content": FIX_NUDGE.format(problems=problem)})
                continue
            for step in self._run_tools(resp.tool_calls):
                steps.append(step)
                messages.append(self.llm.tool_message(step.call, step.output))
        resp = self.llm.chat(messages + [{"role": "user", "content": FINAL_NUDGE}], options=chat_options or None)
        _add_usage(usage, resp)
        messages.append(resp.message)
        problem = self._review(task, steps, resp.content)
        if problem is not None:
            problems.append(problem)
        return self._done(resp.content, messages, steps, self.max_steps + 1, usage, True, first, problem, problems, strict)

    async def arun(self, task: str, **chat_options: Any) -> AgentResult:
        """Async version of :meth:`run`."""
        if self.escalate is None:
            return await self._arun_once(task, chat_options)
        result = await self._arun_once(task, chat_options, strict=False)
        if result.verified is not False:
            return self._strict(result)
        better = await self._escalated()._arun_once(task, chat_options, strict=False)
        better.escalated = True
        better.problems = result.problems + better.problems
        return self._strict(better)

    async def _arun_once(self, task: str, chat_options: Dict[str, Any], strict: Optional[bool] = None) -> AgentResult:
        messages = self._start(task)
        first = len(messages) - 1
        specs = [t.spec() for t in self.tools.values()] or None
        steps: List[AgentStep] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        problems: List[str] = []
        loop = asyncio.get_running_loop()
        for turn in range(1, self.max_steps + 1):
            resp = await self.llm.achat(messages, tools=specs, options=chat_options or None)
            if not resp.tool_calls and not resp.content.strip():
                _add_usage(usage, resp)
                resp = await self.llm.achat(messages + [{"role": "user", "content": EMPTY_NUDGE}], tools=specs, options=chat_options or None)
            _add_usage(usage, resp)
            messages.append(resp.message)
            if not resp.tool_calls:
                problem = await loop.run_in_executor(None, self._review, task, steps, resp.content)
                if problem is None or len(problems) >= self.max_fixes:
                    if problem is not None:
                        problems.append(problem)
                    return self._done(resp.content, messages, steps, turn, usage, False, first, problem, problems, strict)
                problems.append(problem)
                messages.append({"role": "user", "content": FIX_NUDGE.format(problems=problem)})
                continue
            done = await asyncio.gather(*(self._arun_tool(c) for c in resp.tool_calls))
            for step in done:
                steps.append(step)
                messages.append(self.llm.tool_message(step.call, step.output))
        resp = await self.llm.achat(messages + [{"role": "user", "content": FINAL_NUDGE}], options=chat_options or None)
        _add_usage(usage, resp)
        messages.append(resp.message)
        problem = await loop.run_in_executor(None, self._review, task, steps, resp.content)
        if problem is not None:
            problems.append(problem)
        return self._done(resp.content, messages, steps, self.max_steps + 1, usage, True, first, problem, problems, strict)

    def __call__(self, task: str, **chat_options: Any) -> str:
        return self.run(task, **chat_options).output

    def reset(self) -> None:
        """Forget the conversation memory."""
        with self._lock:
            self.history = []

    def as_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
        """Let another agent delegate work to this one (multi-agent teams)."""
        agent = self

        def delegate(task: str) -> str:
            return agent.run(task).output

        delegate.__name__ = name or self.name
        return Tool(delegate, name=name or self.name, description=description or f"Ask the '{self.name}' agent to do a task. Give it a clear, complete instruction.")

    def as_step(self, name: Optional[str] = None, prompt: str = "{task}") -> Callable:
        """A flow step that hands a task to this agent and returns its answer.

        Every ``{field}`` in ``prompt`` is a dependency (a step or an input)::

            flow.add(researcher.as_step("notes", "Research {topic} for a {audience} reader"))
        """
        import inspect

        from ..flow import _template_fields

        fields = _template_fields(prompt)
        agent = self

        def run_agent(**values: Any) -> str:
            return agent.run(prompt.format(**values)).output

        run_agent.__name__ = name or self.name
        run_agent.__doc__ = f"Agent '{self.name}': {prompt[:60]}"
        run_agent.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY) for f in fields]
        )
        return run_agent

    # ---------------------------------------------------------- helpers
    def _start(self, task: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        if _is_conversation(self.memory):
            messages.extend(self.memory.context())
        elif self.memory:
            with self._lock:
                messages.extend(self.history)
        messages.append({"role": "user", "content": task})
        return messages

    def _review(self, task: str, steps: List[AgentStep], answer: str) -> Optional[str]:
        """What is wrong with ``answer`` (``None`` = fine or not checked)."""
        if self.check is not None:
            try:
                verdict = self.check(answer)
            except Exception as exc:
                verdict = f"{type(exc).__name__}: {exc}"
            if isinstance(verdict, str) and verdict:
                return verdict
            if not verdict:
                return "the answer did not pass its check"
        if self.verify:
            if isinstance(self.verify, LLM):
                checkers = [self.verify]
            elif isinstance(self.verify, (list, tuple)):
                checkers = [c for c in self.verify if isinstance(c, LLM)] or [self.llm]
            else:
                checkers = [self.llm]
            evidence = "\n".join(
                f"- {s.call.name}({json.dumps(s.call.arguments, ensure_ascii=False, default=str)}) -> {s.output[:1500]}"
                for s in steps) or "(no tools were used)"
            prompt = VERIFY_PROMPT.format(task=task, evidence=evidence, answer=answer)
            found: List[str] = []
            for checker in checkers:  # every checker must agree
                try:
                    res = checker.json(prompt, schema=VERIFY_SCHEMA)
                except Exception as exc:  # the checker itself failed: report it, never pass silently
                    found.append(f"could not verify the answer ({type(exc).__name__})")
                    continue
                if not res.get("supported"):
                    found.extend(str(p) for p in res.get("problems") or ["claims not supported by the evidence"])
            if found:
                return "; ".join(found)
        return None

    def _done(self, output: str, messages: List[Dict[str, Any]], steps: List[AgentStep], turns: int, usage: Dict[str, int],
              stopped: bool, first: int = 0, problem: Optional[str] = None, problems: Optional[List[str]] = None,
              strict: Optional[bool] = None) -> AgentResult:
        if _is_conversation(self.memory):
            self.memory.add(messages[first:])
        elif self.memory:
            convo = [m for m in messages if m.get("role") != "system"]
            with self._lock:
                self.history = _trim(convo, self.max_history)
        checked = self.check is not None or bool(self.verify)
        verified = (problem is None) if checked else None
        if verified is False and (self.strict if strict is None else strict):
            from ..errors import VerificationError

            raise VerificationError(output, problems or [problem or ""])
        fixes = sum(1 for m in messages[first:] if m.get("role") == "user" and str(m.get("content", "")).startswith(FIX_NUDGE[:30]))
        return AgentResult(output, messages, steps, turns, usage, stopped, verified, problems, fixes)

    def _call_tool(self, call: ToolCall) -> AgentStep:
        t0 = time.perf_counter()
        t = self.tools.get(call.name)
        if t is None:
            known = ", ".join(self.tools) or "none"
            out = f"Error: there is no tool named '{call.name}'. Available tools: {known}."
        else:
            out = t.invoke(call.arguments)
        out = self._cap(out)
        step = AgentStep(call, out, time.perf_counter() - t0)
        if self.on_tool is not None:
            try:
                self.on_tool(step)
            except Exception:
                pass
        return step

    async def _arun_tool(self, call: ToolCall) -> AgentStep:
        t = self.tools.get(call.name)
        if t is not None and t.is_async:
            t0 = time.perf_counter()
            try:
                out = to_text(await t.fn(**t.prepare(call.arguments)))
            except Exception as exc:
                out = f"Error: {type(exc).__name__}: {exc}"
            step = AgentStep(call, self._cap(out), time.perf_counter() - t0)
            if self.on_tool is not None:
                try:
                    self.on_tool(step)
                except Exception:
                    pass
            return step
        return await asyncio.get_running_loop().run_in_executor(None, self._call_tool, call)

    def _cap(self, text: str) -> str:
        if len(text) <= self.max_tool_chars:
            return text
        return text[: self.max_tool_chars] + f"\n[cut: the full result was {len(text)} characters]"

    def _run_tools(self, calls: List[ToolCall]) -> List[AgentStep]:
        if len(calls) == 1 or not self.parallel_tools:
            return [self._call_tool(c) for c in calls]
        with ThreadPoolExecutor(min(8, len(calls)), thread_name_prefix="taskpilot-tool") as pool:
            return list(pool.map(self._call_tool, calls))

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, llm={self.llm!r}, tools={list(self.tools)})"


def _is_conversation(memory: Any) -> bool:
    from ..memory import ConversationMemory

    return isinstance(memory, ConversationMemory)


def _add_usage(usage: Dict[str, int], resp: ChatResponse) -> None:
    for k in usage:
        usage[k] += int(resp.usage.get(k, 0) or 0)


def _trim(messages: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Keep the newest ``limit`` messages, starting at a user turn.

    Starting at a user message guarantees we never keep a tool result whose
    tool request was cut off – providers reject such histories.
    """
    if len(messages) <= limit:
        return messages
    kept = messages[-limit:]
    for i, m in enumerate(kept):
        if m.get("role") == "user":
            return kept[i:]
    return kept[-1:] if kept and kept[-1].get("role") == "user" else []
