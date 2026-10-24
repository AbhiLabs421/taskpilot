"""Ready-made workflow patterns, named after what they do.

Function patterns return a normal :class:`~taskpilot.Flow`, so everything
works on them: ``run``, ``arun``, ``stream``, ``trace``, ``as_step``,
checkpoints, retries::

    Sequential(load, clean, save)                         # a -> b -> c
    Parallel(summary, keywords, sentiment)                # all at once, same inputs
    Router(classify, {"math": solve, "chat": talk})      # pick one branch
    MapReduce(score_one, average)                        # each item in parallel, then combine
    ReviewLoop(write, critique, until=lambda r: r["ok"])  # improve until good
    Vote(answer, n=5)                                     # ask n times, keep the majority

Agent patterns coordinate several :class:`~taskpilot.Agent` s::

    Supervisor(llm, [researcher, writer])   # a boss agent delegates to workers
    Handoff([triage, billing, tech])        # agents pass the conversation along
    GroupChat([pm, dev, qa], rounds=6)      # agents talk in turns on one transcript
"""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from .errors import FlowDefinitionError
from .flow import Flow
from .llm.agent import Agent, AgentStep
from .llm.base import LLM, to_messages
from .llm.tools import Tool
from .step import Step, callable_name, read_params

__all__ = [
    "Sequential", "Parallel", "Router", "MapReduce", "ReviewLoop", "Vote",
    "Supervisor", "Handoff", "HandoffResult", "GroupChat", "GroupChatResult",
]


def _fn(f: Any) -> Callable:
    return f.fn if isinstance(f, Step) else f


def _name(f: Any, fallback: str) -> str:
    name = callable_name(_fn(f))
    return name if name.isidentifier() and name != "step" else fallback


def _signature(names: Sequence[str], defaults: Optional[Dict[str, Any]] = None) -> inspect.Signature:
    defaults = defaults or {}
    return inspect.Signature([
        inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=defaults.get(n, inspect.Parameter.empty))
        for n in names
    ])


# ====================================================================== flows
def Sequential(*steps: Callable, name: str = "sequential", **flow_options: Any) -> Flow:
    """Run functions one after another; each gets the previous result."""
    return Flow.chain(*steps, name=name, **flow_options)


def Parallel(*steps: Callable, combine: Optional[Callable[[Dict[str, Any]], Any]] = None,
             name: str = "parallel", **flow_options: Any) -> Flow:
    """Run every function at the same time with the same inputs.

    ``output`` is ``{function_name: result}``, or ``combine(that_dict)``.
    """
    if not steps:
        raise FlowDefinitionError("Parallel() needs at least one function")
    flow = Flow(name, **flow_options)
    names = []
    for i, f in enumerate(steps):
        n = _name(f, f"branch_{i + 1}")
        if n in names:
            n = f"{n}_{i + 1}"
        flow.add(_fn(f), name=n)
        names.append(n)
    if combine is not None:
        def combined(**results: Any) -> Any:
            return combine(results)

        combined.__signature__ = _signature(names)  # type: ignore[attr-defined]
        flow.add(combined, name="combined")
    return flow


def Router(classify: Callable, routes: Dict[str, Callable], *, default: Optional[Callable] = None,
           name: str = "router", **flow_options: Any) -> Flow:
    """Send the inputs to exactly one branch, chosen by ``classify``.

    ``classify`` returns a key of ``routes``; unknown keys go to ``default``
    (or raise). ``output`` is the chosen branch's result; ``result["route"]``
    says which branch ran.
    """
    if not routes:
        raise FlowDefinitionError("Router() needs at least one route")
    for key in routes:
        if not isinstance(key, str) or not key.isidentifier():
            raise FlowDefinitionError(f"Route names must be identifiers, got {key!r}")
    if "route" in routes or "answer" in routes or "fallback_route" in routes:
        raise FlowDefinitionError("'route', 'answer' and 'fallback_route' are reserved names in Router()")
    known = set(routes)
    flow = Flow(name, **flow_options)
    cls = _fn(classify)

    def route(**inputs: Any) -> str:
        choice = cls(**inputs)
        if choice not in known:
            if default is None:
                raise ValueError(f"classify returned {choice!r}; expected one of {sorted(known)} (or pass default=)")
            return "fallback_route"
        return choice

    params, _ = read_params(cls, "classify")
    route.__signature__ = _signature([p.name for p in params], {p.name: p.default for p in params if p.has_default})  # type: ignore[attr-defined]
    flow.add(route, name="route")
    branches = dict(routes)
    if default is not None:
        branches["fallback_route"] = default
    def chosen(key: str) -> Callable[[str], bool]:
        return lambda route: route == key  # a closure, so 'key' is not mistaken for an input

    for key, fn in branches.items():
        flow.add(_fn(fn), name=key, when=chosen(key))

    def answer(**results: Any) -> Any:
        for value in results.values():
            if value is not _NONE:
                return value
        return None

    answer.__signature__ = _signature(list(branches), {k: _NONE for k in branches})  # type: ignore[attr-defined]
    flow.add(answer, name="answer")
    return flow


_NONE = object()


def MapReduce(mapper: Callable, reducer: Optional[Callable[[List[Any]], Any]] = None, *, items: str = "items",
              retry: Any = None, timeout: Optional[float] = None, cache: Any = None,
              max_concurrency: Optional[int] = None, name: str = "map_reduce", **flow_options: Any) -> Flow:
    """Run ``mapper`` on every item in parallel, then ``reducer`` on the list.

    Run it with ``flow.run(items=[...], ...)``; other inputs are passed to
    the mapper's other parameters. Without a reducer, ``output`` is the list.
    """
    flow = Flow(name, **flow_options)
    m = _fn(mapper)
    params, _ = read_params(m, "mapper")
    if not params:
        raise FlowDefinitionError("MapReduce mapper needs a parameter for the item")
    flow.add(m, name="mapper", dynamic=True, retry=retry, timeout=timeout, cache=cache, max_concurrency=max_concurrency)

    def mapped(ctx: Any, **kw: Any) -> List[Any]:
        return ctx.map("mapper", kw[items])

    mapped.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY), inspect.Parameter(items, inspect.Parameter.KEYWORD_ONLY)]
    )
    flow.add(mapped, name="mapped")
    if reducer is not None:
        r = _fn(reducer)
        flow.add(lambda mapped: r(mapped), name="reduced")
    return flow


def ReviewLoop(generate: Callable, review: Callable, *, until: Callable[[Any], bool], feedback: str = "feedback",
               max_iterations: int = 5, on_max: str = "stop", name: str = "review_loop", **flow_options: Any) -> Flow:
    """Evaluator-optimizer: generate, review, and regenerate with the review until good.

    * ``generate(<inputs>, feedback=None)`` – gets the last review as ``feedback``.
    * ``review(draft)`` – gets the latest draft (its first parameter).
    * ``until(review) -> bool`` – stop when true.

    ``output`` is the final draft; ``result["review"]`` the last review and
    ``result.history["draft"]`` every attempt.
    """
    g, rv = _fn(generate), _fn(review)
    g_params, _ = read_params(g, "generate")
    if feedback not in [p.name for p in g_params]:
        raise FlowDefinitionError(f"generate() needs a parameter named '{feedback}' (default None) to receive the review")
    r_params, _ = read_params(rv, "review")
    if not r_params:
        raise FlowDefinitionError("review() needs a parameter for the draft")
    flow = Flow(name, **flow_options)
    flow.add(g, name="draft")
    flow.add(rv, name="review", bind={r_params[0].name: "draft"})
    flow.add(lambda draft, review: draft, name="final")
    flow.loop("draft", "review", until=lambda review: until(review), feed={feedback: "review"},
              max_iterations=max_iterations, on_max=on_max, name="improve")
    return flow


def Vote(fn: Callable, n: int = 3, *, pick: Optional[Callable[[List[Any]], Any]] = None,
         name: str = "vote", **flow_options: Any) -> Flow:
    """Self-consistency: call ``fn`` ``n`` times in parallel and keep the most common answer.

    ``pick(answers)`` can replace the majority vote. ``result["answers"]``
    holds every answer.
    """
    if n < 1:
        raise FlowDefinitionError("Vote(n=...) must be >= 1")
    f = _fn(fn)
    params, _ = read_params(f, "fn")
    flow = Flow(name, **flow_options)
    flow.add(f, name="ask", dynamic=True)

    def answers(ctx: Any, **kw: Any) -> List[Any]:
        return ctx.map("ask", [dict(kw) for _ in range(n)])

    answers.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY)]
        + [inspect.Parameter(p.name, inspect.Parameter.KEYWORD_ONLY,
                             default=p.default if p.has_default else inspect.Parameter.empty) for p in params]
    )
    flow.add(answers, name="answers")
    chooser = pick or _majority
    flow.add(lambda answers: chooser(answers), name="winner")
    return flow


def _majority(values: List[Any]) -> Any:
    keyed = [(repr(v) if not isinstance(v, str) else v.strip().lower(), v) for v in values]
    counts = Counter(k for k, _ in keyed)
    best = max(counts.values())
    for k, v in keyed:  # earliest of the most common answers
        if counts[k] == best:
            return v
    return None


# ===================================================================== agents
class _Team:
    """Shared conveniences for multi-agent patterns."""

    name = "team"

    def run(self, task: str) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def __call__(self, task: str) -> str:
        return self.run(task).output

    async def arun(self, task: str) -> Any:
        return await asyncio.get_running_loop().run_in_executor(None, self.run, task)

    def as_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
        team = self

        def delegate(task: str) -> str:
            return team.run(task).output

        delegate.__name__ = name or self.name
        return Tool(delegate, name=name or self.name, description=description or f"Ask the '{self.name}' team to do a task.")

    def as_step(self, name: Optional[str] = None, task: str = "task") -> Callable:
        """A flow step that sends its ``task`` input to this team and returns the answer."""
        team = self

        def step(**kw: Any) -> str:
            return team.run(kw[task]).output

        step.__name__ = name or self.name
        step.__signature__ = _signature([task])  # type: ignore[attr-defined]
        return step


def Supervisor(llm: LLM, workers: Iterable[Union[Agent, _Team]], *, system: Optional[str] = None,
               max_steps: int = 12, name: str = "supervisor", **agent_options: Any) -> Agent:
    """A boss agent that plans and delegates sub-tasks to worker agents (orchestrator-workers)."""
    tools = []
    names = []
    for w in workers:
        desc = None
        if isinstance(w, Agent) and w.system:
            desc = f"Agent '{w.name}': {w.system[:200]}. Give it one clear, complete task."
        tools.append(w.as_tool(description=desc))
        names.append(w.name)
    if len(set(names)) != len(names):
        raise FlowDefinitionError(f"Worker names must be unique, got {names}")
    base = (f"You are a supervisor. Break the user's task into sub-tasks and delegate them to your workers "
            f"({', '.join(names)}) using the tools. Combine their results into one final answer.")
    return Agent(llm, tools=tools, system=f"{base}\n\n{system}" if system else base, max_steps=max_steps, name=name, **agent_options)


class HandoffResult:
    """What :class:`Handoff` returns. ``str(result)`` is the answer."""

    def __init__(self, output: str, agent: str, path: List[str], steps: List[AgentStep], messages: List[Dict[str, Any]]):
        self.output = output
        #: the agent that gave the final answer
        self.agent = agent
        #: every agent that handled the conversation, in order
        self.path = path
        self.steps = steps
        self.messages = messages

    def __str__(self) -> str:
        return self.output

    def __repr__(self) -> str:
        return f"HandoffResult({self.output[:50]!r}, path={self.path})"


class Handoff(_Team):
    """Agents that pass the conversation to each other (triage -> specialist).

    Every agent gets ``transfer_to_<name>`` tools for the others. The agent
    holding the conversation answers or transfers; tools of the current agent
    work as usual.

    ::

        support = Handoff([triage, billing, tech], start="triage")
        support("I was charged twice")       # triage -> billing -> answer
    """

    def __init__(self, agents: Sequence[Agent], *, start: Optional[str] = None, max_handoffs: int = 5,
                 max_steps: int = 10, name: str = "handoff"):
        if len(agents) < 2:
            raise FlowDefinitionError("Handoff needs at least two agents")
        self.agents: Dict[str, Agent] = {}
        for a in agents:
            if a.name in self.agents:
                raise FlowDefinitionError(f"Two agents are named '{a.name}'")
            self.agents[a.name] = a
        self.start = start or agents[0].name
        if self.start not in self.agents:
            raise FlowDefinitionError(f"start='{self.start}' is not one of {list(self.agents)}")
        self.max_handoffs = max_handoffs
        self.max_steps = max_steps
        self.name = name

    def _transfer_specs(self, current: str) -> List[dict]:
        specs = []
        for other, agent in self.agents.items():
            if other == current:
                continue
            about = (agent.system or other)[:200]
            specs.append({"type": "function", "function": {
                "name": f"transfer_to_{other}",
                "description": f"Hand the conversation to '{other}' ({about}). Use when they are better suited.",
                "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "why"}}, "required": []},
            }})
        return specs

    def run(self, task: str) -> HandoffResult:
        current = self.start
        path = [current]
        steps: List[AgentStep] = []
        transcript: List[Dict[str, Any]] = [{"role": "user", "content": task}]
        notes: List[str] = []
        handoffs = 0
        while True:
            agent = self.agents[current]
            system = (agent.system or "") + (
                f"\n\nYou are '{current}' in a team: {', '.join(self.agents)}. Answer the user directly if you can, "
                f"otherwise transfer with a transfer_to_<name> tool."
            )
            if notes:
                system += "\n\nHandoff history: " + " ".join(notes) + " You now own the conversation; answer the user."
            # The model sees the user's words last; handoff notes live in the system prompt.
            messages = to_messages([m for m in transcript if not m.get("handoff")], system.strip())
            tools = [t.spec() for t in agent.tools.values()]
            if handoffs < self.max_handoffs:
                tools += self._transfer_specs(current)
            moved = None
            for _ in range(self.max_steps):
                resp = agent.llm.chat(messages, tools=tools or None)
                if not resp.tool_calls and not resp.content.strip():
                    resp = agent.llm.chat(messages + [{"role": "user", "content": "Please answer now."}], tools=tools or None)
                messages.append(resp.message)
                if not resp.tool_calls:
                    transcript.append({"role": "assistant", "content": resp.content})
                    return HandoffResult(resp.content, current, path, steps, transcript)
                for call in resp.tool_calls:
                    if call.name.startswith("transfer_to_") and call.name[12:] in self.agents and moved is None:
                        moved = call.name[12:]
                        reason = call.arguments.get("reason", "") if isinstance(call.arguments, dict) else ""
                        messages.append(agent.llm.tool_message(call, f"Transferred to {moved}."))
                        note = f"[{current} handed the conversation to {moved}{': ' + reason if reason else ''}]"
                        notes.append(note)
                        transcript.append({"role": "assistant", "content": note, "handoff": True})
                    elif call.name.startswith("transfer_to_"):
                        messages.append(agent.llm.tool_message(call, "Only one transfer per turn."))
                    else:
                        step = agent._call_tool(call)
                        steps.append(step)
                        messages.append(agent.llm.tool_message(call, step.output))
                if moved is not None:
                    break
            if moved is None:  # out of steps: ask for a final answer
                resp = agent.llm.chat(messages + [{"role": "user", "content": "Give your final answer now."}])
                transcript.append({"role": "assistant", "content": resp.content})
                return HandoffResult(resp.content, current, path, steps, transcript)
            current = moved
            path.append(current)
            handoffs += 1


class GroupChatResult:
    """What :class:`GroupChat` returns: the transcript and the last word."""

    def __init__(self, messages: List[Dict[str, str]], stopped_by: str):
        #: ``[{"speaker": name, "content": text}, ...]``
        self.messages = messages
        #: ``"done"`` (a speaker said the stop word), ``"until"`` or ``"rounds"``
        self.stopped_by = stopped_by

    @property
    def output(self) -> str:
        return self.messages[-1]["content"] if self.messages else ""

    @property
    def speakers(self) -> List[str]:
        return [m["speaker"] for m in self.messages]

    def __str__(self) -> str:
        return self.output

    def __repr__(self) -> str:
        return f"GroupChatResult({len(self.messages)} messages, stopped_by={self.stopped_by!r})"


class GroupChat(_Team):
    """Several agents discuss a task in turns on one shared transcript.

    Args:
        agents: the participants.
        rounds: maximum number of messages.
        speaker: ``"round_robin"`` (default), an :class:`LLM` that picks who
            speaks next, or ``fn(messages, names) -> name``.
        until: ``fn(messages) -> bool`` to stop early.
        stop_word: a speaker ending with this word (default ``"DONE"``) ends the chat.
        finishers: names of the agents allowed to end the chat (default: anyone).
            The others are told not to use the stop word, and it is ignored from them.
    """

    def __init__(self, agents: Sequence[Agent], *, rounds: int = 6, speaker: Any = "round_robin",
                 until: Optional[Callable[[List[Dict[str, str]]], bool]] = None, stop_word: Optional[str] = "DONE",
                 finishers: Optional[Sequence[str]] = None, name: str = "group_chat"):
        if len(agents) < 2:
            raise FlowDefinitionError("GroupChat needs at least two agents")
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise FlowDefinitionError(f"Agent names must be unique, got {names}")
        if rounds < 1:
            raise FlowDefinitionError("rounds must be >= 1")
        if not (speaker == "round_robin" or isinstance(speaker, LLM) or callable(speaker)):
            raise FlowDefinitionError("speaker must be 'round_robin', an LLM, or a function")
        self.agents = {a.name: a for a in agents}
        unknown = [n for n in (finishers or []) if n not in self.agents]
        if unknown:
            raise FlowDefinitionError(f"finishers {unknown} are not in the chat ({', '.join(self.agents)})")
        self.finishers = set(finishers) if finishers else set(self.agents)
        self.rounds = rounds
        self.speaker = speaker
        self.until = until
        self.stop_word = stop_word
        self.name = name

    def _next(self, turn: int, messages: List[Dict[str, str]]) -> str:
        names = list(self.agents)
        if self.speaker == "round_robin":
            return names[turn % len(names)]
        if isinstance(self.speaker, LLM):
            transcript = "\n".join(f"{m['speaker']}: {m['content']}" for m in messages[-12:])
            choice = self.speaker.ask(
                f"Participants: {', '.join(names)}.\nConversation:\n{transcript}\n\n"
                f"Who should speak next? Reply with exactly one name from the list."
            ).strip()
            for n in names:
                if n.lower() in choice.lower():
                    return n
            return names[turn % len(names)]
        choice = self.speaker(messages, names)
        if choice not in self.agents:
            raise ValueError(f"speaker() returned {choice!r}; expected one of {names}")
        return choice

    def run(self, task: str) -> GroupChatResult:
        messages: List[Dict[str, str]] = [{"speaker": "user", "content": task}]
        for turn in range(self.rounds):
            who = self._next(turn, messages)
            transcript = "\n".join(f"{m['speaker']}: {m['content']}" for m in messages)
            if not self.stop_word:
                stop = ""
            elif who in self.finishers:
                stop = f" If the task is fully solved, end your message with the word {self.stop_word}."
            else:
                stop = f" Do not write the word {self.stop_word}; only {', '.join(sorted(self.finishers))} can end the chat."
            prompt = (f"You are '{who}' in a group conversation with {', '.join(n for n in self.agents if n != who)}."
                      f"\n\nConversation so far:\n{transcript}\n\nWrite your next message only.{stop}")
            text = self.agents[who].run(prompt).output.strip()
            messages.append({"speaker": who, "content": text})
            if (self.stop_word and who in self.finishers
                    and text.rstrip(" .!").upper().endswith(self.stop_word.upper())):
                return GroupChatResult(messages, "done")
            if self.until is not None and self.until(messages):
                return GroupChatResult(messages, "until")
        return GroupChatResult(messages, "rounds")
