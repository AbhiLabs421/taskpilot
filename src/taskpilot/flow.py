"""The :class:`Flow` – the one class you need to build a workflow."""

from __future__ import annotations

import asyncio
import atexit
import inspect
import os
import queue
import re
import string
import threading
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from .checkpoint import Checkpoint
from .engine import Loop, Plan, Runner, compile_plan
from .errors import FlowDefinitionError, FlowTimeout, did_you_mean
from .events import EVENT_TYPES, Event, EventBus, console_printer
from .result import RunResult
from .step import Step, read_params

_UNSET = object()
_local = threading.local()


def _cancel_leftovers(loop: asyncio.AbstractEventLoop) -> None:
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    if not pending:
        return
    for t in pending:
        t.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


class _LoopHolder:
    """Owns one reusable event loop per thread and closes it when the thread ends."""

    __slots__ = ("loop",)

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()

    def close(self) -> None:
        loop = self.loop
        if not loop.is_closed() and not loop.is_running():
            try:
                _cancel_leftovers(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _close_main_loop() -> None:
    holder = getattr(_local, "holder", None)
    if holder is not None:
        holder.close()


atexit.register(_close_main_loop)


def _run_in_temp_loop(make_coro: Callable[[], Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(make_coro())
    finally:
        try:
            _cancel_leftovers(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def run_sync(make_coro: Callable[[], Any], *, reuse_loop: bool = True) -> Any:
    """Run a coroutine to completion from synchronous code, anywhere.

    Reuses one event loop per thread (fast for many small runs). If called
    while an event loop is already running in this thread (Jupyter, an async
    web app), it runs on a helper thread instead of failing.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        box: Dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = _run_in_temp_loop(make_coro)
            except BaseException as exc:
                box["error"] = exc

        t = threading.Thread(target=target, name="taskpilot-sync-run", daemon=True)
        t.start()
        t.join()
        if "error" in box:
            raise box["error"]
        return box["value"]

    holder = getattr(_local, "holder", None)
    if not reuse_loop or (holder is not None and holder.loop.is_running()):
        # A helper thread, or a call from a step taskpilot runs on this thread.
        return _run_in_temp_loop(make_coro)
    if holder is None or holder.loop.is_closed():
        holder = _LoopHolder()
        _local.holder = holder
    loop = holder.loop
    try:
        return loop.run_until_complete(make_coro())
    finally:
        _cancel_leftovers(loop)


def _template_fields(template: str) -> List[str]:
    fields: List[str] = []
    for _, field, _, _ in string.Formatter().parse(template):
        if field is None:
            continue
        root = re.split(r"[.\[]", field, maxsplit=1)[0]
        if not root or root.isdigit():
            raise FlowDefinitionError(
                f"Prompt templates need named fields like '{{text}}', got '{{{field}}}'. "
                f"Write literal braces as '{{{{' and '}}}}'."
            )
        if root not in fields:
            fields.append(root)
    return fields


class Flow:
    """A workflow: steps (plain or async Python functions) wired by name.

    A step's parameters say what it needs. A parameter named like another
    step receives that step's result; any other name is a run input::

        flow = Flow("greet")

        @flow.step
        def name(raw):              # 'raw' is an input
            return raw.strip().title()

        @flow.step
        def hello(name):            # gets the result of step 'name'
            return f"Hello, {name}!"

        flow.run(raw="  asha ").output      # 'Hello, Asha!'

    Args:
        name: shown in logs, errors and traces.
        max_workers: threads used to run plain functions in parallel
            (default: ``min(32, cpu_count + 4)``).
        timeout: limit in seconds for a whole run.
        verbose: print a live trace of every run to stderr.
    """

    def __init__(
        self,
        name: str = "flow",
        *,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        verbose: bool = False,
        description: str = "",
    ):
        if max_workers is not None and max_workers < 1:
            raise FlowDefinitionError("max_workers must be >= 1")
        if timeout is not None and timeout <= 0:
            raise FlowDefinitionError("timeout must be a positive number of seconds")
        self.name = name
        self.description = description
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.timeout = timeout
        self.verbose = verbose
        self._steps: Dict[str, Step] = {}
        self._loops: Dict[str, Loop] = {}
        self._plan: Optional[Plan] = None
        self._plan_lock = threading.Lock()
        self._bus = EventBus()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
        self._pool_prefix = f"taskpilot-{name[:20]}-{id(self):x}"

    # ================================================================ build
    def step(self, fn: Union[Callable, str, None] = None, **options: Any) -> Any:
        """Add a function as a step. Use as ``@flow.step`` or ``@flow.step(...)``.

        Options (all optional):
            name: step name (default: the function name).
            retry: attempts in total, or ``Retry(...)`` for full control.
            timeout: seconds allowed per attempt.
            cache: ``True``, a TTL in seconds, or ``Cache(...)`` – reuse results
                for identical inputs.
            when: ``lambda <deps>: bool`` – run only if true, else skip.
            map: name of a parameter holding a list – run once per item, in
                parallel, and return the list of results.
            max_concurrency: limit for parallel ``map`` items.
            repeat_until: ``lambda result: bool`` – run again until true
                (``ctx.previous`` holds the last result), at most ``max_repeats``.
            fallback: value (or ``fn(error)``) to use if the step fails.
            on_error: ``"raise"`` (default) stops the run; ``"skip"`` skips the
                step; ``"continue"`` marks it failed and keeps going.
            after: steps that must finish first even though no value is used.
            bind: ``{"param": "source"}`` – feed a parameter from a
                differently named step or input.
            keep: keep the value even with ``lean=True``.
            check: ``lambda result: True/False/"reason"`` – reject wrong results; a
                rejection is a failed attempt (retried with ``retry=``).
            dynamic: not part of the graph; other steps run it at runtime
                with ``ctx.call("name", ...)`` / ``ctx.map("name", [...])``.
            tags, description: free metadata.
        """
        if isinstance(fn, str):
            options.setdefault("name", fn)
            fn = None
        if fn is None:
            def decorator(f: Callable) -> Step:
                return self.add(f, **options)

            return decorator
        return self.add(fn, **options)

    def add(self, fn: Callable, **options: Any) -> Step:
        """Add a step without the decorator syntax. Returns the :class:`Step`."""
        if isinstance(fn, Step):
            options = {**_step_options(fn), **options}
            fn = fn.fn
        step = Step(fn, **options)
        if step.name in self._steps:
            raise FlowDefinitionError(
                f"Flow '{self.name}' already has a step named '{step.name}'. "
                f"Pick another name with name='...'."
            )
        self._steps[step.name] = step
        self._plan = None
        return step

    def loop(
        self,
        start: str,
        end: str,
        *,
        until: Callable[..., Any],
        feed: Optional[Dict[str, str]] = None,
        max_iterations: int = 10,
        on_max: str = "stop",
        name: Optional[str] = None,
    ) -> Loop:
        """Run the steps from ``start`` to ``end`` again until ``until`` is true.

        ::

            @flow.step
            def draft(topic, feedback=None):        # feedback: last round's review
                ...

            @flow.step
            def review(draft):
                ...

            flow.loop("draft", "review",
                      until=lambda review: review["score"] >= 8,
                      feed={"feedback": "review"}, max_iterations=5)

        Args:
            start, end: first and last step of the loop. Every step on a path
                between them repeats; steps after the loop wait for it.
            until: checked after ``end`` finishes. Its parameters name steps
                (or inputs) like any step. ``True`` leaves the loop.
            feed: ``{"param": "step"}`` – in the next round, steps in the loop
                that have ``param`` receive that step's latest value.
            max_iterations: safety limit on rounds.
            on_max: ``"stop"`` keeps the last round's values (default);
                ``"raise"`` fails the run with :class:`LoopLimit`.

        ``result.history["step"]`` holds every round's value and
        ``ctx.iteration`` is the current round.
        """
        loop_name = name or f"{start}_loop"
        if loop_name in self._loops:
            raise FlowDefinitionError(f"Flow '{self.name}' already has a loop named '{loop_name}'.")
        lp = Loop(loop_name, start, end, until, feed or {}, max_iterations, on_max)
        self._loops[loop_name] = lp
        self._plan = None
        return lp

    def remove(self, name: str) -> Step:
        """Remove a step by name and return it."""
        if name not in self._steps:
            raise KeyError(f"No step named '{name}'.{did_you_mean(name, self._steps)}")
        self._plan = None
        return self._steps.pop(name)

    def prompt(
        self,
        name: str,
        llm: Any,
        template: str,
        *,
        system: Optional[str] = None,
        json: bool = False,
        schema: Optional[dict] = None,
        **options: Any,
    ) -> Step:
        """Add an LLM step from a prompt template.

        Every ``{field}`` in the template is a dependency (a step or an input)::

            flow.prompt("summary", llm, "Summarize in 3 bullets:\\n{article}")

        With ``json=True`` (or a JSON ``schema``) the step returns parsed JSON.
        """
        fields = _template_fields(template)

        def run_prompt(**values: Any) -> Any:
            text = template.format(**values)
            if json or schema is not None:
                return llm.json(text, schema=schema, system=system)
            return llm.ask(text, system=system)

        run_prompt.__name__ = name
        run_prompt.__doc__ = f"LLM prompt: {template[:60]}"
        run_prompt.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY) for f in fields]
        )
        return self.add(run_prompt, name=name, **options)

    def as_step(self, name: Optional[str] = None) -> Callable:
        """Use this whole flow as one step of another flow::

            outer.add(inner.as_step("enrich"))

        The inner flow's inputs become the step's parameters and its
        ``output`` becomes the step's value.
        """
        plan = self.plan()
        required = plan.required_inputs()
        params = [
            inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=inspect.Parameter.empty if n in required else _UNSET)
            for n in plan.inputs
        ]
        inner = self

        async def run_subflow(**values: Any) -> Any:
            given = {k: v for k, v in values.items() if v is not _UNSET}
            return (await inner.arun(given)).output

        run_subflow.__name__ = name or (self.name if self.name.isidentifier() else "subflow")
        run_subflow.__doc__ = f"Sub-flow '{self.name}'"
        run_subflow.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
        return run_subflow

    def as_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Any:
        """Let an :class:`~taskpilot.Agent` run this whole flow as one tool.

        The flow's inputs become the tool's arguments (type hints of the steps
        that use them become the argument types) and ``output`` is returned
        to the model. A run that pauses for a person tells the model so.
        """
        from .llm.tools import Tool

        plan = self.plan()
        required = plan.required_inputs()
        params, notes = [], {}
        for inp, users in plan.inputs.items():
            ann: Any = inspect.Parameter.empty
            for user in users:
                step = plan.steps[user]
                for p_ in inspect.signature(step.fn).parameters.values():
                    if step.source(p_.name) == inp and p_.annotation is not inspect.Parameter.empty:
                        ann = p_.annotation
                        break
                if ann is not inspect.Parameter.empty:
                    break
            default = inspect.Parameter.empty if inp in required else None
            params.append(inspect.Parameter(inp, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=ann))
            if ann is not inspect.Parameter.empty:
                notes[inp] = ann
        flow = self

        def run_flow(**values: Any) -> Any:
            given = {k: v for k, v in values.items() if v is not None or k in required}
            result = flow.run(given)
            if result.paused:
                keys = ", ".join(r.key for r in result.pending)
                return f"The workflow is paused, waiting for a person to answer: {keys}."
            return result.output

        tool_name = name or (self.name if self.name.isidentifier() else "workflow")
        run_flow.__name__ = tool_name
        run_flow.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
        run_flow.__annotations__ = dict(notes)
        about = description or self.description or f"Run the '{self.name}' workflow ({' -> '.join(plan.order)})."
        return Tool(run_flow, name=tool_name, description=about)

    @classmethod
    def chain(cls, *fns: Callable, name: str = "chain", **flow_options: Any) -> "Flow":
        """Build a straight pipeline: each function gets the previous result.

        ::

            pipeline = Flow.chain(load, clean, save)
            pipeline.run(path="data.csv")
        """
        if not fns:
            raise FlowDefinitionError("Flow.chain() needs at least one function")
        flow = cls(name, **flow_options)
        prev: Optional[str] = None
        for i, fn in enumerate(fns):
            target = fn.fn if isinstance(fn, Step) else fn
            params, _ = read_params(target, f"chain function #{i + 1}")
            step_name = getattr(target, "__name__", "") or f"step_{i + 1}"
            if not step_name.isidentifier() or step_name in flow._steps:
                step_name = f"step_{i + 1}"
            bind = {}
            if prev is not None:
                if not params:
                    raise FlowDefinitionError(
                        f"Chain function '{step_name}' takes no parameters, so it cannot receive "
                        f"the result of '{prev}'."
                    )
                bind[params[0].name] = prev
            flow.add(target, name=step_name, bind=bind)
            prev = step_name
        return flow

    def on(self, event: str, handler: Optional[Callable[[Event], Any]] = None) -> Any:
        """Listen to run events. ``"*"`` means every event.

        Built-in events: ``flow_start, flow_end, step_start, step_end,
        step_retry, step_skip, step_error, log``. Use as ``flow.on("step_end",
        fn)`` or as a decorator ``@flow.on("step_end")``.
        """
        if event != "*" and event not in EVENT_TYPES:
            hint = did_you_mean(event, EVENT_TYPES)
            if hint:
                warnings.warn(f"Unknown event '{event}'.{hint} (custom events are allowed)", stacklevel=2)
        if handler is None:
            def decorator(fn: Callable[[Event], Any]) -> Callable[[Event], Any]:
                self._bus.add(event, fn)
                return fn

            return decorator
        self._bus.add(event, handler)
        return handler

    def trace(self, tracer: Any) -> Any:
        """Record every run as spans with a tracer from :mod:`taskpilot.tracing`.

        ::

            from taskpilot.tracing import ConsoleTracer
            flow.trace(ConsoleTracer())
        """
        self._bus.add("*", tracer)
        return tracer

    def untrace(self, tracer: Any) -> None:
        """Stop sending runs to ``tracer``."""
        self._bus.remove("*", tracer)

    def off(self, event: str, handler: Callable[[Event], Any]) -> None:
        """Stop listening (the reverse of :meth:`on`)."""
        self._bus.remove(event, handler)

    # ============================================================== inspect
    def plan(self) -> Plan:
        """Validate the flow and return its compiled :class:`Plan`."""
        plan = self._plan
        if plan is None:
            with self._plan_lock:
                plan = self._plan
                if plan is None:
                    static = {n: st for n, st in self._steps.items() if not st.dynamic}
                    dynamic = {n: st for n, st in self._steps.items() if st.dynamic}
                    for n, st in static.items():
                        for p in st.params + st.when_params:
                            if st.source(p.name) in dynamic:
                                raise FlowDefinitionError(
                                    f"Step '{n}' asks for '{st.source(p.name)}', which is a dynamic step. "
                                    f"Dynamic steps only run when called: ctx.call('{st.source(p.name)}', ...)."
                                )
                    plan = compile_plan(static, self._loops)
                    plan.dynamic = dynamic
                    self._plan = plan
        return plan

    def validate(self) -> "Flow":
        """Raise a clear error now if the flow is wired wrong (unknown names, cycles)."""
        self.plan()
        return self

    @property
    def steps(self) -> Dict[str, Step]:
        return dict(self._steps)

    @property
    def inputs(self) -> List[str]:
        """Names this flow expects as run inputs."""
        return list(self.plan().inputs)

    @property
    def required_inputs(self) -> List[str]:
        return list(self.plan().required_inputs())

    def __len__(self) -> int:
        return len(self._steps)

    def __contains__(self, name: object) -> bool:
        return name in self._steps

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps.values())

    def __getitem__(self, name: str) -> Step:
        try:
            return self._steps[name]
        except KeyError:
            raise KeyError(f"No step named '{name}'.{did_you_mean(name, self._steps)}") from None

    def __repr__(self) -> str:
        return f"Flow({self.name!r}, steps={list(self._steps)})"

    def describe(self) -> str:
        """A readable outline of the flow: order, wiring and options."""
        plan = self.plan()
        lines = [f"Flow '{self.name}' ({len(plan.order)} steps)"]
        if self.description:
            lines.append(f"  {self.description}")
        fed = {p for lp in plan.loops.values() for p in lp.feed}
        shown = [n for n in plan.inputs if n not in fed]
        if shown:
            req = plan.required_inputs()
            ins = ", ".join(n if n in req else f"{n}?" for n in shown)
            lines.append(f"  inputs: {ins}")
        width = max([len(n) for n in plan.order] + [1])
        for i, n in enumerate(plan.order, 1):
            step = plan.steps[n]
            src = [s for s in plan.upstream[n]] + [
                step.source(p.name) for p in step.params if step.source(p.name) in plan.inputs
            ]
            opts = step.options()
            opt_text = ("  [" + ", ".join(k if v is True else f"{k}={v}" for k, v in opts.items()) + "]") if opts else ""
            arrow = f" <- {', '.join(src)}" if src else ""
            lines.append(f"  {i:>2}. {n:<{width}}{arrow}{opt_text}")
        for lp in plan.loops.values():
            feed = f", feed {lp.feed}" if lp.feed else ""
            lines.append(f"  loop {lp.name}: {' -> '.join(lp.body)} (until=..., max {lp.max_iterations}{feed})")
        lines.append(f"  output: {', '.join(plan.sinks) or '-'}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """The flow as a Mermaid diagram (paste into GitHub markdown or mermaid.live)."""
        plan = self.plan()
        out = ["flowchart TD"]
        fed = {p for lp in plan.loops.values() for p in lp.feed}  # these come from the loop, not the user
        for name in plan.inputs:
            if name not in fed:
                out.append(f"    in_{name}([{name}])")
        for n in plan.order:
            step = plan.steps[n]
            label = n
            opts = step.options()
            if opts:
                label += "<br/><small>" + ", ".join(opts) + "</small>"
            shape = f"{{{{{label}}}}}" if step.when is not None else f"[\"{label}\"]"
            out.append(f"    {n}{shape}")
        for n in plan.order:
            step = plan.steps[n]
            for p in step.params + step.when_params:
                src = step.source(p.name)
                if src in plan.inputs and src not in fed:
                    out.append(f"    in_{src} --> {n}")
            for up in plan.deps[n]:
                out.append(f"    {up} --> {n}")
            for up in plan.upstream[n]:
                if up not in plan.deps[n]:
                    out.append(f"    {up} -.-> {n}")
        for lp in plan.loops.values():
            out.append(f"    {lp.end} -. \"repeat until\" .-> {lp.start}")
        # Remove duplicate edges while keeping order.
        seen = set()
        uniq = []
        for line in out:
            if line not in seen:
                seen.add(line)
                uniq.append(line)
        return "\n".join(uniq)

    # ================================================================== run
    def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        *,
        checkpoint: Any = None,
        lean: bool = False,
        raise_errors: bool = True,
        timeout: Optional[float] = None,
        state: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
        **kwinputs: Any,
    ) -> RunResult:
        """Run the flow and return a :class:`~taskpilot.RunResult`.

        Inputs can be given as keywords (``flow.run(x=1)``) or as a dict
        (``flow.run({"x": 1})``) – use the dict if an input is named like an
        option below.

        Args:
            checkpoint: file path (or :class:`Checkpoint`) to resume from after
                a crash.
            lean: free intermediate values as soon as no step needs them.
            raise_errors: ``False`` returns a failed result instead of raising.
            timeout: limit for this run (overrides the flow's timeout).
            state: initial shared state dict (``ctx.state``).
            verbose: print a live trace for this run.
        """
        return run_sync(lambda: self._arun(
            _merge(inputs, kwinputs), checkpoint, lean, raise_errors, timeout, state, verbose, None, owns_loop=True,
        ))

    async def arun(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        *,
        checkpoint: Any = None,
        lean: bool = False,
        raise_errors: bool = True,
        timeout: Optional[float] = None,
        state: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
        **kwinputs: Any,
    ) -> RunResult:
        """Async version of :meth:`run` – ``await flow.arun(x=1)``."""
        return await self._arun(
            _merge(inputs, kwinputs), checkpoint, lean, raise_errors, timeout, state, verbose, None, owns_loop=False,
        )

    def resume(
        self,
        paused: Any,
        answers: Optional[Dict[str, Any]] = None,
        *,
        checkpoint: Any = None,
        lean: bool = False,
        raise_errors: bool = True,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
        **kwanswers: Any,
    ) -> RunResult:
        """Continue a paused run with answers to its pending questions.

        ``paused`` is the paused :class:`RunResult` or the path it was saved
        to with ``result.save(path)``. Answers are given as keywords
        (``approve=True``) or as a dict. Finished steps are reused, the waiting
        step runs again, and ``ctx.wait_for`` now returns the answer.
        """
        run = self._prepare_resume(paused, _merge(answers, kwanswers))
        return run_sync(lambda: self._arun(
            run.inputs, checkpoint, lean, raise_errors, timeout, run.state, verbose, None,
            owns_loop=True, answers=run.answers, restored=run.restored,
        ))

    async def aresume(
        self,
        paused: Any,
        answers: Optional[Dict[str, Any]] = None,
        *,
        checkpoint: Any = None,
        lean: bool = False,
        raise_errors: bool = True,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
        **kwanswers: Any,
    ) -> RunResult:
        """Async version of :meth:`resume`."""
        run = self._prepare_resume(paused, _merge(answers, kwanswers))
        return await self._arun(
            run.inputs, checkpoint, lean, raise_errors, timeout, run.state, verbose, None,
            owns_loop=False, answers=run.answers, restored=run.restored,
        )

    def _prepare_resume(self, paused: Any, new_answers: Dict[str, Any]) -> Any:
        from .human import PausedRun

        if isinstance(paused, RunResult):
            run = PausedRun.from_result(paused)
        elif isinstance(paused, PausedRun):
            run = paused
        elif isinstance(paused, (str, os.PathLike)):
            run = PausedRun.load(paused)
        else:
            raise TypeError(f"resume() needs a paused RunResult or a file path, got {type(paused).__name__}")
        if run.flow != self.name:
            raise FlowDefinitionError(f"That run belongs to flow '{run.flow}', not '{self.name}'.")
        waiting = {r.key for r in run.pending}
        for key in new_answers:
            if key not in waiting and key not in run.answers:
                hint = did_you_mean(key, waiting)
                pending = ", ".join(sorted(waiting)) or "none"
                raise FlowDefinitionError(f"No question '{key}' is waiting (pending: {pending}).{hint}")
        plan = self.plan()
        restored = {n: v for n, v in run.values.items() if n in plan.steps}
        for name in run.loops:
            restored[f"loop:{name}"] = True
        run.answers = {**run.answers, **new_answers}
        run.restored = restored
        return run

    def stream(self, inputs: Optional[Dict[str, Any]] = None, **kwinputs: Any) -> Iterator[Event]:
        """Run the flow and yield each :class:`Event` as it happens.

        The last event is ``flow_end``; its ``event.result`` is the RunResult.
        Accepts the same options as :meth:`run`.
        """
        options = {k: kwinputs.pop(k) for k in list(kwinputs) if k in _RUN_OPTIONS}
        merged = _merge(inputs, kwinputs)
        q: "queue.Queue[Any]" = queue.Queue()
        done = object()
        box: Dict[str, Any] = {}

        def target() -> None:
            try:
                run_sync(lambda: self._arun(merged, extra=q.put, owns_loop=True, **options), reuse_loop=False)
            except BaseException as exc:
                box["error"] = exc
            finally:
                q.put(done)

        t = threading.Thread(target=target, name="taskpilot-stream", daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is done:
                break
            yield item
        t.join()
        if "error" in box:
            raise box["error"]

    async def astream(self, inputs: Optional[Dict[str, Any]] = None, **kwinputs: Any) -> AsyncIterator[Event]:
        """Async version of :meth:`stream` – ``async for event in flow.astream(...)``."""
        options = {k: kwinputs.pop(k) for k in list(kwinputs) if k in _RUN_OPTIONS}
        merged = _merge(inputs, kwinputs)
        q: "asyncio.Queue[Any]" = asyncio.Queue()
        task = asyncio.ensure_future(self._arun(merged, extra=q.put_nowait, owns_loop=False, **options))
        task.add_done_callback(lambda _: q.put_nowait(None))
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()

    async def _arun(
        self,
        inputs: Dict[str, Any],
        checkpoint: Any = None,
        lean: bool = False,
        raise_errors: bool = True,
        timeout: Optional[float] = None,
        state: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
        extra: Optional[Callable[[Event], Any]] = None,
        *,
        owns_loop: bool,
        answers: Optional[Dict[str, Any]] = None,
        restored: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        plan = self.plan()
        handlers = [h for h in (extra, console_printer if (self.verbose if verbose is None else verbose) else None) if h]
        if len(handlers) > 1:
            def extra_handler(event: Event) -> None:
                for h in handlers:
                    h(event)
        else:
            extra_handler = handlers[0] if handlers else None  # type: ignore[assignment]
        # A step of this flow that runs this same flow again must not wait on
        # its own (possibly full) pool, so such nested runs get a private one.
        private: List[ThreadPoolExecutor] = []
        def private_pool() -> ThreadPoolExecutor:
            if not private:
                private.append(ThreadPoolExecutor(self.max_workers, thread_name_prefix=f"{self._pool_prefix}-nested"))
            return private[0]

        get_pool = private_pool if self._pool_prefix in threading.current_thread().name else self._get_pool
        runner = Runner(
            self.name,
            plan,
            inputs,
            bus=self._bus,
            extra_handler=extra_handler,
            get_pool=get_pool,
            max_workers=self.max_workers,
            thread_prefix=self._pool_prefix,
            owns_loop=owns_loop,
            state=state,
            lean=lean,
            checkpoint=Checkpoint.coerce(checkpoint),
            raise_errors=raise_errors,
            answers=answers,
            restored=restored,
        )
        limit = timeout if timeout is not None else self.timeout
        try:
            if limit is None:
                return await runner.go()
            return await asyncio.wait_for(runner.go(), limit)
        except asyncio.TimeoutError:
            if limit is None:
                raise
            err = FlowTimeout(self.name, limit, runner.result)
            runner.result.error = err
            if raise_errors:
                raise err from None
            return runner.result
        finally:
            for pool in private:
                pool.shutdown(wait=False)

    # ================================================================ pool
    def _get_pool(self) -> ThreadPoolExecutor:
        pool = self._pool
        if pool is None:
            with self._pool_lock:
                pool = self._pool
                if pool is None:
                    pool = ThreadPoolExecutor(self.max_workers, thread_name_prefix=self._pool_prefix)
                    self._pool = pool
                    weakref.finalize(self, pool.shutdown, wait=False)
        return pool

    def close(self) -> None:
        """Release this flow's worker threads (they are recreated on the next run)."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "Flow":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


_RUN_OPTIONS = frozenset({"checkpoint", "lean", "raise_errors", "timeout", "state", "verbose"})
def _merge(inputs: Optional[Dict[str, Any]], kwinputs: Dict[str, Any]) -> Dict[str, Any]:
    if inputs is None:
        return dict(kwinputs)
    if not isinstance(inputs, dict):
        raise TypeError(f"inputs must be a dict, got {type(inputs).__name__}. Pass inputs as keywords: flow.run(x=1)")
    merged = dict(inputs)
    merged.update(kwinputs)
    return merged


def _step_options(step: Step) -> Dict[str, Any]:
    opts: Dict[str, Any] = dict(
        name=step.name, after=step.after, when=step.when, retry=step.retry, timeout=step.timeout,
        cache=step.cache, on_error=step.on_error, map=step.map, max_concurrency=step.max_concurrency,
        repeat_until=step.repeat_until, max_repeats=step.max_repeats, keep=step.keep, tags=step.tags,
        description=step.description, bind=step.bind, dynamic=step.dynamic, check=step.check,
    )
    if step.has_fallback:
        opts["fallback"] = step.fallback
    return opts
