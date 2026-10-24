"""The engine: compiles steps into a :class:`Plan` and runs it.

How a run works
---------------
1. ``compile_plan`` turns step signatures into a dependency graph, checks it
   (unknown names, cycles) and fixes a deterministic topological order.
2. ``Runner.go`` keeps a *ready* queue. A step becomes ready when every step it
   depends on has finished. Ready steps start immediately, so independent
   branches always run in parallel: async steps on the event loop, plain
   functions on a thread pool.
3. When only one step can run and nothing else is in flight, a plain function
   runs directly on the calling thread. Linear pipelines therefore pay no
   thread hand-off cost at all.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import itertools
import threading
import time
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from .context import Context, RunInfo
from .errors import CheckFailed, CycleError, FlowDefinitionError, TaskPilotError, MissingInputError, StepError, StepTimeout, did_you_mean
from .events import Event, EventBus
from .policies import MISSING as CACHE_MISS
from .human import WaitingForInput
from .result import FAILED, PAUSED, SKIPPED, SUCCESS, RunResult, StepResult
from .step import NOTHING, Param, Step, by_name, read_params


class LoopLimit(TaskPilotError):
    """A loop used all its rounds and ``on_max="raise"`` was set."""

    def __init__(self, loop: str, rounds: int):
        self.loop = loop
        self.rounds = rounds
        super().__init__(f"Loop '{loop}' did not finish within max_iterations={rounds}")


class Skip(TaskPilotError):
    """Raise inside a step to skip it (and, by default, the steps that need it).

    ::

        @flow.step
        def maybe_email(user):
            if not user.email:
                raise Skip("user has no email")
    """

    def __init__(self, reason: str = "skipped by step"):
        self.reason = reason
        super().__init__(reason)


def _invoke_in_thread(step: Step, kwargs: Dict[str, Any]) -> Any:
    """Call a plain step function from a worker thread, honouring its timeout."""
    if step.timeout is None:
        value = step.call_fn(**kwargs)
    else:
        box: Dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = step.call_fn(**kwargs)
            except BaseException as exc:
                box["error"] = exc

        t = threading.Thread(target=contextvars.copy_context().run, args=(target,), daemon=True,
                             name=f"taskpilot-timeout-{step.name}")
        t.start()
        t.join(step.timeout)
        if t.is_alive():
            raise StepTimeout(step.name, step.timeout)
        if "error" in box:
            raise box["error"]
        value = box["value"]
    if inspect.isawaitable(value):
        value = asyncio.run(_await(value))
    return value


def _apply_check(step: Step, value: Any) -> None:
    """Run a step's ``check=``; raise :class:`CheckFailed` if it rejects ``value``."""
    try:
        verdict = step.check(value)  # type: ignore[misc]
    except CheckFailed:
        raise
    except Exception as exc:
        raise CheckFailed(step.name, f"{type(exc).__name__}: {exc}") from None
    if isinstance(verdict, str):
        if verdict:
            raise CheckFailed(step.name, verdict)
        return
    if not verdict:
        raise CheckFailed(step.name, f"check rejected {repr(value)[:120]}")


def _safe_name(fn: Callable) -> str:
    name = getattr(fn, "__name__", "") or "call"
    return name if name.isidentifier() else "call"


def _next_or_none(next_fn: Callable[..., Any]) -> Any:
    try:
        return next_fn()
    except StopIteration:
        return None


async def _await(x: Any) -> Any:
    return await x


_enter_task = getattr(asyncio.tasks, "_enter_task", None)
_leave_task = getattr(asyncio.tasks, "_leave_task", None)
_set_running_loop = getattr(asyncio, "_set_running_loop", None)
_get_running_loop = getattr(asyncio, "_get_running_loop", None)
#: Whether plain steps may run directly on the loop thread (the fast path).
#: If this Python lacks the hooks we need, every plain step uses a thread.
INLINE_SUPPORTED = None not in (_enter_task, _leave_task, _set_running_loop, _get_running_loop)


def _call_detached(fn: Callable, kwargs: Dict[str, Any]) -> Any:
    """Call a plain function on the loop thread as if no loop were running.

    So a step can use ``asyncio.run(...)``, start its own event loop or call
    ``flow.run()`` exactly as it would outside taskpilot. Used only when taskpilot
    owns the loop and nothing else is scheduled on it. Both the running loop
    and the current task are stepped out of (Python 3.14 tracks the current
    task per thread, so hiding the loop alone is not enough).
    """
    loop = _get_running_loop()  # type: ignore[misc]
    task = asyncio.current_task(loop) if loop is not None else None
    if task is not None:
        _leave_task(loop, task)  # type: ignore[misc]
    _set_running_loop(None)  # type: ignore[misc]
    try:
        return contextvars.copy_context().run(functools.partial(fn, **kwargs))
    finally:
        _set_running_loop(loop)  # type: ignore[misc]
        if task is not None:
            _enter_task(loop, task)  # type: ignore[misc]


# ---------------------------------------------------------------- the plan
class Loop:
    """Steps ``start`` ... ``end`` that run again until ``until`` is true.

    Made by :meth:`Flow.loop`. The *body* is every step on a path from
    ``start`` to ``end``. Steps after the loop wait until it has finished.
    """

    __slots__ = ("name", "start", "end", "until", "until_params", "until_ctx", "until_call", "feed", "max_iterations", "on_max", "body")

    def __init__(self, name: str, start: str, end: str, until: Callable, feed: Dict[str, str], max_iterations: int, on_max: str):
        if not callable(until):
            raise FlowDefinitionError(f"until= of loop '{name}' must be callable, e.g. lambda review: review['ok']")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise FlowDefinitionError(f"max_iterations of loop '{name}' must be an int >= 1")
        if on_max not in ("stop", "raise"):
            raise FlowDefinitionError(f"on_max of loop '{name}' must be 'stop' or 'raise', got {on_max!r}")
        self.name = name
        self.start = start
        self.end = end
        self.until = until
        self.until_params, self.until_ctx = read_params(until, f"until= of loop '{name}'")
        self.until_call = by_name(until)
        self.feed = dict(feed)
        self.max_iterations = max_iterations
        self.on_max = on_max
        self.body: Tuple[str, ...] = ()

    def __repr__(self) -> str:
        return f"Loop({self.name!r}, {self.start!r} -> {self.end!r}, max={self.max_iterations})"


class Plan:
    """A validated, ordered view of a flow's steps."""

    __slots__ = ("order", "steps", "deps", "upstream", "dependents", "inputs", "sinks", "index", "loops", "loop_of", "dynamic")

    def __init__(self) -> None:
        self.order: Tuple[str, ...] = ()
        self.steps: Dict[str, Step] = {}
        #: step -> steps whose *values* it reads (via its params or ``when``).
        self.deps: Dict[str, Tuple[str, ...]] = {}
        #: step -> every step that must finish first (deps + ``after``).
        self.upstream: Dict[str, Tuple[str, ...]] = {}
        #: step -> steps waiting on it, in plan order.
        self.dependents: Dict[str, Tuple[str, ...]] = {}
        #: run input name -> steps that read it.
        self.inputs: Dict[str, List[str]] = {}
        #: steps nothing depends on – their values form ``result.output``.
        self.sinks: Tuple[str, ...] = ()
        self.index: Dict[str, int] = {}
        #: loop name -> Loop, and step -> the loop whose body holds it.
        self.loops: Dict[str, Loop] = {}
        self.loop_of: Dict[str, Loop] = {}
        #: steps run only on demand via ctx.call / ctx.map.
        self.dynamic: Dict[str, Step] = {}

    def required_inputs(self) -> Dict[str, List[str]]:
        """Inputs that have no default anywhere they are used."""
        out: Dict[str, List[str]] = {}
        for name, users in self.inputs.items():
            needs = [u for u in users if not _input_has_default(self.steps[u], name)]
            if needs:
                out[name] = needs
        for lp in self.loops.values():
            for p in lp.until_params:
                if p.name not in self.steps and not p.has_default:
                    out.setdefault(p.name, []).append(f"loop {lp.name}")
        return out


def _all_params(step: Step) -> List[Param]:
    return list(step.params) + list(step.when_params)


def _input_has_default(step: Step, input_name: str) -> bool:
    for p in _all_params(step):
        if step.source(p.name) == input_name and not p.has_default:
            return False
    return True


def compile_plan(steps: Dict[str, Step], loops: Optional[Dict[str, Loop]] = None) -> Plan:
    plan = Plan()
    plan.steps = dict(steps)
    names = set(steps)
    deps: Dict[str, List[str]] = {}
    upstream: Dict[str, List[str]] = {}
    inputs: Dict[str, List[str]] = {}
    for name, step in steps.items():
        d: List[str] = []
        for p in _all_params(step):
            src = step.source(p.name)
            # A parameter named like its own step is an input: def text(text).
            if src in names and src != name:
                if src not in d:
                    d.append(src)
            else:
                users = inputs.setdefault(src, [])
                if name not in users:
                    users.append(name)
        u = list(d)
        for a in step.after:
            if a not in names:
                raise FlowDefinitionError(
                    f"Step '{name}' has after='{a}', but there is no step named '{a}'.{did_you_mean(a, names)}"
                )
            if a == name:
                raise CycleError([name, name])
            if a not in u:
                u.append(a)
        deps[name] = d
        upstream[name] = u

    # Kahn's algorithm, stable in definition order.
    position = {n: i for i, n in enumerate(steps)}
    remaining = {n: len(upstream[n]) for n in steps}
    children: Dict[str, List[str]] = {n: [] for n in steps}
    for n, ups in upstream.items():
        for up in ups:
            children[up].append(n)
    queue: Deque[str] = deque(n for n in steps if remaining[n] == 0)
    order: List[str] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for c in sorted(children[n], key=position.__getitem__):
            remaining[c] -= 1
            if remaining[c] == 0:
                queue.append(c)
    if len(order) != len(steps):
        raise CycleError(_find_cycle({n for n in steps if remaining[n] > 0}, upstream))

    plan.order = tuple(order)
    plan.index = {n: i for i, n in enumerate(order)}
    plan.deps = {n: tuple(v) for n, v in deps.items()}
    plan.upstream = {n: tuple(v) for n, v in upstream.items()}
    plan.dependents = {n: tuple(sorted(children[n], key=plan.index.__getitem__)) for n in steps}
    plan.inputs = inputs
    plan.sinks = tuple(n for n in order if not children[n])
    for lp in (loops or {}).values():
        _attach_loop(plan, lp)
    return plan


def _closure(start: str, edges: Dict[str, Tuple[str, ...]]) -> Set[str]:
    seen = {start}
    stack = [start]
    while stack:
        for nxt in edges[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _attach_loop(plan: Plan, lp: Loop) -> None:
    names = plan.steps
    for role, step in (("start", lp.start), ("end", lp.end)):
        if step not in names:
            raise FlowDefinitionError(f"Loop '{lp.name}' has {role}='{step}', but there is no such step.{did_you_mean(step, names)}")
    after_start = _closure(lp.start, plan.dependents)
    before_end = _closure(lp.end, plan.upstream)
    body = after_start & before_end
    if lp.end not in after_start:
        raise FlowDefinitionError(
            f"Loop '{lp.name}': step '{lp.end}' does not come after '{lp.start}'. The end step must "
            f"(directly or indirectly) use the start step's result."
        )
    for name in body:
        other = plan.loop_of.get(name)
        if other is not None:
            raise FlowDefinitionError(
                f"Step '{name}' is in loops '{other.name}' and '{lp.name}'. Loops can't overlap; "
                f"put the inner loop in its own flow and use flow.as_step()."
            )
    lp.body = tuple(n for n in plan.order if n in body)
    known = set(body)
    for n in body:
        known |= _closure(n, plan.upstream)
    for p in lp.until_params:
        if p.name in names and p.name not in known:
            raise FlowDefinitionError(
                f"until= of loop '{lp.name}' reads '{p.name}', which only runs after the loop. "
                f"It can read steps inside the loop or before it."
            )
    for param, source in lp.feed.items():
        if source not in known:
            raise FlowDefinitionError(
                f"feed= of loop '{lp.name}': '{source}' must be a step inside or before the loop.{did_you_mean(source, names)}"
            )
        users = [n for n in lp.body if param in names[n].param_names]
        if not users:
            raise FlowDefinitionError(
                f"feed= of loop '{lp.name}' sends '{source}' to parameter '{param}', but no step in the loop "
                f"({', '.join(lp.body)}) has a parameter named '{param}'."
            )
        for n in users:
            if names[n].param_default(param) is NOTHING:
                raise FlowDefinitionError(
                    f"Step '{n}' gets '{param}' from the loop, so give it a default for the first round, "
                    f"e.g. def {n}(..., {param}=None)."
                )
    plan.loops[lp.name] = lp
    for n in lp.body:
        plan.loop_of[n] = lp
    # A step only read by the loop's until= (like a reviewer) is not a final
    # output when other steps continue after the loop.
    if any(c not in body for n in lp.body for c in plan.dependents[n]):
        plan.sinks = tuple(n for n in plan.sinks if n not in body)


def _find_cycle(nodes: Set[str], upstream: Dict[str, List[str]]) -> List[str]:
    start = sorted(nodes)[0]
    path: List[str] = [start]
    seen = {start: 0}
    node = start
    while True:
        node = next(u for u in upstream[node] if u in nodes)
        if node in seen:
            cycle = path[seen[node]:] + [node]
            cycle.reverse()
            return cycle
        seen[node] = len(path)
        path.append(node)


# -------------------------------------------------------------- the runner
class _Outcome:
    __slots__ = ("value", "attempts", "cached", "fallback", "error", "skip", "wait")

    def __init__(self) -> None:
        self.value: Any = None
        self.attempts = 0
        self.cached = False
        self.fallback = False
        self.error: Optional[BaseException] = None
        self.skip: Optional[str] = None
        self.wait: Any = None


def _no_emit(type: str, step: Optional[str], data: Dict[str, Any]) -> None:
    return None


class Runner:
    """Executes one run of a plan. Create a new one per run."""

    def __init__(
        self,
        flow_name: str,
        plan: Plan,
        inputs: Dict[str, Any],
        *,
        bus: EventBus,
        extra_handler: Optional[Callable[[Event], Any]],
        get_pool: Callable[[], ThreadPoolExecutor],
        max_workers: int,
        thread_prefix: str = "taskpilot",
        owns_loop: bool,
        state: Optional[Dict[str, Any]],
        lean: bool,
        checkpoint: Any,
        raise_errors: bool,
        answers: Optional[Dict[str, Any]] = None,
        restored: Optional[Dict[str, Any]] = None,
    ):
        self.plan = plan
        self.inputs = inputs
        self.bus = bus
        self.extra_handler = extra_handler
        self.listening = bool(bus) or extra_handler is not None
        self._emit_lock = threading.RLock()
        self.get_pool = get_pool
        self.max_workers = max_workers
        self.thread_prefix = thread_prefix
        self.owns_loop = owns_loop
        self.lean = lean
        self.checkpoint = checkpoint
        self.raise_errors = raise_errors
        self.run_id = os.urandom(6).hex()
        self.info = RunInfo(flow_name, self.run_id, inputs, state if state is not None else {}, self._emit,
                            dict(answers or {}))
        self.result = RunResult(flow_name, self.run_id, inputs, plan.order, plan.sinks)
        self.result.state = self.info.state
        self.remaining = {n: len(plan.upstream[n]) for n in plan.order}
        self.consumers = {n: len(plan.dependents[n]) for n in plan.order}
        self.ready: Deque[str] = deque()
        self.saved: Dict[str, Any] = dict(restored or {})
        self.result.answers = self.info.answers
        self.info.runner = self
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread = 0
        self._calls = itertools.count(1)
        self._calls_lock = threading.Lock()
        #: loop name -> current round, and values fed into the next round.
        self.rounds: Dict[str, int] = {n: 1 for n in plan.loops}
        self.fed: Dict[str, Dict[str, Any]] = {n: {} for n in plan.loops}

    # ------------------------------------------------------------ events
    def _emit(self, type: str, step: Optional[str], data: Dict[str, Any], at: Optional[float] = None) -> None:
        if not self.listening:
            return
        event = Event(type, self.info.flow, self.run_id, step, data, at)
        handlers = self.bus.handlers_for(type)
        if self.extra_handler is not None:
            handlers.append(self.extra_handler)
        # Steps in worker threads emit too; listeners see one event at a time.
        with self._emit_lock:
            for h in handlers:
                try:
                    h(event)
                except Exception as exc:  # a broken listener must never break the run
                    import warnings

                    warnings.warn(f"taskpilot event handler {h!r} raised {exc!r}", RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------ top level
    async def go(self) -> RunResult:
        try:
            return await self._go()
        finally:
            loop_of = self.plan.loop_of
            self.result.restorable = frozenset(
                n for n in self.plan.order if n not in loop_of or loop_of[n].name in self.result.loops
            )
            # Break the Runner <-> RunInfo cycle so big results are freed at
            # once instead of waiting for the garbage collector.
            self.info.emit = _no_emit
            self.info.runner = None

    async def _go(self) -> RunResult:
        plan = self.plan
        t0 = time.perf_counter()
        self._check_inputs()
        if self.checkpoint is not None:
            self.saved.update(self.checkpoint.load(self.info.flow, self.inputs))
        self._emit("flow_start", None, {"inputs": self.inputs})
        loop = asyncio.get_running_loop()
        self.loop = loop
        self.loop_thread = threading.get_ident()
        running: Dict["asyncio.Task[Any]", str] = {}
        self.ready.extend(n for n in plan.order if self.remaining[n] == 0)
        try:
            while self.ready or running:
                while self.ready:
                    name = self.ready.popleft()
                    step = plan.steps[name]
                    if name in self.saved:
                        self._resume(name, self.saved[name])
                        continue
                    args, why = self._collect(step, step.params)
                    if why is not None:
                        self._skip(name, why)
                        continue
                    if not running and not self.ready and self._can_inline(step):
                        outcome = await self._run_step(step, args, inline=True, t0=time.perf_counter())
                        self._finish(name, outcome)
                        continue
                    task = loop.create_task(self._run_step(step, args, inline=False, t0=time.perf_counter()))
                    running[task] = name
                if not running:
                    break
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    name = running.pop(task)
                    self._finish(name, task.result())
        except BaseException as exc:
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
            self.result.duration = time.perf_counter() - t0
            self.result.error = exc
            if isinstance(exc, StepError):
                exc.result = self.result
            if isinstance(exc, Exception):
                self._emit("flow_end", None, {"ok": False, "duration": self.result.duration, "result": self.result, "error": exc})
            raise
        self.result.duration = time.perf_counter() - t0
        if self.checkpoint is not None and self.result.ok:
            self.checkpoint.complete(self.info.flow)
        self._emit("flow_end", None, {"ok": self.result.ok, "duration": self.result.duration, "result": self.result})
        return self.result

    def _check_inputs(self) -> None:
        known = list(self.plan.steps) + list(self.inputs)
        for name, users in self.plan.required_inputs().items():
            if name not in self.inputs:
                raise MissingInputError(name, users, known)

    def _can_inline(self, step: Step) -> bool:
        return INLINE_SUPPORTED and self.owns_loop and not step.is_async and step.timeout is None and step.map is None

    # ------------------------------------------------------------ arguments
    def _collect(self, step: Step, params: List[Param]) -> Tuple[Dict[str, Any], Optional[str]]:
        """Build keyword arguments, or return a reason to skip the step."""
        args: Dict[str, Any] = {}
        steps = self.result.steps
        lp = self.plan.loop_of.get(step.name)
        fed = self.fed[lp.name] if lp is not None else None
        for p in params:
            if fed and p.name in fed:
                args[p.name] = fed[p.name]
                continue
            src = step.source(p.name)
            if src in self.plan.steps and src != step.name:
                sr = steps[src]
                if sr.status == SUCCESS:
                    args[p.name] = sr.value
                elif p.has_default:
                    args[p.name] = p.default
                else:
                    state = "failed" if sr.status == FAILED else "was skipped"
                    return args, f"needs '{src}', which {state}"
            elif src in self.inputs:
                args[p.name] = self.inputs[src]
            else:
                args[p.name] = p.default  # validated up front: a default exists
        return args, None

    # ------------------------------------------------------------ running
    async def _run_step(self, step: Step, args: Dict[str, Any], inline: bool, t0: float) -> Tuple[_Outcome, float]:
        out = _Outcome()
        try:
            self._emit("step_start", step.name, {"args": args})
            if step.when is not None:
                wargs, why = self._collect(step, step.when_params)
                if why is not None:
                    out.skip = why
                    return out, t0
                if step.when_ctx:
                    wargs["ctx"] = Context(self.info, step.name)
                ok = step.when_call(**wargs)
                if inspect.isawaitable(ok):
                    ok = await ok
                if not ok:
                    out.skip = "condition (when=) is false"
                    return out, t0
            if step.map is None:
                out.value = await self._repeat(step, args, out, None, inline)
            else:
                out.value = await self._map(step, args, out)
        except Skip as s:
            out.skip = s.reason
        except WaitingForInput as w:
            out.wait = w.request
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if step.has_fallback:
                out.error = exc
                out.fallback = True
                try:
                    out.value = step.fallback_value(exc)
                except Exception as fb_exc:
                    out.fallback = False
                    out.error = fb_exc
            else:
                out.error = exc
        return out, t0

    async def _map(self, step: Step, args: Dict[str, Any], out: _Outcome) -> List[Any]:
        items = args[step.map]  # type: ignore[index]
        try:
            items = list(items)
        except TypeError:
            raise TypeError(
                f"map='{step.map}' needs an iterable, got {type(items).__name__}"
            ) from None
        if not items:
            return []
        if not step.is_async and step.timeout is None:
            return await self._map_threads(step, args, out, items)
        sem = asyncio.Semaphore(step.max_concurrency) if step.max_concurrency else None

        async def one(i: int, item: Any) -> Any:
            a = dict(args)
            a[step.map] = item  # type: ignore[index]
            if sem is None:
                return await self._repeat(step, a, out, i, False)
            async with sem:
                return await self._repeat(step, a, out, i, False)

        tasks = [asyncio.ensure_future(one(i, x)) for i, x in enumerate(items)]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _map_threads(self, step: Step, args: Dict[str, Any], out: _Outcome, items: List[Any]) -> List[Any]:
        """Map a plain function: a few worker threads pull items one by one.

        Costs about a microsecond per item instead of one thread hand-off
        per item, and stays balanced when some items are much slower.
        """
        n = len(items)
        results: List[Any] = [None] * n
        next_index = iter(range(n)).__next__
        lock = threading.Lock()
        failure: List[BaseException] = []
        name = step.map

        def emit(type: str, data: Dict[str, Any]) -> None:
            self._emit(type, step.name, data)

        def worker() -> None:
            while not failure:
                with lock:
                    i = _next_or_none(next_index)
                if i is None:
                    return
                a = dict(args)
                a[name] = items[i]  # type: ignore[index]
                ctx = self._new_ctx(step, i) if step.wants_ctx or step.repeat_until is not None else None
                try:
                    results[i] = self._repeat_sync(step, a, out, ctx, emit, lock)
                except BaseException as exc:
                    failure.append(exc)
                    return

        workers = min(n, step.max_concurrency or self.max_workers)
        pool = self.get_pool()
        futures = [asyncio.wrap_future(pool.submit(contextvars.copy_context().run, worker)) for _ in range(workers)]
        try:
            await asyncio.gather(*futures)
        except BaseException:
            failure.append(asyncio.CancelledError())
            raise
        if failure:
            raise failure[0]
        return results

    def _repeat_sync(self, step: Step, args: Dict[str, Any], out: _Outcome, ctx: Optional[Context], emit: Callable, lock: threading.Lock) -> Any:
        if step.repeat_until is None:
            return self._call_sync(step, args, out, ctx, emit, lock)
        assert ctx is not None
        value: Any = None
        for i in range(1, step.max_repeats + 1):
            ctx.iteration = i
            value = self._call_sync(step, args, out, ctx if step.wants_ctx else None, emit, lock)
            done = step.repeat_until(value)
            if inspect.isawaitable(done):
                done = asyncio.run(_await(done))
            if done:
                break
            ctx.previous = value
        return value

    def _call_sync(self, step: Step, args: Dict[str, Any], out: _Outcome, ctx: Optional[Context], emit: Callable, lock: threading.Lock) -> Any:
        """Thread-side twin of :meth:`_call` (cache, then attempts with retry)."""
        key = None
        if step.cache is not None:
            key = step.cache.make_key(args)
            if key is not None:
                hit = step.cache.get(key)
                if hit is not CACHE_MISS:
                    out.cached = True
                    return hit
        kwargs = dict(args)
        if ctx is not None:
            kwargs["ctx"] = ctx
        retry = step.retry
        attempt = 0
        while True:
            attempt += 1
            if ctx is not None:
                ctx.attempt = attempt
            with lock:
                if attempt > out.attempts:
                    out.attempts = attempt
            try:
                value = _invoke_in_thread(step, kwargs)
                if step.check is not None:
                    _apply_check(step, value)
                break
            except (Skip, WaitingForInput):
                raise
            except Exception as exc:
                if not retry.should_retry(exc, attempt):
                    raise
                delay = retry.delay(attempt)
                emit("step_retry", {"attempt": attempt, "attempts": retry.attempts, "delay": delay, "error": exc})
                if delay:
                    time.sleep(delay)
        if key is not None:
            step.cache.set(key, value)  # type: ignore[union-attr]
        return value

    def _new_ctx(self, step: Step, item: Optional[int]) -> Context:
        ctx = Context(self.info, step.name, item)
        lp = self.plan.loop_of.get(step.name)
        if lp is not None:
            ctx.iteration = self.rounds[lp.name]
        return ctx

    async def _repeat(self, step: Step, args: Dict[str, Any], out: _Outcome, item: Optional[int], inline: bool) -> Any:
        ctx = self._new_ctx(step, item) if step.wants_ctx else None
        if step.repeat_until is None:
            return await self._call(step, args, out, ctx, inline)
        if ctx is None:
            ctx = Context(self.info, step.name, item)
        value: Any = None
        for i in range(1, step.max_repeats + 1):
            ctx.iteration = i
            value = await self._call(step, args, out, ctx if step.wants_ctx else None, inline)
            done = step.repeat_until(value)
            if inspect.isawaitable(done):
                done = await done
            if done:
                break
            ctx.previous = value
        return value

    async def _call(
        self,
        step: Step,
        args: Dict[str, Any],
        out: _Outcome,
        ctx: Optional[Context],
        inline: bool,
    ) -> Any:
        """One logical call: cache lookup, then attempts with retry."""
        key = None
        if step.cache is not None:
            key = step.cache.make_key(args)
            if key is not None:
                hit = step.cache.get(key)
                if hit is not CACHE_MISS:
                    out.cached = True
                    return hit
        kwargs = dict(args)
        if ctx is not None:
            kwargs["ctx"] = ctx
        retry = step.retry
        attempt = 0
        while True:
            attempt += 1
            if ctx is not None:
                ctx.attempt = attempt
            if attempt > out.attempts:
                out.attempts = attempt
            try:
                value = await self._invoke(step, kwargs, inline)
                if step.check is not None:
                    _apply_check(step, value)
                break
            except (Skip, WaitingForInput, asyncio.CancelledError):
                raise
            except Exception as exc:
                if not retry.should_retry(exc, attempt):
                    raise
                delay = retry.delay(attempt)
                self._emit("step_retry", step.name, {"attempt": attempt, "attempts": retry.attempts, "delay": delay, "error": exc})
                if delay:
                    await asyncio.sleep(delay)
        if key is not None:
            step.cache.set(key, value)  # type: ignore[union-attr]
        return value

    async def _invoke(self, step: Step, kwargs: Dict[str, Any], inline: bool) -> Any:
        fn = step.call_fn
        timeout = step.timeout
        if step.is_async:
            coro = fn(**kwargs)
            if timeout is None:
                return await coro
            try:
                return await asyncio.wait_for(coro, timeout)
            except asyncio.TimeoutError:
                raise StepTimeout(step.name, timeout) from None
        if inline:
            value = _call_detached(fn, kwargs)
        else:
            call = functools.partial(contextvars.copy_context().run, functools.partial(fn, **kwargs))
            fut = asyncio.wrap_future(self.get_pool().submit(call))
            if timeout is None:
                value = await fut
            else:
                try:
                    value = await asyncio.wait_for(fut, timeout)
                except asyncio.TimeoutError:
                    raise StepTimeout(step.name, timeout) from None
        if inspect.isawaitable(value):
            value = await value
        return value

    # ------------------------------------------------------------ dynamic calls
    def _emit_any_thread(self, type: str, step: Optional[str], data: Dict[str, Any]) -> None:
        self._emit(type, step, data)

    def _dynamic_step(self, target: Any) -> Step:
        if isinstance(target, Step):
            return target
        if isinstance(target, str):
            step = self.plan.dynamic.get(target)
            if step is not None:
                return step
            if target in self.plan.steps:
                raise FlowDefinitionError(
                    f"'{target}' is a normal step of the graph. Mark it @flow.step(dynamic=True) to call it at runtime."
                )
            raise FlowDefinitionError(f"No dynamic step named '{target}'.{did_you_mean(target, self.plan.dynamic)}")
        if callable(target):
            return Step(target, name=_safe_name(target))
        raise TypeError(f"ctx.call() needs a dynamic step name or a function, got {target!r}")

    def _dynamic_args(self, step: Step, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        unknown = [k for k in kwargs if k not in step.param_names]
        if unknown:
            raise TypeError(f"Step '{step.name}' has no parameter(s) {unknown}. Parameters are: {step.param_names}")
        args: Dict[str, Any] = {}
        for p in step.params:
            if p.name in kwargs:
                args[p.name] = kwargs[p.name]
            elif p.name in self.inputs:
                args[p.name] = self.inputs[p.name]
            elif p.has_default:
                args[p.name] = p.default
            else:
                raise TypeError(f"ctx.call('{step.name}') is missing a value for '{p.name}'")
        return args

    def _dynamic_items(self, step: Step, items: Any) -> List[Dict[str, Any]]:
        try:
            items = list(items)
        except TypeError:
            raise TypeError(f"ctx.map('{step.name}', items) needs a list, got {type(items).__name__}") from None
        out = []
        for item in items:
            if isinstance(item, dict):
                out.append(item)
            elif step.params:
                out.append({step.params[0].name: item})  # like Python's map(): the first parameter
            else:
                raise TypeError(f"ctx.map('{step.name}'): the step takes no parameters, so items must be {{}}")
        return out

    def _dynamic_start(self, step: Step, parent: str, args: Dict[str, Any]) -> int:
        call_id = next(self._calls)
        self._emit_any_thread("step_start", step.name, {"args": args, "dynamic": True, "parent": parent, "call_id": call_id})
        return call_id

    def _dynamic_finish(self, step: Step, parent: str, call_id: int, out: _Outcome, t0: float, value: Any, exc: Optional[BaseException]) -> Any:
        sr = StepResult(step.name)
        sr.parent = parent
        sr.duration = time.perf_counter() - t0
        sr.attempts = out.attempts
        sr.cached = out.cached and out.attempts == 0
        base = {"dynamic": True, "parent": parent, "call_id": call_id, "duration": sr.duration}
        try:
            if isinstance(exc, Skip):
                sr.status, sr.reason = SKIPPED, exc.reason
                self._emit_any_thread("step_skip", step.name, {**base, "reason": exc.reason})
                return None
            if exc is not None:
                if step.has_fallback:
                    sr.status, sr.fallback, sr.error = SUCCESS, True, exc
                    sr.value = value = step.fallback_value(exc)
                    self._emit_any_thread("step_end", step.name, {**base, "value": value, "fallback": True, "attempts": sr.attempts})
                    return value
                sr.status, sr.error = FAILED, exc
                self._emit_any_thread("step_error", step.name, {**base, "error": exc, "attempts": sr.attempts})
                raise StepError(step.name, exc, max(out.attempts, 1)) from exc
            sr.status = SUCCESS
            if not self.lean:
                sr.value = value
            self._emit_any_thread("step_end", step.name, {**base, "value": value, "cached": sr.cached, "attempts": sr.attempts})
            return value
        finally:
            with self._calls_lock:
                self.result.calls.append(sr)

    def call_sync(self, target: Any, kwargs: Dict[str, Any], parent: str) -> Any:
        if _get_running_loop is not None and _get_running_loop() is not None:
            raise TaskPilotError("Inside an async step use `await ctx.acall(...)` / `await ctx.amap(...)`.")
        step = self._dynamic_step(target)
        args = self._dynamic_args(step, kwargs)
        out = _Outcome()
        t0 = time.perf_counter()
        call_id = self._dynamic_start(step, parent, args)
        try:
            if step.is_async:
                value = asyncio.run(self._repeat(step, args, out, None, False))
            else:
                ctx = Context(self.info, step.name) if step.wants_ctx or step.repeat_until is not None else None
                value = self._repeat_sync(step, args, out, ctx, lambda t, d: self._emit_any_thread(t, step.name, d), self._calls_lock)
        except WaitingForInput:
            raise
        except Exception as exc:
            return self._dynamic_finish(step, parent, call_id, out, t0, None, exc)
        return self._dynamic_finish(step, parent, call_id, out, t0, value, None)

    def map_sync(self, target: Any, items: Any, parent: str, max_concurrency: Optional[int]) -> List[Any]:
        step = self._dynamic_step(target)
        calls = self._dynamic_items(step, items)
        if not calls:
            return []
        workers = min(len(calls), max_concurrency or step.max_concurrency or self.max_workers)
        if workers == 1:
            return [self.call_sync(step, kw, parent) for kw in calls]
        with ThreadPoolExecutor(workers, thread_name_prefix=f"{self.thread_prefix}-dyn") as pool:
            futures = [pool.submit(contextvars.copy_context().run, self.call_sync, step, kw, parent) for kw in calls]
            try:
                return [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    f.cancel()
                raise

    async def call_async(self, target: Any, kwargs: Dict[str, Any], parent: str) -> Any:
        step = self._dynamic_step(target)
        args = self._dynamic_args(step, kwargs)
        out = _Outcome()
        t0 = time.perf_counter()
        call_id = self._dynamic_start(step, parent, args)
        try:
            value = await self._repeat(step, args, out, None, False)
        except (WaitingForInput, asyncio.CancelledError):
            raise
        except Exception as exc:
            return self._dynamic_finish(step, parent, call_id, out, t0, None, exc)
        return self._dynamic_finish(step, parent, call_id, out, t0, value, None)

    async def map_async(self, target: Any, items: Any, parent: str, max_concurrency: Optional[int]) -> List[Any]:
        step = self._dynamic_step(target)
        calls = self._dynamic_items(step, items)
        limit = max_concurrency or step.max_concurrency
        sem = asyncio.Semaphore(limit) if limit else None

        async def one(kw: Dict[str, Any]) -> Any:
            if sem is None:
                return await self.call_async(step, kw, parent)
            async with sem:
                return await self.call_async(step, kw, parent)

        tasks = [asyncio.ensure_future(one(kw)) for kw in calls]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    # ------------------------------------------------------------ bookkeeping
    def _finish(self, name: str, done: Tuple[_Outcome, float]) -> None:
        out, t0 = done
        step = self.plan.steps[name]
        sr = self.result.steps[name]
        sr.duration = time.perf_counter() - t0
        sr.attempts = out.attempts
        sr.cached = out.cached and out.attempts == 0
        if out.skip is not None:
            self._skip(name, out.skip, sr.duration)
            return
        if out.wait is not None:
            # Waiting for a person: the steps after this one stay pending.
            sr.status = PAUSED
            sr.reason = f"waiting for '{out.wait.key}'"
            self.result.pending.append(out.wait)
            self._emit("step_wait", name, {"request": out.wait, "key": out.wait.key, "question": out.wait.question})
            return
        if out.error is not None and not out.fallback:
            sr.error = out.error
            if step.on_error == "skip":
                self._skip(name, f"error: {type(out.error).__name__}: {out.error}", sr.duration)
                return
            sr.status = FAILED
            sr.reason = ""
            self._emit("step_error", name, {"error": out.error, "attempts": out.attempts, "duration": sr.duration})
            if step.on_error == "raise" and self.raise_errors:
                raise StepError(name, out.error, max(out.attempts, 1), self.result)
            self._release_inputs(name)
            self._advance(name)
            return
        sr.status = SUCCESS
        sr.value = out.value
        if out.fallback:
            sr.fallback = True
            sr.error = out.error
        self._emit("step_end", name, {"value": out.value, "duration": sr.duration, "cached": sr.cached, "fallback": sr.fallback, "attempts": sr.attempts})
        if self.checkpoint is not None and name not in self.plan.loop_of:
            self.checkpoint.save(self.info.flow, self.inputs, name, out.value)
        self._release_inputs(name)
        self._advance(name)

    def _resume(self, name: str, value: Any) -> None:
        sr = self.result.steps[name]
        sr.status = SUCCESS
        sr.value = value
        sr.resumed = True
        self._emit("step_end", name, {"value": value, "duration": 0.0, "resumed": True})
        self._release_inputs(name)
        self._advance(name)

    def _skip(self, name: str, reason: str, duration: float = 0.0) -> None:
        sr = self.result.steps[name]
        sr.status = SKIPPED
        sr.reason = reason
        sr.duration = duration
        self._emit("step_skip", name, {"reason": reason})
        self._release_inputs(name)
        self._advance(name)

    def _advance(self, name: str) -> None:
        remaining = self.remaining
        lp = self.plan.loop_of.get(name)
        for child in self.plan.dependents[name]:
            if lp is not None and child not in lp.body:
                continue  # steps after a loop wait for the whole loop
            remaining[child] -= 1
            if remaining[child] == 0:
                self.ready.append(child)
        if lp is not None and name == lp.end:
            self._loop_round_done(lp)

    def _loop_round_done(self, lp: Loop) -> None:
        """The loop's end step finished: run another round or leave the loop."""
        k = self.rounds[lp.name]
        steps = self.result.steps
        for n in lp.body:
            sr = steps[n]
            self.result.history.setdefault(n, []).append(sr.value if sr.status == SUCCESS else None)
        marker = f"loop:{lp.name}"
        if marker in self.saved:
            done, why = True, "resumed"
        else:
            done, why = self._loop_until(lp, k)
        if not done and k >= lp.max_iterations:
            if lp.on_max == "raise":
                raise StepError(lp.end, LoopLimit(lp.name, lp.max_iterations), k, self.result)
            done, why = True, "max_iterations"
        self._emit("loop_round", lp.end, {"loop": lp.name, "round": k, "done": done, "reason": why})
        if done:
            self.result.loops[lp.name] = {"rounds": k, "stopped_by": why}
            if self.checkpoint is not None:
                for n in lp.body:
                    if steps[n].status == SUCCESS:
                        self.checkpoint.save(self.info.flow, self.inputs, n, steps[n].value)
                self.checkpoint.save(self.info.flow, self.inputs, marker, True)
            for n in lp.body:
                for child in self.plan.dependents[n]:
                    if child not in lp.body:
                        self.remaining[child] -= 1
                        if self.remaining[child] == 0:
                            self.ready.append(child)
            return
        fed = {}
        for param, source in lp.feed.items():
            sr = steps[source]
            if sr.status == SUCCESS:
                fed[param] = sr.value
        self.fed[lp.name] = fed
        self.rounds[lp.name] = k + 1
        body = set(lp.body)
        for n in lp.body:
            steps[n] = StepResult(n)
            self.remaining[n] = sum(1 for up in self.plan.upstream[n] if up in body)
        self.ready.extend(n for n in lp.body if self.remaining[n] == 0)

    def _loop_until(self, lp: Loop, k: int) -> Tuple[bool, str]:
        args: Dict[str, Any] = {}
        steps = self.result.steps
        for p in lp.until_params:
            if p.name in self.plan.steps:
                sr = steps[p.name]
                if sr.status == SUCCESS:
                    args[p.name] = sr.value
                elif p.has_default:
                    args[p.name] = p.default
                else:
                    return True, f"'{p.name}' {'failed' if sr.status == FAILED else 'was skipped'}"
            elif p.name in self.inputs:
                args[p.name] = self.inputs[p.name]
            else:
                args[p.name] = p.default
        if lp.until_ctx:
            ctx = Context(self.info, lp.end)
            ctx.iteration = k
            args["ctx"] = ctx
        try:
            ok = lp.until_call(**args)
        except Exception as exc:
            raise StepError(lp.end, exc, 1, self.result) from exc
        if inspect.isawaitable(ok):
            close = getattr(ok, "close", None)
            if close is not None:
                close()
            raise StepError(lp.end, TypeError("until= must be a plain function, not async"), 1, self.result)
        return (True, "until") if ok else (False, "")

    def _release_inputs(self, name: str) -> None:
        """With ``lean=True`` drop values nobody will read again."""
        if not self.lean:
            return
        steps = self.result.steps
        loop_of = self.plan.loop_of
        if name in loop_of:
            return  # loop steps may run again and read their inputs again
        for up in self.plan.upstream[name]:
            if up in loop_of:
                continue
            self.consumers[up] -= 1
            if self.consumers[up] == 0 and up not in self.plan.sinks and not self.plan.steps[up].keep:
                sr = steps[up]
                if sr.status == SUCCESS:
                    sr.value = None
                    sr.released = True
