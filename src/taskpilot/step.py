"""A :class:`Step` is one function in a flow plus the rules for running it."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .errors import FlowDefinitionError
from .policies import Cache, Retry

#: Parameter names that are injected by the framework instead of looked up.
CONTEXT_NAMES = frozenset({"ctx", "context"})

_EMPTY = inspect.Parameter.empty
_NOTHING = object()

ON_ERROR = ("raise", "skip", "continue")


class Param:
    """One parameter of a step function."""

    __slots__ = ("name", "default", "has_default")

    def __init__(self, name: str, default: Any = _EMPTY):
        self.name = name
        self.default = default
        self.has_default = default is not _EMPTY

    def __repr__(self) -> str:
        return f"Param({self.name!r}{', default' if self.has_default else ''})"


def read_params(fn: Callable, what: str) -> Tuple[List[Param], bool]:
    """Return the named parameters of ``fn`` and whether it takes ``ctx``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise FlowDefinitionError(f"Cannot read the signature of {what}: {exc}") from None
    params: List[Param] = []
    wants_ctx = False
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in CONTEXT_NAMES:
            wants_ctx = True
            continue
        params.append(Param(p.name, p.default))
    return params, wants_ctx


def by_name(fn: Callable) -> Callable:
    """Make ``fn`` callable with keyword arguments only.

    taskpilot passes every value by name. Built-ins like ``len``, ``sorted``
    or ``str.upper`` have positional-only parameters, so those are handed
    over positionally.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn
    posonly = [p.name for p in sig.parameters.values() if p.kind is p.POSITIONAL_ONLY]
    if not posonly:
        return fn

    def split(kw: Dict[str, Any]) -> list:
        args = []
        for name in posonly:
            if name not in kw:
                break
            args.append(kw.pop(name))
        return args

    if is_async_callable(fn):
        async def call_async(**kw: Any) -> Any:
            return await fn(*split(kw), **kw)

        return call_async

    def call(**kw: Any) -> Any:
        return fn(*split(kw), **kw)

    return call


def is_async_callable(fn: Callable) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(fn, "__call__", None)
    return call is not None and inspect.iscoroutinefunction(call)


def callable_name(fn: Callable) -> str:
    name = getattr(fn, "__name__", None) or type(fn).__name__
    return "step" if name == "<lambda>" else name


class Step:
    """A single unit of work inside a :class:`~taskpilot.Flow`.

    You normally don't build this yourself; ``@flow.step`` does it. Every
    option of ``@flow.step(...)`` is stored here.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        name: Optional[str] = None,
        after: Iterable[Any] = (),
        when: Optional[Callable] = None,
        retry: Any = None,
        timeout: Optional[float] = None,
        cache: Any = None,
        fallback: Any = _NOTHING,
        on_error: str = "raise",
        map: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        repeat_until: Optional[Callable] = None,
        max_repeats: int = 10,
        keep: bool = False,
        tags: Iterable[str] = (),
        description: Optional[str] = None,
        bind: Optional[Dict[str, str]] = None,
        dynamic: bool = False,
        check: Optional[Callable[[Any], Any]] = None,
    ):
        if not callable(fn):
            raise FlowDefinitionError(f"A step must be callable, got {fn!r}")
        self.fn = fn
        self.name = name or callable_name(fn)
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise FlowDefinitionError(
                f"Step name {self.name!r} must be a valid Python identifier "
                f"(letters, digits, underscore) so other steps can ask for it by name."
            )
        what = f"step '{self.name}'"
        self.is_async = is_async_callable(fn)
        #: what the engine calls: ``fn`` adapted to keyword-only calls.
        self.call_fn = by_name(fn)
        self.params, self.wants_ctx = read_params(fn, what)
        self.param_names = [p.name for p in self.params]

        self.bind: Dict[str, str] = dict(bind or {})
        for param, source in self.bind.items():
            if param not in self.param_names:
                raise FlowDefinitionError(
                    f"bind= of {what} maps '{param}', but the function has no such parameter. "
                    f"Parameters are: {self.param_names}"
                )
            if not isinstance(source, str) or not source.isidentifier():
                raise FlowDefinitionError(f"bind= of {what}: source for '{param}' must be a step or input name")

        self.after = tuple(a if isinstance(a, str) else getattr(a, "name", None) or callable_name(a) for a in after)
        self.when = when
        if when is not None:
            if not callable(when):
                raise FlowDefinitionError(f"when= of {what} must be callable")
            self.when_params, self.when_ctx = read_params(when, f"when= of {what}")
            self.when_call = by_name(when)
        else:
            self.when_params, self.when_ctx = [], False
            self.when_call = None

        self.retry = Retry.coerce(retry)
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise FlowDefinitionError(f"timeout= of {what} must be a positive number of seconds")
        self.timeout = float(timeout) if timeout is not None else None
        self.cache = Cache.coerce(cache)
        self.fallback = fallback
        if on_error not in ON_ERROR:
            raise FlowDefinitionError(f"on_error= of {what} must be one of {ON_ERROR}, got {on_error!r}")
        self.on_error = on_error

        if map is not None and map not in self.param_names:
            raise FlowDefinitionError(
                f"map='{map}' on {what}, but the function has no parameter named '{map}'. "
                f"Parameters are: {self.param_names}"
            )
        self.map = map
        if max_concurrency is not None and max_concurrency < 1:
            raise FlowDefinitionError(f"max_concurrency= of {what} must be >= 1")
        self.max_concurrency = max_concurrency
        if repeat_until is not None and not callable(repeat_until):
            raise FlowDefinitionError(f"repeat_until= of {what} must be callable")
        if max_repeats < 1:
            raise FlowDefinitionError(f"max_repeats= of {what} must be >= 1")
        self.repeat_until = repeat_until
        self.max_repeats = max_repeats
        self.keep = keep
        if check is not None and not callable(check):
            raise FlowDefinitionError(f"check= of {what} must be a function: lambda result: True/False")
        #: ``fn(result) -> True | False | "reason"``; a rejected result counts as a failed attempt.
        self.check = check
        #: Not part of the graph; other steps run it with ctx.call / ctx.map.
        self.dynamic = bool(dynamic)
        if self.dynamic:
            for opt, val in (("map", map), ("when", when), ("after", self.after)):
                if val:
                    raise FlowDefinitionError(
                        f"{what} is dynamic, so it can't use {opt}=. Dynamic steps run only when another "
                        f"step calls them; use ctx.map('{self.name}', [...]) to run one per item."
                    )
        self.tags = frozenset(tags)
        doc = inspect.getdoc(fn) or ""
        self.description = description if description is not None else (doc.splitlines()[0] if doc else "")

    @property
    def has_fallback(self) -> bool:
        return self.fallback is not _NOTHING

    def fallback_value(self, exc: BaseException) -> Any:
        fb = self.fallback
        if callable(fb):
            try:
                return fb(exc)
            except TypeError:
                return fb()
        return fb

    def source(self, param: str) -> str:
        """The step or input name that feeds ``param``."""
        return self.bind.get(param, param)

    def param_default(self, name: str) -> Any:
        for p in self.params:
            if p.name == name:
                return p.default if p.has_default else _NOTHING
        return _NOTHING

    def options(self) -> Dict[str, Any]:
        """The non-default options of this step (handy for printing)."""
        out: Dict[str, Any] = {}
        if self.retry.attempts > 1:
            out["retry"] = self.retry.attempts
        if self.timeout:
            out["timeout"] = self.timeout
        if self.cache:
            out["cache"] = True
        if self.map:
            out["map"] = self.map
        if self.when is not None:
            out["when"] = True
        if self.repeat_until is not None:
            out["repeat_until"] = True
        if self.on_error != "raise":
            out["on_error"] = self.on_error
        if self.has_fallback:
            out["fallback"] = True
        if self.dynamic:
            out["dynamic"] = True
        if self.check is not None:
            out["check"] = True
        return out

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function directly (useful in unit tests)."""
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Step({self.name!r}, needs={self.param_names})"


NOTHING = _NOTHING
