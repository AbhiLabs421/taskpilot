"""Turn plain Python functions into tools a model can call."""

from __future__ import annotations

import asyncio
import enum
import inspect
import json
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..errors import ToolError

try:  # Python 3.10+
    from types import UnionType  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    UnionType = None  # type: ignore[assignment,misc]

_SIMPLE = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", tuple: "array", dict: "object", set: "array"}


def type_schema(tp: Any) -> Dict[str, Any]:
    """JSON schema for a Python type hint."""
    if tp is inspect.Parameter.empty or tp is Any:
        return {"type": "string"}
    if isinstance(tp, str):
        return {"type": {"int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}.get(tp, "string")}
    if tp in _SIMPLE:
        return {"type": _SIMPLE[tp]}
    if tp is type(None):
        return {"type": "null"}
    if inspect.isclass(tp) and issubclass(tp, enum.Enum):
        return {"type": "string", "enum": [m.value for m in tp]}
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Literal:
        kinds = {type(a) for a in args}
        schema: Dict[str, Any] = {"enum": list(args)}
        if len(kinds) == 1:
            schema["type"] = _SIMPLE.get(kinds.pop(), "string")
        return schema
    if origin is typing.Union or (UnionType is not None and origin is UnionType):
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1:
            return type_schema(rest[0])
        return {"anyOf": [type_schema(a) for a in rest]}
    if origin in (list, tuple, set, frozenset):
        schema = {"type": "array"}
        if args and args[0] is not Ellipsis:
            schema["items"] = type_schema(args[0])
        return schema
    if origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _parse_doc(doc: str) -> Tuple[str, Dict[str, str]]:
    """Split a docstring into (description, {param: description}).

    Understands Google style (``Args:``) and Sphinx style (``:param x:``).
    """
    if not doc:
        return "", {}
    params: Dict[str, str] = {}
    for name, text in re.findall(r":param\s+(?:\w+\s+)?(\w+):\s*(.+)", doc):
        params[name] = text.strip()
    lines = doc.splitlines()
    desc_lines: List[str] = []
    in_args = False
    current = None
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(Args|Arguments|Parameters|Params)\s*:\s*$", stripped):
            in_args = True
            continue
        if re.match(r"^(Returns?|Raises?|Yields?|Examples?|Notes?)\s*:\s*$", stripped) or stripped.startswith(":"):
            in_args = False
            current = None
            if stripped.startswith(":"):
                continue
            desc_lines.append("\0")
            continue
        if in_args:
            m = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
            if m:
                current = m.group(1)
                params[current] = m.group(2).strip()
            elif current and stripped:
                params[current] += " " + stripped
            continue
        desc_lines.append(line)
    desc = "\n".join(desc_lines).split("\0")[0].strip()
    return desc, params


def _coerce(value: Any, schema: Dict[str, Any]) -> Any:
    """Gently fix argument types models often get wrong ("5" for 5)."""
    t = schema.get("type")
    try:
        if t == "string" and value is not None and not isinstance(value, str):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False, default=str)
            return str(value)
        if t == "integer" and isinstance(value, str):
            return int(value.strip())
        if t == "integer" and isinstance(value, float) and value.is_integer():
            return int(value)
        if t == "number" and isinstance(value, str):
            return float(value.strip())
        if t == "boolean" and isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "yes", "1"):
                return True
            if low in ("false", "no", "0"):
                return False
        if t in ("array", "object") and isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, list if t == "array" else dict):
                return parsed
    except (ValueError, TypeError):
        pass
    return value


def to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


class Tool:
    """A function the model can call. Usually made with ``@tool``.

    The name, description and parameter schema come from the function's
    name, docstring and type hints.
    """

    def __init__(self, fn: Callable, *, name: Optional[str] = None, description: Optional[str] = None):
        if not callable(fn):
            raise ToolError(f"A tool must be callable, got {fn!r}")
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "tool")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$", self.name):
            raise ToolError(f"Tool name {self.name!r} must be 1-64 letters, digits, '_' or '-'")
        doc_desc, doc_params = _parse_doc(inspect.getdoc(fn) or "")
        self.description = description or doc_desc or self.name.replace("_", " ")
        self.is_async = inspect.iscoroutinefunction(fn)
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            hints = getattr(fn, "__annotations__", {}) or {}
        sig = inspect.signature(fn)
        props: Dict[str, Any] = {}
        required: List[str] = []
        self._accepts_any = False
        #: parameters with a real type hint; only these are converted.
        self._typed = set()
        for p in sig.parameters.values():
            if p.kind == p.VAR_KEYWORD:
                self._accepts_any = True
                continue
            if p.kind == p.VAR_POSITIONAL:
                continue
            hint = hints.get(p.name, p.annotation)
            if hint is not inspect.Parameter.empty and hint is not Any:
                self._typed.add(p.name)
            schema = type_schema(hint)
            if p.name in doc_params:
                schema["description"] = doc_params[p.name]
            if p.default is not p.empty and p.default is not None and isinstance(p.default, (str, int, float, bool)):
                schema["default"] = p.default
            props[p.name] = schema
            if p.default is p.empty:
                required.append(p.name)
        self.parameters = {"type": "object", "properties": props, "required": required}

    @classmethod
    def from_schema(cls, fn: Callable[..., Any], name: str, description: str = "",
                    parameters: Optional[Dict[str, Any]] = None) -> "Tool":
        """A tool whose arguments are described by a JSON schema instead of type hints.

        Used for tools that live somewhere else (an MCP server, a REST API):
        ``fn`` is called with the arguments as keywords.
        """
        def call(**args: Any) -> Any:
            return fn(**args)

        call.__name__ = name
        t = cls(call, name=name, description=description or name.replace("_", " "))
        schema = dict(parameters or {})
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = [r for r in schema.get("required") or [] if isinstance(r, str)]
        t.parameters = {**schema, "type": "object", "properties": props, "required": required}
        t._accepts_any = schema.get("additionalProperties", True) is not False
        t._typed = {k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get("type"), str)}
        return t

    def spec(self) -> Dict[str, Any]:
        """The tool definition sent to the model."""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def prepare(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and coerce model-supplied arguments. Raises ``ValueError``."""
        if not isinstance(arguments, dict):
            raise ValueError(f"arguments must be an object, got {type(arguments).__name__}")
        props = self.parameters["properties"]
        clean = {}
        for k, v in arguments.items():
            if k in props:
                clean[k] = _coerce(v, props[k]) if k in self._typed else v
            elif self._accepts_any:
                clean[k] = v
        missing = [r for r in self.parameters["required"] if r not in clean]
        if missing:
            raise ValueError(f"missing required argument(s): {', '.join(missing)}")
        return clean

    def invoke(self, arguments: Dict[str, Any]) -> str:
        """Call the tool with model arguments and return text for the model.

        Never raises for ordinary errors: the error is returned as text so
        the model can correct itself.
        """
        try:
            args = self.prepare(arguments)
            if self.is_async:
                value = asyncio.run(self.fn(**args))
            else:
                value = self.fn(**args)
                if inspect.isawaitable(value):
                    value = asyncio.run(_await(value))
            return to_text(value)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


async def _await(x: Any) -> Any:
    return await x


def tool(fn: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None) -> Any:
    """Decorator that turns a function into a :class:`Tool`.

    ::

        @tool
        def weather(city: str, unit: Literal["c", "f"] = "c") -> str:
            '''Current weather for a city.

            Args:
                city: city name, e.g. "Delhi".
            '''
    """
    if fn is None:
        return lambda f: Tool(f, name=name, description=description)
    return Tool(fn, name=name, description=description)
