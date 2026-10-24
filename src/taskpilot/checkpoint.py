"""Crash-safe checkpoints: a failed or interrupted run continues where it stopped.

::

    flow.run({"url": u}, checkpoint="runs/job.ckpt")   # crashes at step 7
    flow.run({"url": u}, checkpoint="runs/job.ckpt")   # steps 1-6 are reused

Values are stored as JSON when that round-trips exactly, otherwise pickled.
Only load checkpoint files you wrote yourself (pickle can run code).
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import tempfile
import threading
from typing import Any, Dict, Optional

from .policies import Cache

_VERSION = 1


def _encode(value: Any) -> Dict[str, Any]:
    try:
        text = json.dumps(value)
        if json.loads(text) == value:
            return {"j": value}
    except (TypeError, ValueError, OverflowError):
        pass
    return {"p": base64.b64encode(pickle.dumps(value, protocol=4)).decode("ascii")}


def _decode(entry: Dict[str, Any]) -> Any:
    if "j" in entry:
        return entry["j"]
    return pickle.loads(base64.b64decode(entry["p"]))


class Checkpoint:
    """Stores finished step values in one file, written atomically.

    Args:
        path: file to use. Parent folders are created.
        keep: keep the file after a successful run (default: delete it, so the
            next run starts fresh).
    """

    def __init__(self, path: str, keep: bool = False):
        self.path = os.fspath(path)
        self.keep = keep
        self._lock = threading.Lock()
        self._data: Optional[Dict[str, Any]] = None

    @classmethod
    def coerce(cls, value: Any) -> Optional["Checkpoint"]:
        if value is None or isinstance(value, Checkpoint):
            return value
        return cls(value)

    @staticmethod
    def _fingerprint(inputs: Dict[str, Any]) -> str:
        return Cache.make_key(inputs) or ""

    def load(self, flow: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``{step: value}`` saved for this flow and these exact inputs."""
        fp = self._fingerprint(inputs)
        with self._lock:
            self._data = {"version": _VERSION, "flow": flow, "fingerprint": fp, "steps": {}}
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                return {}
            if not isinstance(data, dict) or data.get("version") != _VERSION:
                return {}
            if data.get("flow") != flow or data.get("fingerprint") != fp:
                return {}  # different flow or inputs: start fresh
            restored: Dict[str, Any] = {}
            for name, entry in (data.get("steps") or {}).items():
                try:
                    restored[name] = _decode(entry)
                except Exception:
                    continue  # one unreadable value just means that step reruns
            self._data["steps"] = {n: data["steps"][n] for n in restored}
            return restored

    def save(self, flow: str, inputs: Dict[str, Any], step: str, value: Any) -> None:
        with self._lock:
            if self._data is None:
                self._data = {"version": _VERSION, "flow": flow, "fingerprint": self._fingerprint(inputs), "steps": {}}
            try:
                self._data["steps"][step] = _encode(value)
            except Exception:
                return  # value can't be stored; that step will simply rerun
            self._write()

    def complete(self, flow: str) -> None:
        if self.keep:
            return
        with self._lock:
            try:
                os.remove(self.path)
            except OSError:
                pass

    def _write(self) -> None:
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".taskpilot-", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def __repr__(self) -> str:
        return f"Checkpoint({self.path!r})"
