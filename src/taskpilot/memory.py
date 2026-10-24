"""Memory for agents: short-term with auto-compaction, and long-term facts.

* :class:`ConversationMemory` – the running conversation. When it grows past
  a token budget, older turns are summarised (keeping facts, numbers,
  decisions and open tasks) and only recent turns stay word for word, so the
  model never loses the thread or overflows its context.
* :class:`MemoryStore` – facts to keep across runs and days ("Ravi prefers
  Hindi"), saved to a JSON file and found again by keywords. Give agents
  ``store.as_tools()`` so they can remember and recall on their own.

::

    agent = Agent(llm, memory=ConversationMemory(llm, max_tokens=3000))
    facts = MemoryStore("memory.json")
    agent = Agent(llm, tools=[*facts.as_tools(), get_order], memory=ConversationMemory(llm))
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from .llm.base import LLM
from .llm.tools import Tool

__all__ = ["ConversationMemory", "MemoryStore", "estimate_tokens"]

SUMMARY_PROMPT = (
    "Summarise this conversation so it can be continued later. Keep every fact, number, name, id, date, "
    "decision, promise and open task exactly. Drop greetings and repetition. Write short bullet points.\n\n"
    "{previous}Conversation:\n{text}"
)


def estimate_tokens(messages: Iterable[Dict[str, Any]]) -> int:
    """Rough token count (about 4 characters per token, plus a little per message)."""
    total = 0
    for m in messages:
        total += 4 + len(str(m.get("content") or "")) // 4
        if m.get("tool_calls"):
            total += len(json.dumps(m["tool_calls"], default=str)) // 4
    return total


def _render(messages: Iterable[Dict[str, Any]], limit: int = 800) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        text = str(m.get("content") or "")
        if m.get("tool_calls"):
            calls = ", ".join(f"{c.get('function', {}).get('name')}({json.dumps(c.get('function', {}).get('arguments'), default=str)})"
                              for c in m["tool_calls"])
            text = (text + f" [called {calls}]").strip()
        if role == "tool":
            role = "tool result"
        lines.append(f"{role}: {text[:limit]}{'...' if len(text) > limit else ''}")
    return "\n".join(lines)


class ConversationMemory:
    """A conversation that compacts itself when it gets long.

    Args:
        llm: writes the summaries. Without one, old turns are shortened instead.
        max_tokens: budget for the remembered conversation.
        keep_recent: turns (messages) always kept word for word.
    """

    def __init__(self, llm: Optional[LLM] = None, *, max_tokens: int = 4000, keep_recent: int = 8):
        if max_tokens < 200:
            raise ValueError("max_tokens must be at least 200")
        self.llm = llm
        self.max_tokens = max_tokens
        self.keep_recent = max(2, keep_recent)
        self.messages: List[Dict[str, Any]] = []
        self.summary = ""
        self.compactions = 0
        self._lock = threading.Lock()

    def context(self) -> List[Dict[str, Any]]:
        """Messages to put before a new task (summary first, then recent turns)."""
        with self._lock:
            out: List[Dict[str, Any]] = []
            if self.summary:
                out.append({"role": "user", "content": f"(Summary of our earlier conversation)\n{self.summary}"})
                out.append({"role": "assistant", "content": "Understood. I will use this."})
            out.extend(self.messages)
            return out

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.context())

    def add(self, messages: Iterable[Dict[str, Any]]) -> None:
        """Remember new messages; compacts automatically when over budget."""
        with self._lock:
            self.messages.extend(m for m in messages if m.get("role") != "system")
        if self.tokens > self.max_tokens:
            self.compact()

    def compact(self) -> None:
        """Summarise everything but the recent turns (keeps tool calls paired with their results)."""
        with self._lock:
            if len(self.messages) <= self.keep_recent:
                return
            # Keep at least keep_recent messages, starting at a user turn so no tool
            # result loses its tool call. Look further back if the window has no user turn.
            users = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
            limit = len(self.messages) - self.keep_recent
            before = [i for i in users if i <= limit]
            cut = before[-1] if before else (users[-1] if users else 0)
            if cut <= 0:
                return
            old, keep = self.messages[:cut], self.messages[cut:]
            text = _render(old)
            previous = f"Earlier summary:\n{self.summary}\n\n" if self.summary else ""
        summary = None
        if self.llm is not None:
            try:
                summary = self.llm.ask(SUMMARY_PROMPT.format(previous=previous, text=text)).strip()
            except Exception:
                summary = None
        if not summary:  # no model (or it failed): keep a shortened transcript instead of losing it
            summary = (previous + _render(old, limit=200))[-4000:]
        with self._lock:
            self.summary = summary
            self.messages = keep + self.messages[len(old) + len(keep):]
            self.compactions += 1

    def clear(self) -> None:
        with self._lock:
            self.messages, self.summary = [], ""

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "messages": self.messages, "compactions": self.compactions}

    def load(self, data: Dict[str, Any]) -> "ConversationMemory":
        self.summary = data.get("summary", "")
        self.messages = list(data.get("messages", []))
        self.compactions = int(data.get("compactions", 0))
        return self


_WORD = re.compile(r"[\w@.]+", re.UNICODE)
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "for", "on", "with", "what", "who",
         "how", "ka", "ki", "ke", "hai", "aur", "ko", "se", "me", "mein", "kya"}


def _words(text: str) -> set:
    return {w.lower().strip(".") for w in _WORD.findall(text or "") if len(w) > 1 and w.lower() not in _STOP}


class MemoryStore:
    """Long-term facts, saved to a JSON file and found again by keywords.

    ``remember("Ravi prefers Hindi emails", key="ravi.language")`` replaces any
    earlier fact with the same key, so facts stay current instead of piling up.
    """

    def __init__(self, path: Optional[str] = None, *, max_items: int = 5000):
        self.path = os.fspath(path) if path else None
        self.max_items = max_items
        self.items: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path and os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                self.items = list(json.load(fh).get("items", []))

    def remember(self, text: str, *, key: Optional[str] = None, tags: Iterable[str] = ()) -> Dict[str, Any]:
        text = str(text).strip()
        if not text:
            raise ValueError("Nothing to remember")
        item = {"id": os.urandom(4).hex(), "key": key or None, "text": text, "tags": list(tags), "time": time.time()}
        with self._lock:
            if key:
                self.items = [i for i in self.items if i.get("key") != key]
            self.items.append(item)
            del self.items[:-self.max_items]
            self._save()
        return item

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """The facts that best match ``query`` (newest first among equals)."""
        q = _words(query)
        with self._lock:
            scored = []
            for n, item in enumerate(self.items):
                words = _words(item["text"]) | {t.lower() for t in item.get("tags", [])} | _words(item.get("key") or "")
                score = len(q & words)
                if score:
                    scored.append((score, n, item))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item for _, _, item in scored[:k]]

    def forget(self, id_or_key: str) -> bool:
        with self._lock:
            before = len(self.items)
            self.items = [i for i in self.items if i["id"] != id_or_key and i.get("key") != id_or_key]
            self._save()
            return len(self.items) < before

    def as_tools(self) -> List[Tool]:
        """``remember`` and ``recall`` tools for an agent."""
        store = self

        def remember(fact: str, key: str = "") -> str:
            """Save a fact for later (a preference, a decision, a detail about a customer).

            Args:
                fact: one clear sentence.
                key: optional stable name like "ravi.language"; a new fact with the same key replaces the old one.
            """
            store.remember(fact, key=key or None)
            return "remembered"

        def recall(query: str) -> str:
            """Find saved facts about something (a customer, a decision, a preference)."""
            found = store.recall(query)
            return "\n".join(f"- {i['text']}" for i in found) or "nothing saved about that"

        return [Tool(remember), Tool(recall)]

    def _save(self) -> None:
        if not self.path:
            return
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".taskpilot-", suffix=".tmp", dir=folder)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"items": self.items}, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.items)
