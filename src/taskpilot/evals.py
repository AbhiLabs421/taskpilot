"""Measure how often a workflow or agent gets the right answer.

::

    from taskpilot.evals import Case, evaluate

    report = evaluate(auto.run, [
        Case("add 125 and 75, minus 40", expect=160),
        Case("what is 10 + 20?", expect=lambda out: "30" in str(out)),
    ], runs=5)
    print(report.summary())        # pass rate with a 95% confidence range

Why the range matters: 20 passes out of 20 sounds like 100%, but it only
shows the true rate is probably above ~84%. To show 99.9% you need
thousands of passing runs. The report tells you how much you really know.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional

__all__ = ["Case", "CaseResult", "EvalReport", "evaluate", "wilson"]


def wilson(passed: int, total: int, z: float = 1.96) -> tuple:
    """95% Wilson confidence interval for a pass rate (low, high)."""
    if total == 0:
        return (0.0, 1.0)
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class Case:
    """One test question and how to judge the answer.

    ``expect`` is the exact right answer, or a function ``fn(output) -> bool``
    (or an error message string when wrong). Numbers are compared with a
    small tolerance; text is compared after trimming.
    """

    def __init__(self, input: Any, expect: Any = None, *, name: Optional[str] = None, tags: Iterable[str] = ()):
        self.input = input
        self.expect = expect
        self.name = name or (input if isinstance(input, str) else json.dumps(input, default=str))[:60]
        self.tags = list(tags)

    def judge(self, output: Any) -> Optional[str]:
        """``None`` if the output is right, otherwise why it is wrong."""
        exp = self.expect
        if callable(exp):
            verdict = exp(output)
            if isinstance(verdict, str):
                return verdict or None
            return None if verdict else f"check failed for output {_short(output)}"
        if exp is None:
            return None if output is not None else "no output"
        if isinstance(exp, (int, float)) and not isinstance(exp, bool):
            try:
                got = float(output)
            except (TypeError, ValueError):
                return f"expected {exp}, got {_short(output)}"
            return None if math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-6) else f"expected {exp}, got {got:g}"
        if isinstance(exp, str) and isinstance(output, str):
            return None if output.strip() == exp.strip() else f"expected {exp!r}, got {_short(output)}"
        return None if output == exp else f"expected {_short(exp)}, got {_short(output)}"


class CaseResult:
    __slots__ = ("case", "run", "passed", "abstained", "crashed", "output", "error", "seconds")

    def __init__(self, case: Case, run: int, passed: bool, output: Any, error: str, seconds: float, abstained: bool = False,
                 crashed: bool = False):
        #: the target raised an unexpected error (the workflow "broke")
        self.crashed = crashed
        self.case = case
        self.run = run
        self.passed = passed
        #: the system declined or asked instead of answering (safe, but not an answer)
        self.abstained = abstained
        self.output = output
        self.error = error
        self.seconds = seconds


class EvalReport:
    """Pass rate, confidence range, the failures and timings."""

    def __init__(self, name: str, results: List[CaseResult]):
        self.name = name
        self.results = results

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple:
        return wilson(self.passed, self.total)

    @property
    def failures(self) -> List[CaseResult]:
        return [r for r in self.results if not r.passed]

    @property
    def abstained(self) -> int:
        return sum(r.abstained for r in self.results)

    @property
    def crashed(self) -> int:
        return sum(r.crashed for r in self.results)

    @property
    def wrong(self) -> int:
        """Answers that were given as if correct but were not (the dangerous kind)."""
        return sum(1 for r in self.results if not r.passed and not r.abstained and not r.crashed)

    @property
    def wrong_upper(self) -> float:
        """95% upper bound of the wrong-answer rate."""
        return wilson(self.wrong, self.total)[1]

    def by_case(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in self.results:
            d = out.setdefault(r.case.name, {"passed": 0, "total": 0})
            d["total"] += 1
            d["passed"] += int(r.passed)
        return out

    def summary(self) -> str:
        low, high = self.interval
        secs = sorted(r.seconds for r in self.results) or [0.0]
        lines = [
            f"{self.name}: {self.passed}/{self.total} correct = {self.rate * 100:.1f}%  "
            f"(95% range {low * 100:.1f}% - {high * 100:.1f}%)  median {secs[len(secs) // 2]:.1f}s",
            f"  asked / declined: {self.abstained}   crashed: {self.crashed}   wrong answers: {self.wrong}  "
            f"(wrong-answer rate is below {self.wrong_upper * 100:.1f}% with 95% confidence)",
        ]
        for name, d in self.by_case().items():
            if d["passed"] != d["total"]:
                lines.append(f"  ✗ {d['passed']}/{d['total']}  {name}")
        for r in self.failures[:12]:
            kind = "asked/declined" if r.abstained else "CRASHED" if r.crashed else "WRONG"
            lines.append(f"    - [{kind}] {r.case.name[:40]} run {r.run + 1}: {r.error}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        low, high = self.interval
        return {"name": self.name, "passed": self.passed, "total": self.total, "rate": self.rate,
                "low": low, "high": high, "abstained": self.abstained, "crashed": self.crashed, "wrong": self.wrong,
                "wrong_upper": self.wrong_upper, "cases": self.by_case(),
                "failures": [{"case": r.case.name, "run": r.run, "error": r.error, "output": _short(r.output)}
                             for r in self.failures]}

    def __repr__(self) -> str:
        return f"EvalReport({self.name!r}, {self.passed}/{self.total})"


def evaluate(target: Callable[[Any], Any], cases: Iterable[Case], *, runs: int = 1, parallel: int = 4,
             name: str = "eval", output: Optional[Callable[[Any], Any]] = None,
             abstain: Optional[Callable[[Any], bool]] = None, abstain_on: tuple = ()) -> EvalReport:
    """Run every case ``runs`` times and judge each answer.

    ``target(input)`` is your workflow: a function, ``flow.run``, ``agent.run``,
    ``auto.run``... ``output(result)`` pulls the answer out of what it returns
    (default: ``result.output`` when present). A crash counts as a failure.

    ``abstain(output) -> bool`` and ``abstain_on`` (exception types) mark runs
    where the system asked a question or refused instead of answering. They are
    not correct, but they are not wrong answers either, and the report counts
    them apart from the wrong ones.
    """
    cases = list(cases)
    pick = output or (lambda r: getattr(r, "output", r))

    def one(job: tuple) -> CaseResult:
        case, i = job
        t0 = time.perf_counter()
        try:
            arg = case.input
            out = pick(target(**arg) if isinstance(arg, dict) else target(arg))
            why = case.judge(out)
            held = why is not None and abstain is not None and bool(abstain(out))
            return CaseResult(case, i, why is None, out, why or "", time.perf_counter() - t0, held)
        except Exception as exc:
            held = bool(abstain_on) and isinstance(exc, abstain_on)
            return CaseResult(case, i, False, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0, held, not held)

    jobs = [(c, i) for c in cases for i in range(runs)]
    with ThreadPoolExecutor(max(1, parallel)) as pool:
        results = list(pool.map(one, jobs))
    return EvalReport(name, results)


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."
