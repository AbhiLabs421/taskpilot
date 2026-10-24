# Reliability: how sure can we be?

No AI system is 100% right. Not the big hosted models, not this one. What a
system *can* promise is narrower and more useful: **it must not hand you a
wrong answer as if it were right, and it must not break.** When it isn't sure,
it should ask or say so. taskpilot is built around that promise, and it is
**measured**, not assumed.

## The numbers (real model, `benchmarks/reliability.py`)

Two suites, each case run 3 times:

* **planner:** 15 business calculations planned by AutoFlow (some in Hinglish), with exact answers.
* **agent:** 8 questions answered from order data. Two of them are traps where the honest answer is "that doesn't exist" or "that isn't known".

To check that results don't depend on a good model, the working model was also
made to misbehave on purpose with `taskpilot.testing.ChaosLLM`: **25% of
replies had a number changed, and 12% of calls failed like an outage.**

| setup | planner correct | planner **wrong** | agent correct | agent crashed | agent **wrong** |
|---|---:|---:|---:|---:|---:|
| gpt-oss:120b, no guards | 93.3% | 3 | 95.8% | 0 | 1 (invented a date) |
| gpt-oss:120b, guards | 93.3% (+3 asked) | **0** | 100% | 0 | **0** |
| gpt-oss:20b (smaller), no guards | 100% | 0 | 100% | 0 | 0 |
| 120b + **25% chaos**, no guards | 82.2% | 8 | 45.8% | 10 | 3 |
| 120b + **25% chaos**, guards | 93.3% (+3 asked) | **0** | **100%** | **0** | **0** |

"Asked" means the request really could be read two ways ("har din 450 kamaye
aur 2000 kharch kiye": was the 2000 spent per day or in total?). With guards
on, the system asks instead of guessing.

**How to read this honestly:**
* 0 wrong in 69 guarded runs means the wrong-answer rate is very likely below
  ~5%, not that it is 0%. Each report prints this bound. To *show* 99.9% you
  need about 3,000 passing runs on cases like yours, so run
  `taskpilot.evals.evaluate` on your own cases.
* The small model scoring 100% shows the suite is still easy. The chaos runs
  are the real stress test.

## What does the work

| guard | what it stops | how to turn it on |
|---|---|---|
| **Tools do the facts** | the model inventing numbers or data | AutoFlow plans tool calls; the model never calculates |
| **Plan validation** | unknown tools, missing arguments, steps that use later results | always on in AutoFlow |
| **Plan review panel** | wrong numbers copied, wrong operations, missing/extra steps, ambiguous requests | `AutoFlow(review=[other_llm, ...])` (on by default in Workspace) |
| **Ungrounded-number hints** | numbers that aren't in the request (12500 for 125000) | automatic with `review` |
| **Ask when unclear** | guessing between two readings | `Unclear` is raised with a question |
| **Consensus** | one planner's random mistake | `AutoFlow(consensus=[llm_a, llm_b, llm_c])`: the majority plan wins, otherwise it asks |
| **Answer verification** | made-up facts in agent answers | `Agent(verify=other_llm)` (on by default in Workspace) |
| **Fix loop** | answers that fail a check | the agent is told what is wrong and retries (`max_fixes`) |
| **Escalation** | a weak model that keeps failing | `escalate=strong_llm` on Agent, AutoFlow and Workspace |
| **Never silently wrong** | passing an unchecked answer on as trusted | `result.verified` is `False` after the fixes run out; `strict=True` raises instead |
| **Step checks** | impossible results (negative totals, empty lists) | `@flow.step(check=lambda r: r > 0 or "must be positive")` |
| **Model fallback** | outages, rate limits, out-of-credit errors | `FallbackLLM([primary, backup])` |
| **Approvals** | risky actions done without a person | `approve=["send_email"]`, enforced in plans *and* inside agents |
| **Retries, timeouts, fallbacks, checkpoints** | flaky tools, crashes mid-run | step options, see the README |

**Use a different model to check.** The single most useful finding: when
gpt-oss:120b checked its own plan, it repeated its own misreading. Every
*different* model we tried (gpt-oss:20b, gemma4, nemotron) caught it. Checkers
should not share the worker's blind spots.

## Memory

| kind | what it is | how |
|---|---|---|
| **Inside one run** | step results passed by name, `ctx.state` | each step and agent sees only what it needs, so prompts stay small and focused |
| **Tool results** | long outputs are cut before the model sees them | `Agent(max_tool_chars=6000)` |
| **Conversation (auto-compact)** | when the conversation passes a token budget, older turns are summarised (facts, numbers, names, decisions and open tasks are kept) and recent turns stay word for word; tool calls are never split from their results | `Agent(memory=ConversationMemory(llm, max_tokens=4000))` |
| **Long-term facts** | facts saved to a file and found again by keywords; a fact with the same `key` replaces the old one, so memory stays current | `MemoryStore("memory.json")`, give agents `store.as_tools()` |
| **Paused runs, checkpoints, saved workflows** | resume after hours or a crash | `result.save()`, `checkpoint=`, Workspace "Saved" |

## Test your own workflow

```python
from taskpilot.evals import Case, evaluate
from taskpilot.testing import ChaosLLM

report = evaluate(lambda q: auto.run(q).output, [
    Case("Net sales: 125000 + 75000 - 40000", 160000),
    Case("Status of order Z9?", lambda out: "not" in out.lower() or "should say it does not exist"),
], runs=10, abstain_on=(Unclear,))
print(report.summary())   # correct %, asked, crashed, wrong, and the 95% bound
```

Run it again with `ChaosLLM(your_llm, numbers=0.25, outage=0.1)` to see what
your guards catch.
