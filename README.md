# taskpilot

**Fast, dependency-free workflows and AI agents for Python.**

Write plain Python functions. Name a parameter after another step and it gets
that step's result. That's the whole idea. taskpilot works out the order, runs
independent steps in parallel, retries failures, caches results, resumes after
crashes, and tells you exactly what went wrong when something breaks.

```bash
pip install taskpilot
```

* **Zero dependencies.** Standard library only. Python 3.9 – 3.14.
* **Fast.** About 5 µs of framework overhead per step, and 2 µs per item in a `map` ([benchmarks](#performance)).
* **Plain functions.** No state classes, no graph objects, no decorators you have to learn first.
* **Sync and async.** Mix them freely. Parallel by default.
* **Reliable.** Retry, timeout, fallback, cache, checkpoints, fail-fast or best-effort.
* **Every workflow shape.** Chains, parallel, routing, map-reduce, multi-step loops, human approval (pause/resume), steps spawned at runtime.
* **Ready-made patterns.** `Sequential`, `Parallel`, `Router`, `MapReduce`, `ReviewLoop`, `Vote`, `Supervisor`, `Handoff`, `GroupChat`.
* **AI built in.** Ollama (local or cloud) and any OpenAI-compatible API, tools, agents, JSON with self-repair.
* **Observable.** Events, live streaming, span tracing (JSONL, console, optional OpenTelemetry).
* **Measured reliability.** Plan review panels, answer verification, consensus, escalation, model fallback, step checks, evals and a chaos tester. When the system isn't sure, it asks instead of guessing.
* **Memory.** Auto-compacting conversation memory and a long-term fact store.
* **Plugs into other platforms.** MCP client and server, OpenAPI/REST tools, A2A remote agents, and a REST API ([docs/INTEGRATIONS.md](https://github.com/AbhiLabs421/taskpilot/blob/main/docs/INTEGRATIONS.md)).

New here? Start with [docs/TUTORIAL.md](https://github.com/AbhiLabs421/taskpilot/blob/main/docs/TUTORIAL.md): every workflow
scenario using only *add* and *subtract* (in Hinglish).
How sure can you be of the AI parts? See [docs/RELIABILITY.md](https://github.com/AbhiLabs421/taskpilot/blob/main/docs/RELIABILITY.md):
measured results, including runs where the model was made to misbehave on
purpose. With guards on there were 0 wrong answers and 0 crashes in 69 runs.

---

## 30-second tour

```python
from taskpilot import Flow

flow = Flow("greet")

@flow.step
def name(raw):                 # 'raw' isn't a step → it's an input
    return raw.strip().title()

@flow.step
def greeting(name):            # 'name' is a step → gets its result
    return f"Namaste, {name}!"

result = flow.run(raw="  asha ")
result.output          # 'Namaste, Asha!'
result["name"]         # 'Asha'
print(result.summary())
```

**The rule:** a parameter named like a step receives that step's result. Any
other parameter is a run input (passed to `flow.run(...)`) or uses its
default. A parameter named `ctx` receives the run [context](#context-shared-state-logs).

---

## Everything a step can do

Every option reads like what it does:

```python
from taskpilot import Flow, Retry

flow = Flow("orders")

@flow.step(retry=3, timeout=10)                    # 3 attempts, 10s each
def orders(day): ...

@flow.step(map="orders", max_concurrency=8)        # once per item, in parallel
def checked(orders): ...

@flow.step(cache=True)                             # same inputs → instant result
def prices(checked): ...

@flow.step(when=lambda prices: len(prices) > 0)    # run only if true, else skip
def invoice(prices): ...

@flow.step(fallback=[])                            # on failure use [] instead
def recommendations(orders): ...

@flow.step(repeat_until=lambda d: "TODO" not in d, max_repeats=5)
def draft(ctx): ...                                # loop; ctx.previous = last result

@flow.step(after=["invoice"])                      # order only, no data needed
def notify(): ...

@flow.step(bind={"data": "orders"})                # feed param from another name
def audit(data): ...
```

| Option | Meaning |
|---|---|
| `retry=3` / `Retry(5, backoff=0.5, on=ConnectionError)` | attempts in total, exponential backoff with jitter |
| `timeout=10` | seconds allowed per attempt |
| `cache=True` / `cache=60` / `Cache(ttl, maxsize)` | memoize by inputs (thread-safe LRU, optional TTL) |
| `when=lambda <deps>: bool` | conditional step / routing |
| `map="param"` | fan-out over a list, results come back in order |
| `max_concurrency=n` | limit parallel `map` items |
| `repeat_until=fn`, `max_repeats=n` | loop a step until its result is good |
| `fallback=value` or `fallback=fn(error)` | never fail, use this instead |
| `on_error="raise" \| "skip" \| "continue"` | stop the run (default), skip the step, or mark it failed and keep going |
| `after=[...]` | run after these steps, without using their values |
| `bind={"param": "source"}` | wire a parameter to a differently named step/input |
| `keep=True` | keep this value even with `lean=True` |
| `name=`, `tags=`, `description=` | metadata |

### Branching and merging

A skipped step also skips the steps that need it, unless the parameter has a
default. That gives you if/else routing for free:

```python
@flow.step
def kind(question):
    return "math" if any(c.isdigit() for c in question) else "chat"

@flow.step(when=lambda kind: kind == "math")
def math_answer(question): ...

@flow.step(when=lambda kind: kind == "chat")
def chat_answer(question): ...

@flow.step
def answer(math_answer=None, chat_answer=None):   # whichever branch ran
    return math_answer or chat_answer
```

Raise `Skip("reason")` inside any step to skip it on purpose.

### Running

```python
result = flow.run(x=1)                              # inputs as keywords
result = flow.run({"x": 1})                         # ...or as a dict
result = await flow.arun(x=1)                       # inside async code
result = flow.run(x=1, checkpoint="job.ckpt")       # resume after a crash
result = flow.run(x=1, lean=True)                   # free memory as you go
result = flow.run(x=1, raise_errors=False)          # get a failed result instead of an exception
result = flow.run(x=1, timeout=30, verbose=True)    # whole-run limit, live trace

for event in flow.stream(x=1):                      # live progress
    print(event.type, event.step)
```

`flow.run()` works anywhere, even inside Jupyter or a running event loop.
Steps can call `asyncio.run()` or another flow's `run()` themselves.

### The result

```python
result.output          # final value (the steps nothing depends on)
result["step"]         # any step's value
result.ok              # no step failed
result.failed          # ['step', ...]
result.skipped
result.steps["x"]      # StepResult: status, value, error, attempts, duration, cached...
result.summary()       # readable table
result.to_dict()       # JSON-friendly
```

### Errors that tell you what to do

```text
MissingInputError: Missing input 'subject' (needed by 'topic'). Pass it as
flow.run(subject=...), give the parameter a default value, or add a step named
'subject'. Did you mean 'subjct'?

CycleError: Cycle detected: a -> c -> b -> a. A step cannot (directly or
indirectly) depend on itself. Use `repeat_until=` on a single step for loops.

StepError: Step 'enrich' failed after 3 attempts: TimeoutError: api slow
```

Wiring mistakes surface before anything runs. `StepError` carries
`.step`, `.cause`, `.attempts` and the partial `.result`. Every taskpilot error
inherits from `TaskPilotError`.

### Context: shared state, logs

```python
@flow.step
def score(answer, ctx):
    ctx.set("seen", True)                        # thread-safe shared state
    ctx.update("count", lambda n: n + 1, 0)      # atomic read-modify-write
    ctx.log("scored", attempt=ctx.attempt)       # goes to listeners / verbose
    return len(answer)
```

`ctx` also has `step`, `attempt`, `iteration`, `previous`, `item` (map index),
`inputs`, `run_id` and `emit(event, **data)`.

### Events

```python
flow.on("step_end", lambda e: print(e.step, e.duration))
flow.on("step_retry", lambda e: print("retrying", e.step, e.error))
flow.on("*", my_tracer)        # everything
```

Events: `flow_start, flow_end, step_start, step_end, step_retry, step_skip,
step_error, log`, plus your own from `ctx.emit`. A listener that crashes only
raises a warning. It never breaks the run.

### Composition

```python
pipeline = Flow.chain(load, clean, save)        # straight line, each gets the previous result
outer.add(inner_flow.as_step("enrich"))         # a whole flow as one step
flow.add(existing_step, name="copy")            # reuse a step with its options
print(flow.describe())                          # text outline
print(flow.to_mermaid())                        # diagram for GitHub / mermaid.live
```

---

## No code for business users: AutoFlow

A developer registers tools once. Then people describe the job in plain
words (any language the model knows), see the plan, and run it.

```python
from taskpilot import AutoFlow, Ollama

auto = AutoFlow(Ollama("gpt-oss:120b"), tools=[add_numbers, subtract_numbers, send_email],
                approve=["send_email"])          # these always ask a person first
result = auto.run("Sales 125000 aur 75000 thi, returns 40000. Net sales boss@company.com ko email karo")
print(result.plan.describe())
```

```text
1. s1: add_numbers(a=125000, b=75000)  (can start right away)
2. s2: subtract_numbers(a=<result of s1>, b=40000)  (after s1)
3. s3: send_email(to="boss@company.com", subject="Net Sales", body="The net sales amount is <s2>.")  [needs approval]  (after s2)
```

* The model only **plans**. The plan is checked (real tools, required
  arguments, references only to earlier steps), and the model is asked to fix
  mistakes. Then the normal engine runs it, so independent steps run in
  parallel, failures are retried, and pause/resume works.
* Tools in `approve` **always** wait for a person. The engine enforces this,
  whatever the model wrote.
* Built-in `ask_ai` (text work) and `ask_human` (ask the user a question).
  Agents, teams and whole flows can be tools too.
* `plan.save("monthly.json")` makes a reusable no-code workflow.
  `auto.load("monthly.json").run()` runs it again without calling the model.

Business users don't touch Python:

```bash
taskpilot serve examples/office_tools.py       # web page: type, review the plan, run, approve
taskpilot ask examples/office_tools.py "net sales nikaalo aur boss ko email karo"
taskpilot ask examples/office_tools.py --plan monthly.json --yes   # rerun a saved plan
```

### The web page (`taskpilot serve`)

Five tabs, no code:

* **Run** – type the request and choose *how*: **AI decides (recommended)**
  shows the AI's suggestion with a reason, or pick a type yourself: Step plan,
  One agent, One after another, All at once, Pick the right agent, Do and
  check, Each item separately, Boss delegates, Pass to specialist, Team
  discussion. Click agents in the order they should work.
* **Agents** – see agents from code, and create, edit or delete your own:
  name, instructions, and which tools each may use.
* **Tools** – what the system can do; tools that ask first are marked.
* **Saved** – save any run as a named workflow and run it again with one click
  (saved step plans run without calling the AI).
* **History** – recent runs; open any of them again.

Approval works everywhere. When an agent calls a tool from `approve`, the
call is held and shown as an *Approval needed* card. Nothing happens until
someone clicks Approve. In *Do and check*, the doer drafts without those
tools and acts only after the checker says OK.

The same features are available from Python:

```python
from taskpilot import Workspace

ws = Workspace(llm, tools=[get_order, send_email], approve=["send_email"], store="workspace.json")
ws.create_agent("support", "Polite support agent.", tools=["get_order", "send_email"])
ws.create_agent("reviewer", "Checks emails for tone and facts.")
tip = ws.suggest("Thank the customer of order A7 by email; reviewer checks first")
run = ws.start(request, tip["pattern"], tip["agents"])      # run.status == "waiting"
ws.approve_action(run.id, run.actions[0]["id"], True)        # now the email is sent
```

The tools file holds a `tools` list (plus optional `agents`, `approve`,
`instructions` and `llm`). Agents and workflows made on the page are kept in
`--store` (default `taskpilot_workspace.json`). Without `llm`, the model is picked from the environment
(`OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` or `OLLAMA_API_KEY`, with
`TASKPILOT_MODEL` naming the model). The web page listens on 127.0.0.1;
put your own login in front before sharing it.

## Loops: repeat several steps until good

```python
@flow.step
def draft(topic, feedback=None):          # feedback = last round's review (None at first)
    ...

@flow.step
def review(draft):
    ...

flow.loop("draft", "review",
          until=lambda review: review["score"] >= 8,
          feed={"feedback": "review"},
          max_iterations=5)               # safety limit; on_max="raise" to fail instead
```

Every step between `start` and `end` repeats. Steps after the loop wait for
it to finish. `ctx.iteration` is the round, `result.history["draft"]` holds
every round's value, and `result.loops` says how many rounds ran and why the
loop stopped. Wiring mistakes (end not after start, overlapping loops, a fed
parameter without a default) fail before anything runs.

## Human in the loop: pause, ask, resume

```python
@flow.step
def approval(draft, ctx):
    return ctx.wait_for("approve", question="Publish this?", draft=draft)

result = flow.run(topic="AI")
result.paused                      # True. Independent branches still finished.
result.pending                     # [Request('approve', step='approval', 'Publish this?')]
result.save("run.json")            # resume later, even in another process

result = flow.resume("run.json", approve=True)   # finished steps are reused
```

The waiting step runs again from its first line on resume, and `wait_for`
returns the answer. Waiting is never retried or replaced by a fallback.
Typos in answer keys are caught ("Did you mean 'approve'?"). Inside loops, use
a key per round, e.g. `f"review-{ctx.iteration}"`. From the shell:
`taskpilot run f.py` exits with code 3 and saves the run, and
`taskpilot resume f.py paused-run.json approve=true` continues it.

## Dynamic steps: decide the work at runtime

```python
@flow.step(dynamic=True, retry=3, cache=True)     # not part of the graph
def research(question):
    ...

@flow.step
def plan(goal, ctx):
    questions = make_questions(goal)               # how many? known only now
    return ctx.map("research", questions)           # parallel, keeps order

# ctx.call("research", question="...") for one call; `await ctx.acall/amap` in async steps.
```

The dynamic step's own retry, timeout, cache and fallback apply to every
call. Calls can nest (a dynamic step can call others), and each one is
recorded in `result.calls` with its parent step.

## Tracing

```python
from taskpilot.tracing import ConsoleTracer, JSONLTracer, MemoryTracer, OpenTelemetryTracer

flow.trace(ConsoleTracer())                 # timing tree after each run
flow.trace(JSONLTracer("spans.jsonl"))      # one span per line
flow.trace(OpenTelemetryTracer())           # if opentelemetry-api is installed (optional)
```

```text
✓ research                     22.15ms
  ✓ plan                       22.02ms
    ✓ fetch                    20.34ms
    ✓ fetch                    20.28ms
  ✓ report                      0.05ms
```

Each run is one trace. Steps and dynamic calls are child spans with status,
attempts, cache hits and errors. Retries, logs, loop rounds and waits are span
events.

---

## Ready-made patterns

```python
from taskpilot import Sequential, Parallel, Router, MapReduce, ReviewLoop, Vote

Sequential(load, clean, save)                          # a -> b -> c
Parallel(summary, keywords, sentiment)                 # same inputs, all at once
Router(classify, {"math": solve, "chat": talk}, default=fallback)
MapReduce(grade_one, average)                          # run(items=[...])
ReviewLoop(write, critique, until=lambda r: r["ok"])   # evaluator-optimizer
Vote(answer, n=5)                                      # self-consistency, majority wins
```

Each returns a normal `Flow`, so `run`, `arun`, `stream`, `trace`,
`as_step` and checkpoints all work. Built-ins work as steps too:
`Sequential(str.strip, str.upper, len)`.

Patterns for teams of agents:

```python
from taskpilot import Supervisor, Handoff, GroupChat

boss = Supervisor(llm, [researcher, writer])          # plans and delegates (orchestrator-workers)
support = Handoff([triage, billing, tech])            # agents pass the conversation along
review = GroupChat([pm, dev, qa], rounds=6, finishers=["qa"])   # turns on one transcript; only qa may end it with "DONE"

support.run("I was charged twice").path                # ["triage", "billing"]
flow.add(support.as_step("reply"))                     # a team inside a flow
Supervisor(llm, [support, review])                     # teams of teams
```

---

## AI: models, tools, agents

No SDKs needed. Everything uses `urllib`.

```python
from taskpilot import Ollama, OpenAICompatible

llm = Ollama("gpt-oss:120b")       # Ollama Cloud if OLLAMA_API_KEY is set, else localhost:11434
llm = Ollama("llama3.2", host="http://localhost:11434")
llm = OpenAICompatible("gpt-4o-mini")                                 # OPENAI_API_KEY
llm = OpenAICompatible("llama-3.3-70b", base_url="https://api.groq.com/openai/v1", api_key="...")

llm.ask("Capital of India?")                       # 'New Delhi'
llm.json("Country and capital of Japan", schema={...})   # parsed + validated, self-repairs bad JSON
for piece in llm.stream("Tell a story"): print(piece, end="")
await llm.aask("...")                              # async
llm.embed(["text one", "text two"])                # Ollama embeddings
```

### Where the URL and API key go

Pass them in code, or set environment variables and leave them out:

| client | in code | environment variables |
|---|---|---|
| OpenAI | `OpenAICompatible("gpt-4o-mini", api_key="sk-...")` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| any OpenAI-compatible server (Groq, Together, OpenRouter, vLLM, LM Studio, Ollama `/v1`...) | `OpenAICompatible("model", base_url="https://host/v1", api_key="...")` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` |
| Azure OpenAI | `OpenAICompatible.azure("deployment", endpoint="https://NAME.openai.azure.com", api_key="...", api_version="2024-10-21")` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` |
| Ollama (native API) | `Ollama("gpt-oss:120b", host="...", api_key="...")` | `OLLAMA_HOST`, `OLLAMA_API_KEY` |

Extra options for unusual servers: `headers={"OpenAI-Organization": "..."}`,
`api_key_header="X-Api-Key"` (sends the key without `Bearer`), `query={...}`
(URL parameters), `timeout=`, `retries=`, and `options={"temperature": 0}`.
Every client has the same methods, so switching providers is a one-line change.
Keep keys in environment variables or a secrets manager, not in code.

Network errors, 429s and 5xx responses are retried with backoff (and
`Retry-After` is honoured). Other errors raise `LLMError` with a hint, like
"Check your API key."

### LLM steps in a flow

```python
flow.prompt("summary", llm, "Summarize in 3 bullets:\n{article}")   # {article} = step or input
flow.prompt("facts", llm, "Extract facts from {article}", json=True)
```

### Tools and agents

```python
from typing import Literal
from taskpilot import Agent, tool

@tool
def weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Current weather for a city.

    Args:
        city: city name, e.g. "Delhi".
    """
    return f"31°{unit.upper()} in {city}"

agent = Agent(llm, tools=[weather], system="Be brief.", max_steps=8)
result = agent.run("Is it hot in Delhi?")
result.output        # the answer
result.tools_used    # ['weather']
```

* The tool schema comes from type hints and the docstring (`Literal`/`Enum` become enums, `Optional`, `list[str]`...).
* Arguments are checked and gently coerced (`"5"` becomes `5`). Tool errors are sent back to the model so it can recover.
* Several tool calls in one turn run in parallel.
* `memory=True` keeps the conversation and trims it safely. `max_steps` forces a final answer.
* Teams: `Agent(llm, tools=[researcher.as_tool()])`.
* Tests without network: `FakeLLM(["reply", {"tool": "weather", "args": {"city": "Delhi"}}])`.

### Workflow, agent, or both?

| | you write | who decides the steps | good for |
|---|---|---|---|
| **Workflow** (`Flow`) | the steps, as functions | you | fixed processes: predictable, testable, cheap |
| **Agent** (`Agent`) | tools + a goal | the model, at run time | open-ended tasks where the path is not known |
| **Both** | a few of each | you set the rules, the model chooses inside them | most real apps |

```python
agent = Agent(llm, tools=[order_status, issue_refund])          # the model picks the tools
agent = Agent(llm, tools=[refund_flow.as_tool()])               # a whole workflow is one tool (fixed rules inside)
flow.add(agent.as_step("reply", "Customer says: {message}"))     # an agent is one step of a workflow
```

`examples/09_agent_or_workflow.py` does the same refund job all three ways, and
`examples/11_multi_agent_tutorial.py` shows eight ways to combine several agents.

---

## Connect to other platforms

Your platform already has tool and agent services? Use them as they are, and
offer taskpilot back to them. More in [docs/INTEGRATIONS.md](https://github.com/AbhiLabs421/taskpilot/blob/main/docs/INTEGRATIONS.md).

```python
from taskpilot import A2AAgent, Agent, MCPClient, MCPServer, Supervisor, openapi_tools

crm = MCPClient.http("https://tools.example.com/mcp", headers={"Authorization": "Bearer ..."})
tools = crm.tools() + openapi_tools("https://api.example.com/openapi.json")   # MCP + REST tools
reporter = A2AAgent("https://agents.example.com/reporter")                     # a remote agent (A2A)
team = Supervisor(llm, [Agent(llm, tools=tools, name="support"), reporter])

MCPServer([my_tool, my_flow, my_agent]).serve_stdio()     # taskpilot as an MCP server
```

```bash
taskpilot mcp tools.py                    # tools, agents and workspace runs for MCP hosts (stdio)
taskpilot serve tools.py --token SECRET   # web page + REST API (/v1, with OpenAPI spec) + MCP (/mcp)
```

Checked against the official MCP SDK (both directions, stdio and HTTP) and the
official A2A SDK (protocol 1.0 and 0.3). `examples/12_connect_platforms.py`
runs the whole thing offline.

---

## Command line

```bash
taskpilot run pipeline.py subject=AI count=3 --verbose   # values are parsed as JSON when possible
taskpilot run pipeline.py:flow --json --checkpoint job.ckpt
taskpilot resume pipeline.py paused-run.json approve=true   # continue a paused run
taskpilot show pipeline.py        # outline
taskpilot graph pipeline.py       # Mermaid
taskpilot ask tools.py "add 125 and 75, then take away 40"
taskpilot serve tools.py --token SECRET    # web page, REST API and MCP
taskpilot mcp tools.py                     # MCP server on stdio (--http for HTTP)
```

---

## Performance

`python benchmarks/bench.py` measures framework overhead with steps that do
nothing. Python 3.13, one Linux container:

| benchmark | total | per unit |
|---|---:|---:|
| linear chain, 1000 sync steps | 4.8 ms | 4.8 µs / step |
| fan-out, 1000 async steps in parallel | 8.6 ms | 8.6 µs / step |
| fan-out, 1000 sync steps (thread pool) | 69 ms | 69 µs / step |
| 1000 runs of a 3-step flow | 64 ms | 64 µs / run |
| map over 10,000 items | 19 ms | 1.9 µs / item |
| 20 steps × 5 MB, peak memory | 100 MB → **10 MB** with `lean=True` | |

### Compared with LangGraph and Microsoft Agent Framework

`python benchmarks/compare.py` (needs `pip install langgraph agent-framework`)
runs the same workflows written the idiomatic way in each framework, checks
every answer, and reports the best of several runs. LangGraph 1.2.16,
agent-framework 1.19.0, Python 3.13:

| scenario | taskpilot | LangGraph | MS Agent Framework |
|---|---:|---:|---:|
| linear chain, 50 steps | **0.28 ms** | 11.9 ms | 11.1 ms |
| fan-out 20 × 50 ms I/O + join | **51 ms** | 58 ms | 54 ms |
| router (2 runs) | **0.54 ms** | 1.61 ms | 1.48 ms |
| loop 0 → 10 | **0.07 ms** | 2.80 ms | 2.88 ms |
| 300 runs of a 3-step flow | **20 ms** | 295 ms | 221 ms |
| 2-step cycle × 5 (draft ↔ review) | **0.12 ms** | 2.63 ms | 2.56 ms |
| runtime fan-out, 20 items | **2.9 ms** | 8.7 ms | not measured |
| import time | 73 ms | 743 ms | **49 ms** |
| installed dependencies | **0** | 38 packages | 11 packages |

Read this honestly. These rows measure the framework's own overhead. When
steps do real work, like the fan-out row or any LLM call that takes
seconds, all three take about the same time. The other two are far ahead
in ecosystem and maturity: hosted tracing and deployment, database
checkpointers, many integrations, and many production users.

How it stays fast:

* **No thread hand-off when nothing runs in parallel.** A lone plain function runs directly on the calling thread, so linear pipelines cost microseconds.
* **Work-stealing `map`.** A few threads pull items one by one. That stays balanced when some items are slow.
* **One reusable event loop per thread**, plans compiled once and cached, no per-step allocations beyond the result.
* **Memory:** `lean=True` frees each value as soon as nothing needs it, and runs leave no reference cycles, so memory is released at once.

---

## Tests

```bash
python -m unittest discover -s tests -t .
```

The core suite is organised by difficulty: `test_easy` (basics),
`test_medium` (retry, timeout, cache, branching, errors), `test_high`
(parallelism, map, checkpoints, memory, streaming, nested runs, threads) and
`test_complex` (randomized DAGs checked against a reference, chaos testing
with random failures, 20,000-step chains, 2,000-node graphs, real-world-shaped
pipelines). Each newer feature has its own file with Easy, Medium, High and
Complex classes: `test_loops`, `test_human` (including a resume in a second
process), `test_dynamic` (dynamic steps and tracing) and `test_patterns`.
`test_llm` covers tools, agents, JSON repair and the real HTTP clients
against a local fake server, `test_connect` covers MCP, REST/OpenAPI tools,
A2A agents and the REST API, and `test_cli` covers the command line.
`test_live_ollama` runs everything against a real model when `OLLAMA_API_KEY`
is set.

## License

MIT
