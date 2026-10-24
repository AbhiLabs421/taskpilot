"""Every workflow scenario, using only "add" and "subtract".

    python examples/10_calculator_tutorial.py
    OLLAMA_API_KEY=... python examples/10_calculator_tutorial.py    # also runs the real-AI part

Each section is independent. Read it top to bottom.
"""

import asyncio
import os
import tempfile

from taskpilot import Agent, FakeLLM, Flow, Retry, Router, StepError, tool


def title(text):
    print(f"\n=== {text} ===")


# ---------------------------------------------------------------- 1. basics
title("1. Ek step: add")
flow = Flow("add_only")


@flow.step
def add(a, b):                      # 'a' aur 'b' kisi step ke naam nahi -> inputs
    return a + b


print(flow.run(a=10, b=4).output)   # 14


# ---------------------------------------------------------------- 2. parallel
title("2. Do alag steps: add aur subtract (saath-saath chalte hain)")
flow = Flow("both")


@flow.step
def add(a, b):
    return a + b


@flow.step
def subtract(a, b):
    return a - b


result = flow.run(a=10, b=4)
print(result.output)                # {'add': 14, 'subtract': 6}
print(result["add"], result["subtract"])


# ---------------------------------------------------------------- 3. chain
title("3. Ek ke baad ek: (a + b) - c")
flow = Flow("chain")


@flow.step
def add(a, b):
    return a + b


@flow.step
def subtract(add, c):               # parameter ka naam 'add' -> add step ka result milta hai
    return add - c


print(flow.run(a=10, b=4, c=3).output)   # 11
print(flow.describe())


# ---------------------------------------------------------------- 4. result
title("4. Result ke andar kya hai")
result = flow.run(a=10, b=4, c=3)
print("ok:", result.ok, "| status:", result.status)
print("har step:", result.values)
print(result.summary())


# ---------------------------------------------------------------- 5. choose
title("5. Operation chuno: op='add' ya op='subtract' (when=)")
flow = Flow("choose")


@flow.step(when=lambda op: op == "add")
def add(a, b):
    return a + b


@flow.step(when=lambda op: op == "subtract")
def subtract(a, b):
    return a - b


@flow.step
def answer(add=None, subtract=None):     # default = "agar woh branch skip hui"
    return add if add is not None else subtract


print(flow.run(a=10, b=4, op="add").output)        # 14
print(flow.run(a=10, b=4, op="subtract").output)   # 6

title("5b. Wahi kaam, ready-made Router se")
calc = Router(lambda op: op, {"add": lambda a, b: a + b, "subtract": lambda a, b: a - b})
print(calc.run(a=10, b=4, op="subtract").output)   # 6


# ---------------------------------------------------------------- 6. many
title("6. Bahut saare jode ek saath (map=)")
flow = Flow("many")


@flow.step(map="pair")                   # list ke har item pe ek baar, parallel
def add(pair):
    return pair[0] + pair[1]


@flow.step
def total(add):
    return sum(add)


r = flow.run(pair=[(1, 2), (3, 4), (5, 6)])
print(r["add"], "-> total", r.output)  # [3, 7, 11] -> total 21


# ---------------------------------------------------------------- 7. errors
title("7. Galat input aur errors")
flow = Flow("safe")


@flow.step
def add(a, b):
    return a + b


try:
    flow.run(a=10)                       # b bhool gaye
except Exception as e:
    print(type(e).__name__, "->", e)

try:
    flow.run(a=10, b="4")                # number + text
except StepError as e:
    print("StepError -> step:", e.step, "| cause:", type(e.cause).__name__)

flow = Flow("safe2")
flow.add(lambda a, b: a + b, name="add", fallback=0)        # fail ho toh 0
print("fallback:", flow.run(a=10, b="4").output)

calls = {"n": 0}
flow = Flow("retry")


@flow.step(retry=Retry(3, backoff=0))    # pehli baar fail (jaise network), phir sahi
def add(a, b):
    calls["n"] += 1
    if calls["n"] == 1:
        raise ConnectionError("network blip")
    return a + b


r = flow.run(a=10, b=4)
print("retry:", r.output, "| attempts:", r.steps["add"].attempts)


# ---------------------------------------------------------------- 8. loop
title("8. Loop: 100 me se 30 tab tak ghatao jab tak 0 se kam na ho")
flow = Flow("countdown")


@flow.step
def subtract(start, step, last=None):    # last = pichhle round ka result
    return (last if last is not None else start) - step


@flow.step
def check(subtract):
    return subtract


flow.loop("subtract", "check", until=lambda check: check < 0, feed={"last": "check"}, max_iterations=10)
r = flow.run(start=100, step=30)
print("har round:", r.history["subtract"], "-> final", r.output, "| rounds:", r.loops["subtract_loop"]["rounds"])


# ---------------------------------------------------------------- 9. human
title("9. Insaan se poochho: result 1000 se bada ho toh approval")
flow = Flow("approve")


@flow.step
def add(a, b):
    return a + b


@flow.step(when=lambda add: add > 1000)
def approval(add, ctx):
    return ctx.wait_for("ok", question=f"Result {add} hai. Save karein?")


@flow.step
def save(add, approval=True):
    return f"saved {add}" if approval else "not saved"


r = flow.run(a=900, b=200)
print("status:", r.status, "| sawaal:", r.pending[0].question)
path = r.save(os.path.join(tempfile.gettempdir(), "calc_run.json"))   # baad me, doosre process me bhi
print("resume:", flow.resume(path, ok=True).output)
print("chhota number, koi sawaal nahi:", flow.run(a=1, b=2).output)


# ---------------------------------------------------------------- 10. async
title("10. Async steps (jaise kisi API se number aaye)")
flow = Flow("async")


@flow.step
async def a(x):
    await asyncio.sleep(0.1)
    return x


@flow.step
async def b(y):
    await asyncio.sleep(0.1)             # a aur b saath chalte hain: total ~0.1s, 0.2s nahi
    return y


@flow.step
def add(a, b):
    return a + b


print(flow.run(x=7, y=8).output)                         # normal code se
print(asyncio.run(flow.arun(x=7, y=8)).output)           # async code se


# ---------------------------------------------------------------- 11. dynamic
title("11. Kitne jode hain, run time pe pata chale (dynamic)")
flow = Flow("dynamic")


@flow.step(dynamic=True)
def add_pair(a, b):
    return a + b


@flow.step
def plan(text, ctx):                     # "1+2, 3+4, 10+20"
    pairs = [tuple(int(n) for n in part.split("+")) for part in text.split(",")]
    return ctx.map("add_pair", [{"a": x, "b": y} for x, y in pairs])


print(flow.run(text="1+2, 3+4, 10+20").output)     # [3, 7, 30]


# ---------------------------------------------------------------- 12. agent
title("12. Agent: AI khud tay kare add ya subtract")


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


# Bina internet: FakeLLM pehle se likhe jawab deta hai (tests ke liye)
fake = FakeLLM([
    {"tool": "add_numbers", "args": {"a": 5, "b": 3}},
    {"tool": "subtract_numbers", "args": {"a": 8, "b": 2}},
    "5 + 3 - 2 = 6",
])
r = Agent(fake, tools=[add_numbers, subtract_numbers]).run("5 plus 3 minus 2?")
print(r.output, "| tools:", r.tools_used)

if os.environ.get("OLLAMA_API_KEY"):     # asli AI
    from taskpilot import Ollama

    real = Agent(Ollama("gpt-oss:120b", options={"temperature": 0}), tools=[add_numbers, subtract_numbers],
                 system="Use the tools for every calculation. Reply with the final number only.")
    r = real.run("What is 125 plus 75, minus 40?")
    print("asli AI:", r.output, "| tools:", r.tools_used)


# ---------------------------------------------------------------- 13. both
title("13. Agent + workflow: poora workflow ek tool")
chain = Flow("add_then_subtract", description="Compute (a + b) - c exactly.")


@chain.step
def add(a: float, b: float):
    return a + b


@chain.step
def subtract(add, c: float):
    return add - c


fake = FakeLLM([{"tool": "add_then_subtract", "args": {"a": 10, "b": 4, "c": 3}}, "Answer: 11"])
r = Agent(fake, tools=[chain.as_tool()]).run("(10 + 4) - 3?")
print(r.output, "| workflow tool ne diya:", r.steps[0].output)


# ---------------------------------------------------------------- 14. watch
title("14. Dekho kya ho raha hai (verbose)")
chain.run(a=10, b=4, c=3, verbose=True)
