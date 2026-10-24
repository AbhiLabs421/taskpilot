"""Same workflows in taskpilot, LangGraph and Microsoft Agent Framework.

Each scenario is written the idiomatic way for each framework, its answer is
checked, and the best of several timings is reported.

    pip install langgraph agent-framework
    python benchmarks/compare.py
"""


import asyncio
import operator
import os
import subprocess
import sys
import time
from typing import Annotated, Any, Callable, Dict, List, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHAIN = 50          # steps in the linear chain
WIDTH = 20          # parallel branches, each waits 50 ms (like an API call)
WAIT = 0.05
LOOP_TO = 10        # loop scenario counts 0 -> 10
RUNS = 300          # small-flow invocations for per-run overhead
ROUNDS = 5          # draft -> review cycles in the multi-step loop
SPAWN = 20          # work items whose count is only known at run time


def best(fn: Callable[[], Any], repeat: int = 5) -> tuple:
    out, t_best = None, float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        t_best = min(t_best, time.perf_counter() - t)
    return out, t_best


# ------------------------------------------------------------------ taskpilot
def taskpilot_suite() -> Dict[str, Callable[[], Any]]:
    from taskpilot import Flow

    chain = Flow("chain")
    chain.add(lambda x: x + 1, name="s0")
    for i in range(1, CHAIN):
        chain.add(lambda p: p + 1, name=f"s{i}", bind={"p": f"s{i - 1}"})

    fan = Flow("fan")

    async def branch(x):
        await asyncio.sleep(WAIT)
        return x

    for i in range(WIDTH):
        fan.add(branch, name=f"b{i}")
    import inspect

    def join(**kw):
        return sum(kw.values())

    join.__signature__ = inspect.Signature([inspect.Parameter(f"b{i}", inspect.Parameter.KEYWORD_ONLY) for i in range(WIDTH)])
    fan.add(join, name="join")

    route = Flow("route")
    route.add(lambda q: "math" if any(c.isdigit() for c in q) else "chat", name="kind")
    route.add(lambda q: "M", name="math", when=lambda kind: kind == "math")
    route.add(lambda q: "C", name="chat", when=lambda kind: kind == "chat")

    loop = Flow("loop")

    def count(ctx):
        return (ctx.previous or 0) + 1

    loop.add(count, name="count", repeat_until=lambda v: v >= LOOP_TO, max_repeats=100)

    small = Flow("small")
    small.add(lambda x: x + 1, name="a")
    small.add(lambda a: a * 2, name="b")
    small.add(lambda b: b - 1, name="c")

    cycle = Flow("cycle")
    cycle.add(lambda x=0: x + 1, name="draft")
    cycle.add(lambda draft: draft, name="review")
    cycle.loop("draft", "review", until=lambda review: review >= ROUNDS, feed={"x": "review"}, max_iterations=100)

    spawn = Flow("spawn")
    spawn.add(lambda item: item * 2, name="work", dynamic=True)
    spawn.add(lambda n, ctx: sum(ctx.map("work", range(n))), name="total")

    return {
        "cycle": lambda: cycle.run().output,
        "spawn": lambda: spawn.run(n=SPAWN).output,
        "chain": lambda: chain.run(x=0).output,
        "fan": lambda: fan.run(x=1).output,
        "route": lambda: (route.run(q="2+2").output, route.run(q="hi").output),
        "loop": lambda: loop.run().output,
        "runs": lambda: [small.run(x=i).output for i in range(RUNS)][-1],
    }


# --------------------------------------------------------------- LangGraph
def langgraph_suite() -> Dict[str, Callable[[], Any]]:
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        x: int

    g = StateGraph(S)
    for i in range(CHAIN):
        g.add_node(f"s{i}", lambda s: {"x": s["x"] + 1})
        g.add_edge(START if i == 0 else f"s{i - 1}", f"s{i}")
    g.add_edge(f"s{CHAIN - 1}", END)
    chain = g.compile()

    class F(TypedDict):
        x: int
        vals: Annotated[List[int], operator.add]
        total: int

    g = StateGraph(F)

    async def branch(s):
        await asyncio.sleep(WAIT)
        return {"vals": [s["x"]]}

    for i in range(WIDTH):
        g.add_node(f"b{i}", branch)
        g.add_edge(START, f"b{i}")
        g.add_edge(f"b{i}", "join")
    g.add_node("join", lambda s: {"total": sum(s["vals"])})
    g.add_edge("join", END)
    fan = g.compile()

    class R(TypedDict):
        q: str
        out: str

    g = StateGraph(R)
    g.add_node("kind", lambda s: {})
    g.add_node("math", lambda s: {"out": "M"})
    g.add_node("chat", lambda s: {"out": "C"})
    g.add_edge(START, "kind")
    g.add_conditional_edges("kind", lambda s: "math" if any(c.isdigit() for c in s["q"]) else "chat")
    g.add_edge("math", END)
    g.add_edge("chat", END)
    route = g.compile()

    g = StateGraph(S)
    g.add_node("count", lambda s: {"x": s["x"] + 1})
    g.add_edge(START, "count")
    g.add_conditional_edges("count", lambda s: END if s["x"] >= LOOP_TO else "count")
    loop = g.compile()

    g = StateGraph(S)
    g.add_node("a", lambda s: {"x": s["x"] + 1})
    g.add_node("b", lambda s: {"x": s["x"] * 2})
    g.add_node("c", lambda s: {"x": s["x"] - 1})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    small = g.compile()

    g = StateGraph(S)
    g.add_node("draft", lambda s: {"x": s["x"] + 1})
    g.add_node("review", lambda s: {})
    g.add_edge(START, "draft")
    g.add_edge("draft", "review")
    g.add_conditional_edges("review", lambda s: END if s["x"] >= ROUNDS else "draft")
    cycle = g.compile()

    from langgraph.types import Send

    class W(TypedDict):
        n: int
        results: Annotated[List[int], operator.add]
        total: int

    g = StateGraph(W)
    g.add_node("work", lambda s: {"results": [s["item"] * 2]})
    g.add_node("join", lambda s: {"total": sum(s["results"])})
    g.add_conditional_edges(START, lambda s: [Send("work", {"item": i}) for i in range(s["n"])])
    g.add_edge("work", "join")
    g.add_edge("join", END)
    spawn = g.compile()

    return {
        "cycle": lambda: cycle.invoke({"x": 0})["x"],
        "spawn": lambda: spawn.invoke({"n": SPAWN, "results": []})["total"],
        "chain": lambda: chain.invoke({"x": 0}, {"recursion_limit": CHAIN + 10})["x"],
        "fan": lambda: asyncio.run(fan.ainvoke({"x": 1, "vals": []}))["total"],
        "route": lambda: (route.invoke({"q": "2+2"})["out"], route.invoke({"q": "hi"})["out"]),
        "loop": lambda: loop.invoke({"x": 0})["x"],
        "runs": lambda: [small.invoke({"x": i})["x"] for i in range(RUNS)][-1],
    }


# ------------------------------------------- Microsoft Agent Framework
def msaf_suite() -> Dict[str, Callable[[], Any]]:
    from typing_extensions import Never

    from agent_framework import Case, Default, WorkflowBuilder, WorkflowContext, executor

    def step(i: int, last: bool = False, fn: Callable[[int], int] = lambda v: v + 1):
        if last:
            @executor(id=f"s{i}")
            async def run(x: int, ctx: WorkflowContext[Never, int]) -> None:
                await ctx.yield_output(fn(x))
        else:
            @executor(id=f"s{i}")
            async def run(x: int, ctx: WorkflowContext[int]) -> None:
                await ctx.send_message(fn(x))
        return run

    def build_chain():
        nodes = [step(i, last=i == CHAIN - 1) for i in range(CHAIN)]
        return WorkflowBuilder(start_executor=nodes[0], max_iterations=CHAIN + 10).add_chain(nodes).build()

    def build_fan():
        @executor(id="start")
        async def start(x: int, ctx: WorkflowContext[int]) -> None:
            await ctx.send_message(x)

        def make(i):
            @executor(id=f"b{i}")
            async def b(x: int, ctx: WorkflowContext[int]) -> None:
                await asyncio.sleep(WAIT)
                await ctx.send_message(x)
            return b

        branches = [make(i) for i in range(WIDTH)]

        @executor(id="join")
        async def join(vals: list[int], ctx: WorkflowContext[Never, int]) -> None:
            await ctx.yield_output(sum(vals))

        return (WorkflowBuilder(start_executor=start).add_fan_out_edges(start, branches)
                .add_fan_in_edges(branches, join).build())

    def build_route():
        @executor(id="kind")
        async def kind(q: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(q)

        @executor(id="math")
        async def math(q: str, ctx: WorkflowContext[Never, str]) -> None:
            await ctx.yield_output("M")

        @executor(id="chat")
        async def chat(q: str, ctx: WorkflowContext[Never, str]) -> None:
            await ctx.yield_output("C")

        return (WorkflowBuilder(start_executor=kind).add_switch_case_edge_group(
            kind, [Case(condition=lambda q: any(c.isdigit() for c in q), target=math), Default(target=chat)]).build())

    def build_loop():
        @executor(id="count")
        async def count(x: int, ctx: WorkflowContext[int, int]) -> None:
            if x + 1 >= LOOP_TO:
                await ctx.yield_output(x + 1)
            else:
                await ctx.send_message(x + 1)

        return WorkflowBuilder(start_executor=count).add_edge(count, count).build()

    def build_small():
        nodes = [step(0), step(1, fn=lambda v: v * 2), step(2, last=True, fn=lambda v: v - 1)]
        return WorkflowBuilder(start_executor=nodes[0]).add_chain(nodes).build()

    def build_cycle():
        @executor(id="draft")
        async def draft(x: int, ctx: WorkflowContext[int]) -> None:
            await ctx.send_message(x + 1)

        @executor(id="review")
        async def review(x: int, ctx: WorkflowContext[int, int]) -> None:
            if x >= ROUNDS:
                await ctx.yield_output(x)
            else:
                await ctx.send_message(x)

        return WorkflowBuilder(start_executor=draft).add_edge(draft, review).add_edge(review, draft).build()

    async def out(wf, msg):
        return (await wf.run(msg)).get_outputs()[-1]

    # Build once and reuse, like the other two (a built workflow can run many times).
    chain, fan, route, loop, small = build_chain(), build_fan(), build_route(), build_loop(), build_small()
    cycle = build_cycle()

    def runs():
        async def main():
            last = None
            for i in range(RUNS):
                last = await out(small, i)
            return last
        return asyncio.run(main())

    return {
        "chain": lambda: asyncio.run(out(chain, 0)),
        "fan": lambda: asyncio.run(out(fan, 1)),
        "route": lambda: (asyncio.run(out(route, "2+2")), asyncio.run(out(route, "hi"))),
        "loop": lambda: asyncio.run(out(loop, 0)),
        "runs": runs,
        "cycle": lambda: asyncio.run(out(cycle, 0)),
        # "spawn" is not written for this framework here, so it shows n/a (not measured).
    }


EXPECTED = {
    "chain": CHAIN,
    "fan": WIDTH,
    "route": ("M", "C"),
    "loop": LOOP_TO,
    "runs": (RUNS - 1 + 1) * 2 - 1,
    "cycle": ROUNDS,
    "spawn": sum(i * 2 for i in range(SPAWN)),
}
LABELS = {
    "chain": f"linear chain, {CHAIN} steps",
    "fan": f"fan-out {WIDTH} x {int(WAIT * 1000)}ms I/O + join",
    "route": "router (2 runs)",
    "loop": f"loop 0 -> {LOOP_TO}",
    "runs": f"{RUNS} runs of a 3-step flow",
    "cycle": f"2-step cycle x{ROUNDS} (draft<->review)",
    "spawn": f"runtime fan-out, {SPAWN} items",
}


def import_time(module: str) -> float:
    code = f"import time; t=time.perf_counter(); import {module}; print(time.perf_counter()-t)"
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    vals = []
    for _ in range(3):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        vals.append(float(out.stdout.strip() or "nan"))
    return min(vals)


def main() -> None:
    import logging

    logging.disable(logging.WARNING)  # MS warns about the intentional self-loop
    suites = {"taskpilot": taskpilot_suite}
    for name, fn, mod in (("LangGraph", langgraph_suite, "langgraph"), ("MS Agent Framework", msaf_suite, "agent_framework")):
        try:
            __import__(mod)
            suites[name] = fn
        except ImportError:
            print(f"(skipping {name}: pip install {mod.replace('_', '-')})")
    built = {name: make() for name, make in suites.items()}
    names = list(built)
    print(f"{'scenario':<34}" + "".join(f"{n:>22}" for n in names))
    for key in EXPECTED:
        row = f"{LABELS[key]:<34}"
        for n in names:
            if key not in built[n]:
                row += f"{'n/a':>22}"
                continue
            try:
                value, t = best(built[n][key], repeat=3 if key == "runs" else 5)
                ok = value == EXPECTED[key]
                row += f"{t * 1000:>18.2f}ms {'ok' if ok else 'XX'}"
            except Exception as exc:
                row += f"{'error: ' + type(exc).__name__:>22}"
        print(row)
    mods = {"taskpilot": "taskpilot", "LangGraph": "langgraph.graph", "MS Agent Framework": "agent_framework"}
    print(f"{'import time':<34}" + "".join(f"{import_time(mods[n]) * 1000:>20.0f}ms" for n in names))


if __name__ == "__main__":
    main()
