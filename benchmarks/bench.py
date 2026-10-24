"""Measure taskpilot's own overhead (the steps do almost nothing).

    python benchmarks/bench.py
"""

import asyncio
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from taskpilot import Flow  # noqa: E402


def timeit(label, fn, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return label, best


def chain(n):
    flow = Flow("chain")
    flow.add(lambda x: x + 1, name="s0")
    for i in range(1, n):
        flow.add(lambda prev: prev + 1, name=f"s{i}", bind={"prev": f"s{i - 1}"})
    return flow


def fan_out(n, use_async):
    flow = Flow("fan")
    if use_async:
        async def leaf(x):
            return x + 1
    else:
        def leaf(x):
            return x + 1
    for i in range(n):
        flow.add(leaf, name=f"leaf{i}")
    return flow


def main():
    rows = []
    c = chain(1000)
    c.validate()
    label, t = timeit("linear chain, 1000 sync steps", lambda: c.run(x=0))
    rows.append((label, t, t / 1000))

    f = fan_out(1000, use_async=True)
    f.validate()
    label, t = timeit("fan-out, 1000 async steps in parallel", lambda: f.run(x=0))
    rows.append((label, t, t / 1000))

    f = fan_out(1000, use_async=False)
    f.validate()
    label, t = timeit("fan-out, 1000 sync steps (thread pool)", lambda: f.run(x=0))
    rows.append((label, t, t / 1000))

    small = chain(3)
    small.validate()
    label, t = timeit("1000 runs of a 3-step flow", lambda: [small.run(x=0) for _ in range(1000)])
    rows.append((label, t, t / 1000))

    m = Flow("map")
    m.add(lambda items: items * 2, name="double", map="items")
    data = list(range(10000))
    label, t = timeit("map over 10,000 items (sync)", lambda: m.run(items=data), repeat=3)
    rows.append((label, t, t / 10000))

    async def many_async():
        return await asyncio.gather(*(small.arun(x=i) for i in range(1000)))
    label, t = timeit("1000 concurrent arun() of 3-step flow", lambda: asyncio.run(many_async()), repeat=3)
    rows.append((label, t, t / 1000))

    print(f"{'benchmark':<42} {'total':>10} {'per unit':>12}")
    for label, total, per in rows:
        print(f"{label:<42} {total * 1000:>8.1f}ms {per * 1e6:>10.1f}µs")

    tracemalloc.start()
    c.run(x=0)
    _, peak_full = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    big = Flow("big")
    big.add(lambda x: bytearray(5_000_000), name="b0")
    for i in range(1, 20):
        big.add(lambda prev: bytearray(5_000_000), name=f"b{i}", bind={"prev": f"b{i - 1}"})
    tracemalloc.reset_peak()
    big.run(x=0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    big.run(x=0, lean=True)
    _, peak_lean = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nmemory, 20 steps x 5MB: keep-all peak {peak / 1e6:.0f}MB, lean=True peak {peak_lean / 1e6:.0f}MB")


if __name__ == "__main__":
    main()
