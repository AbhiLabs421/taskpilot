"""COMPLEX: randomized graphs, chaos, scale and real-world-shaped pipelines."""

import asyncio
import random
import threading
import time
import unittest
from collections import Counter

from taskpilot import Agent, FakeLLM, Flow, Retry, StepError, tool


def random_dag(seed, n, max_parents=4, async_ratio=0.5):
    """A random DAG whose steps sum their parents; returns (flow, expected)."""
    rnd = random.Random(seed)
    flow = Flow(f"rand{seed}", max_workers=8)
    parents = {}
    calls = Counter()
    lock = threading.Lock()
    for i in range(n):
        ps = rnd.sample(range(i), k=min(i, rnd.randint(0, max_parents))) if i else []
        parents[i] = ps
        names = [f"s{p}" for p in ps]
        weight = rnd.randint(1, 9)

        def make(i=i, names=names, weight=weight, is_async=rnd.random() < async_ratio):
            if is_async:
                async def fn(**kw):
                    with lock:
                        calls[i] += 1
                    if rnd.random() < 0.1:
                        await asyncio.sleep(0)
                    return weight + sum(kw[nm] for nm in names if nm != "x") + kw["x"]
            else:
                def fn(**kw):
                    with lock:
                        calls[i] += 1
                    return weight + sum(kw[nm] for nm in names if nm != "x") + kw["x"]
            import inspect

            fn.__signature__ = inspect.Signature(
                [inspect.Parameter(nm, inspect.Parameter.KEYWORD_ONLY) for nm in names + ["x"]]
            )
            return fn

        fn = make()
        fn_weight = weight
        flow.add(fn, name=f"s{i}")
        parents[i] = (ps, fn_weight)

    def expected(x):
        val = {}
        for i in range(n):
            ps, w = parents[i]
            val[i] = w + sum(val[p] for p in ps) + x
        return val

    return flow, expected, calls


class TestRandomGraphs(unittest.TestCase):
    def test_random_dags_compute_exactly(self):
        for seed in range(25):
            flow, expected, calls = random_dag(seed, n=60)
            r = flow.run(x=seed)
            want = expected(seed)
            for i, v in want.items():
                self.assertEqual(r[f"s{i}"], v, f"seed {seed} step s{i}")
            self.assertTrue(all(c == 1 for c in calls.values()), "every step runs exactly once")
            self.assertEqual(len(calls), 60)

    def test_random_dag_async_entrypoint_matches_sync(self):
        flow, expected, _ = random_dag(99, n=120)
        r1 = flow.run(x=3)
        r2 = asyncio.run(flow.arun(x=3))
        self.assertEqual(r1.values, r2.values)

    def test_large_graph_scale(self):
        flow, expected, calls = random_dag(7, n=2000, max_parents=3)
        t = time.perf_counter()
        r = flow.run(x=1)
        elapsed = time.perf_counter() - t
        self.assertEqual(r["s1999"], expected(1)[1999])
        self.assertLess(elapsed, 10)

    def test_deep_chain_no_recursion_limit(self):
        flow = Flow("deep")
        flow.add(lambda x: x, name="s0")
        for i in range(1, 20000):
            flow.add(lambda p: p + 1, name=f"s{i}", bind={"p": f"s{i - 1}"})
        r = flow.run(x=0, lean=True)
        self.assertEqual(r.output, 19999)

    def test_wide_fan_in(self):
        flow = Flow("wide")
        for i in range(500):
            flow.add(lambda x, i=i: x + i, name=f"leaf{i}")
        import inspect

        def total(**kw):
            return sum(kw.values())

        total.__signature__ = inspect.Signature(
            [inspect.Parameter(f"leaf{i}", inspect.Parameter.KEYWORD_ONLY) for i in range(500)]
        )
        flow.add(total, name="total")
        self.assertEqual(flow.run(x=1).output, 500 + sum(range(500)))


class TestChaos(unittest.TestCase):
    def test_random_transient_failures_recover_with_retry(self):
        rnd = random.Random(1)
        lock = threading.Lock()
        flow = Flow("chaos", max_workers=16)
        attempts = Counter()

        def make(i):
            def fn(x):
                with lock:
                    attempts[i] += 1
                    fail = rnd.random() < 0.4 and attempts[i] < 4
                if fail:
                    raise ConnectionError(f"blip {i}")
                return x + i

            return fn

        for i in range(100):
            flow.add(make(i), name=f"s{i}", retry=Retry(6, backoff=0))
        r = flow.run(x=1)
        self.assertTrue(r.ok)
        self.assertEqual(r.output, {f"s{i}": 1 + i for i in range(100)})
        self.assertGreater(sum(attempts.values()), 100, "some retries happened")

    def test_failure_stops_downstream_and_nothing_runs_twice(self):
        ran = Counter()
        flow = Flow("stop")

        @flow.step
        def a(x):
            ran["a"] += 1
            return x

        @flow.step
        def b(a):
            ran["b"] += 1
            raise ValueError("b broke")

        @flow.step
        def c(b):
            ran["c"] += 1

        @flow.step
        def d(c):
            ran["d"] += 1

        with self.assertRaises(StepError):
            flow.run(x=1)
        self.assertEqual(ran, Counter({"a": 1, "b": 1}))

    def test_continue_mode_runs_every_branch_it_can(self):
        flow = Flow()
        for i in range(20):
            if i % 5 == 0:
                flow.add(lambda x: 1 / 0, name=f"s{i}", on_error="continue")
            else:
                flow.add(lambda x, i=i: i, name=f"s{i}")
        r = flow.run(x=1)
        self.assertEqual(len(r.failed), 4)
        self.assertEqual(len(r.succeeded), 16)

    def test_concurrent_runs_with_cache_and_state(self):
        calls = Counter()
        lock = threading.Lock()
        flow = Flow("shared")

        @flow.step(cache=True)
        def slow_lookup(key):
            with lock:
                calls[key] += 1
            time.sleep(0.01)
            return key.upper()

        @flow.step
        def count(slow_lookup, ctx):
            ctx.update("n", lambda n: n + 1, 0)
            return slow_lookup

        results = []

        def worker(k):
            for _ in range(20):
                results.append(flow.run(key=k).output)

        threads = [threading.Thread(target=worker, args=(k,)) for k in "abcd"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(set(results)), ["A", "B", "C", "D"])
        self.assertTrue(all(v <= 2 for v in calls.values()), calls)


class TestRealWorldShapes(unittest.TestCase):
    def test_rag_style_pipeline_with_agent(self):
        docs = {
            "refund": "Refunds are processed within 5 business days.",
            "shipping": "Shipping is free above 500 rupees.",
        }

        @tool
        def search_docs(query: str) -> str:
            """Search the help center.

            Args:
                query: words to look for.
            """
            hits = [t for k, t in docs.items() if k in query.lower()]
            return "\n".join(hits) or "no results"

        def brain(messages, tools):
            last = messages[-1]
            if last["role"] == "user":
                return {"tool": "search_docs", "args": {"query": last["content"]}}
            return "Answer: " + last["content"]

        agent = Agent(FakeLLM(fn=brain), tools=[search_docs])
        flow = Flow("support")

        @flow.step
        def cleaned(question):
            return question.strip()

        @flow.step(map="cleaned_list")
        def answers(cleaned_list):
            return agent.run(cleaned_list).output

        @flow.step
        def cleaned_list(cleaned):
            return [part for part in cleaned.split("?") if part.strip()]

        @flow.step
        def report(answers):
            return "\n".join(answers)

        out = flow.run(question=" how long does a refund take? is shipping free?").output
        self.assertIn("5 business days", out)
        self.assertIn("free above 500", out)

    def test_etl_with_branch_retry_checkpoint_lean(self):
        import os
        import tempfile

        state = {"flaky": 0}
        flow = Flow("etl")

        @flow.step
        def extract(source):
            return [{"id": i, "amount": i * 10, "country": "IN" if i % 2 else "US"} for i in range(50)]

        @flow.step(retry=Retry(3, backoff=0))
        def enrich(extract):
            state["flaky"] += 1
            if state["flaky"] == 1:
                raise TimeoutError("api slow")
            return [dict(r, tax=r["amount"] * (0.18 if r["country"] == "IN" else 0.07)) for r in extract]

        @flow.step(map="enrich")
        def validate(enrich):
            if enrich["amount"] < 0:
                raise ValueError("negative")
            return enrich

        @flow.step(when=lambda validate: len(validate) > 10)
        def big_batch(validate):
            return round(sum(r["tax"] for r in validate), 2)

        @flow.step(when=lambda validate: len(validate) <= 10)
        def small_batch(validate):
            return 0.0

        @flow.step
        def load(big_batch=None, small_batch=None):
            return {"tax_total": big_batch if big_batch is not None else small_batch}

        with tempfile.TemporaryDirectory() as d:
            r = flow.run(source="db", checkpoint=os.path.join(d, "etl.ckpt"), lean=True)
        expected = round(sum(i * 10 * (0.18 if i % 2 else 0.07) for i in range(50)), 2)
        self.assertEqual(r.output, {"tax_total": expected})
        self.assertEqual(r.steps["enrich"].attempts, 2)
        self.assertEqual(r.skipped, ["small_batch"])


if __name__ == "__main__":
    unittest.main()
