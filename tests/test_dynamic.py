"""Dynamic steps (ctx.call / ctx.map) and tracing, from easy to complex."""

import asyncio
import io
import json
import os
import random
import tempfile
import threading
import time
import unittest

from taskpilot import Flow, FlowDefinitionError, Retry, StepError
from taskpilot.tracing import ConsoleTracer, JSONLTracer, MemoryTracer, OpenTelemetryTracer, Tracer


class EasyDynamic(unittest.TestCase):
    def test_call_and_map(self):
        flow = Flow()

        @flow.step(dynamic=True)
        def square(n):
            return n * n

        @flow.step
        def plan(count, ctx):
            return ctx.map("square", range(count)) + [ctx.call("square", n=10)]

        self.assertEqual(flow.run(count=4).output, [0, 1, 4, 9, 100])

    def test_dynamic_step_is_not_in_the_graph(self):
        flow = Flow()
        flow.add(lambda x: x, name="helper", dynamic=True)
        flow.add(lambda: "only me", name="main")
        r = flow.run()
        self.assertEqual(r.output, "only me")
        self.assertNotIn("helper", r.steps)
        self.assertIn("dynamic", flow.describe() + str(flow["helper"].options()))

    def test_call_plain_function(self):
        flow = Flow()

        @flow.step
        def go(ctx):
            return ctx.call(lambda word: word.upper(), word="hi")

        self.assertEqual(flow.run().output, "HI")

    def test_calls_are_recorded(self):
        flow = Flow()
        flow.add(lambda n: n + 1, name="inc", dynamic=True)
        flow.add(lambda ctx: ctx.map("inc", [1, 2]), name="main")
        r = flow.run()
        self.assertEqual([c.name for c in r.calls], ["inc", "inc"])
        self.assertEqual({c.parent for c in r.calls}, {"main"})
        self.assertEqual(sorted(c.value for c in r.calls), [2, 3])


class MediumDynamic(unittest.TestCase):
    def test_policies_of_the_dynamic_step_apply(self):
        calls = {"n": 0}
        flow = Flow()

        @flow.step(dynamic=True, retry=Retry(3, backoff=0), cache=True)
        def flaky(x):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError
            return x * 2

        flow.add(lambda ctx: [ctx.call("flaky", x=2), ctx.call("flaky", x=2)], name="main")
        r = flow.run()
        self.assertEqual(r.output, [4, 4])
        self.assertEqual(calls["n"], 2, "one retry, then a cache hit")
        self.assertTrue(r.calls[1].cached)

    def test_timeout_and_fallback(self):
        flow = Flow()
        flow.add(lambda: time.sleep(2), name="slow", dynamic=True, timeout=0.05, fallback="too slow")
        flow.add(lambda ctx: ctx.call("slow"), name="main")
        t = time.perf_counter()
        self.assertEqual(flow.run().output, "too slow")
        self.assertLess(time.perf_counter() - t, 1)

    def test_failure_surfaces_in_the_caller(self):
        flow = Flow()
        flow.add(lambda: 1 / 0, name="bad", dynamic=True)
        flow.add(lambda ctx: ctx.call("bad"), name="main")
        with self.assertRaises(StepError) as cm:
            flow.run()
        self.assertEqual(cm.exception.step, "main")
        self.assertIsInstance(cm.exception.cause, StepError)
        self.assertEqual(cm.exception.cause.step, "bad")

    def test_caller_can_handle_failure(self):
        flow = Flow()
        flow.add(lambda: 1 / 0, name="bad", dynamic=True)

        @flow.step
        def main(ctx):
            try:
                return ctx.call("bad")
            except StepError as e:
                return f"handled {type(e.cause).__name__}"

        self.assertEqual(flow.run().output, "handled ZeroDivisionError")

    def test_clear_errors(self):
        flow = Flow()
        flow.add(lambda a, b=1: a + b, name="add", dynamic=True)
        flow.add(lambda: 1, name="static")
        cases = [
            (lambda ctx: ctx.call("ad", a=1), "Did you mean 'add'"),
            (lambda ctx: ctx.call("static"), "dynamic=True"),
            (lambda ctx: ctx.call("add"), "missing a value for 'a'"),
            (lambda ctx: ctx.call("add", a=1, c=2), "no parameter"),
        ]
        for fn, message in cases:
            f = Flow()
            f.add(lambda a, b=1: a + b, name="add", dynamic=True)
            f.add(lambda: 1, name="static")
            f.add(fn, name="main")
            with self.subTest(message=message):
                with self.assertRaises(StepError) as cm:
                    f.run()
                self.assertIn(message, str(cm.exception.cause))

    def test_definition_errors(self):
        flow = Flow()
        with self.assertRaises(FlowDefinitionError):
            flow.add(lambda x: x, name="a", dynamic=True, map="x")
        with self.assertRaises(FlowDefinitionError):
            flow.add(lambda x: x, name="b", dynamic=True, when=lambda: True)
        flow.add(lambda x: x, name="helper", dynamic=True)
        flow.add(lambda helper: helper, name="uses_it")
        with self.assertRaises(FlowDefinitionError) as cm:
            flow.validate()
        self.assertIn("ctx.call('helper'", str(cm.exception))

    def test_inputs_fill_missing_arguments(self):
        flow = Flow()
        flow.add(lambda text, lang: f"{text}->{lang}", name="translate", dynamic=True)
        flow.add(lambda ctx: ctx.map("translate", ["hi", "bye"]), name="main")
        self.assertEqual(flow.run(lang="hi_IN").output, ["hi->hi_IN", "bye->hi_IN"])


class HighDynamic(unittest.TestCase):
    def test_map_runs_in_parallel_and_respects_limit(self):
        lock = threading.Lock()
        running, peak = [0], [0]
        flow = Flow()

        @flow.step(dynamic=True)
        def work(n):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1
            return n

        flow.add(lambda ctx: ctx.map("work", range(8)), name="fast")
        flow.add(lambda fast, ctx: ctx.map("work", range(6), max_concurrency=2), name="limited")
        t = time.perf_counter()
        r = flow.run()
        self.assertEqual(r["fast"], list(range(8)))
        self.assertLess(time.perf_counter() - t, 0.4)
        self.assertLessEqual(peak[0], 8)

    def test_async_steps_use_acall_amap(self):
        flow = Flow()

        @flow.step(dynamic=True)
        async def fetch(url):
            await asyncio.sleep(0.05)
            return len(url)

        @flow.step(dynamic=True)
        def parse(n):
            return n * 10

        @flow.step
        async def crawl(urls, ctx):
            sizes = await ctx.amap("fetch", urls)
            return [await ctx.acall("parse", n=s) for s in sizes]

        t = time.perf_counter()
        self.assertEqual(flow.run(urls=["a", "bb", "ccc"] * 4).output, [10, 20, 30] * 4)
        self.assertLess(time.perf_counter() - t, 0.3)

    def test_sync_call_inside_async_step_is_explained(self):
        flow = Flow()
        flow.add(lambda: 1, name="one", dynamic=True)

        @flow.step
        async def bad(ctx):
            return ctx.call("one")

        with self.assertRaises(StepError) as cm:
            flow.run()
        self.assertIn("acall", str(cm.exception))

    def test_dynamic_async_step_from_sync_caller(self):
        flow = Flow()

        @flow.step(dynamic=True)
        async def double(n):
            await asyncio.sleep(0)
            return n * 2

        flow.add(lambda ctx: ctx.map("double", [1, 2, 3]), name="main")
        self.assertEqual(flow.run().output, [2, 4, 6])

    def test_nested_dynamic_calls(self):
        flow = Flow()
        flow.add(lambda n: n + 1, name="leaf", dynamic=True)
        flow.add(lambda n, ctx: ctx.call("leaf", n=n) * 10, name="middle", dynamic=True)
        flow.add(lambda ctx: ctx.map("middle", [1, 2]), name="top")
        r = flow.run()
        self.assertEqual(r.output, [20, 30])
        self.assertEqual({(c.name, c.parent) for c in r.calls}, {("leaf", "middle"), ("middle", "top")})

    def test_waiting_inside_dynamic_call_pauses_run(self):
        flow = Flow()
        flow.add(lambda item, ctx: ctx.wait_for(f"ok-{item}") if item == "b" else True, name="check", dynamic=True)
        flow.add(lambda items, ctx: ctx.map("check", items), name="main")
        r = flow.run(items=["a", "b"])
        self.assertTrue(r.paused)
        self.assertEqual(flow.resume(r, **{"ok-b": False}).output, [True, False])

    def test_dynamic_inside_loops(self):
        flow = Flow()
        flow.add(lambda n: n * 2, name="dbl", dynamic=True)
        flow.add(lambda ctx, total=1: ctx.call("dbl", n=total), name="grow")
        flow.add(lambda grow: grow, name="check")
        flow.loop("grow", "check", until=lambda check: check >= 32, feed={"total": "check"})
        self.assertEqual(flow.run().output, 32)


class ComplexDynamic(unittest.TestCase):
    def test_planner_spawns_variable_work(self):
        rnd = random.Random(3)
        for case in range(15):
            depth = rnd.randint(1, 4)
            fan = rnd.randint(1, 4)
            flow = Flow(f"tree{case}")

            @flow.step(dynamic=True)
            def node(level, ctx, fan=fan, depth=depth):
                if level == depth:
                    return 1
                return sum(ctx.map("node", [{"level": level + 1}] * fan))

            flow.add(lambda ctx: ctx.call("node", level=0), name="root")
            r = flow.run()
            self.assertEqual(r.output, fan ** depth, (fan, depth))
            self.assertEqual(len(r.calls), sum(fan ** i for i in range(depth + 1)))

    def test_concurrent_runs_with_dynamic_calls(self):
        flow = Flow()
        flow.add(lambda x: x * x, name="sq", dynamic=True)
        flow.add(lambda n, ctx: sum(ctx.map("sq", range(n))), name="total")
        out = []
        lock = threading.Lock()

        def worker(i):
            for _ in range(10):
                v = flow.run(n=i + 5).output
                with lock:
                    out.append(v == sum(x * x for x in range(i + 5)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(out), 60)
        self.assertTrue(all(out))


class Tracing(unittest.TestCase):
    def make(self):
        flow = Flow("traced")
        flow.add(lambda url: url, name="fetch", dynamic=True)
        state = {"n": 0}

        @flow.step(retry=Retry(2, backoff=0))
        def plan(topic, ctx):
            state["n"] += 1
            if state["n"] == 1:
                raise ConnectionError("blip")
            ctx.log("planning")
            return ctx.map("fetch", ["a", "b"])

        flow.add(lambda plan: 1 / 0, name="broken", on_error="continue")
        flow.add(lambda topic: topic, name="skipped", when=lambda: False)
        return flow

    def test_memory_tracer_builds_a_tree(self):
        flow = self.make()
        mem = flow.trace(MemoryTracer())
        flow.run(topic="x")
        spans = mem.last()
        root = spans[0]
        by_name = {}
        for s in spans:
            by_name.setdefault(s.name, []).append(s)
        self.assertEqual(root.kind, "flow")
        self.assertEqual(root.status, "failed")
        plan = by_name["plan"][0]
        self.assertEqual(plan.parent_id, root.span_id)
        self.assertEqual(plan.attributes["attempts"], 2)
        self.assertEqual([e["name"] for e in plan.events], ["step_retry", "log"])
        self.assertEqual({s.parent_id for s in by_name["fetch"]}, {plan.span_id})
        self.assertEqual(by_name["broken"][0].status, "error")
        self.assertEqual(by_name["broken"][0].attributes["error.type"], "ZeroDivisionError")
        self.assertEqual(by_name["skipped"][0].status, "skipped")
        self.assertTrue(all(s.trace_id == root.trace_id for s in spans))
        self.assertTrue(all(s.end is not None and s.end >= s.start for s in spans))

    def test_jsonl_and_console(self):
        flow = self.make()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t", "spans.jsonl")
            flow.trace(JSONLTracer(path))
            buf = io.StringIO()
            flow.trace(ConsoleTracer(stream=buf))
            flow.run(topic="x")
            with open(path) as fh:
                rows = [json.loads(line) for line in fh]
        self.assertEqual(rows[0]["kind"], "flow")
        self.assertIn("plan", [r["name"] for r in rows])
        self.assertIn("fetch", buf.getvalue())

    def test_untrace_and_concurrent_runs(self):
        flow = Flow()
        flow.add(lambda x: x, name="a")
        mem = flow.trace(MemoryTracer())
        threads = [threading.Thread(target=lambda: [flow.run(x=1) for _ in range(20)]) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(mem.traces), 100)
        self.assertTrue(all(len(t) == 2 for t in mem.traces))
        flow.untrace(mem)
        flow.run(x=1)
        self.assertEqual(len(mem.traces), 100)

    def test_custom_tracer_and_loops_and_waits(self):
        exported = []

        class Mine(Tracer):
            def export(self, spans):
                exported.append([(s.name, s.status) for s in spans])

        flow = Flow()
        flow.add(lambda n=0: n + 1, name="inc")
        flow.add(lambda inc: inc, name="chk")
        flow.loop("inc", "chk", until=lambda chk: chk >= 2, feed={"n": "chk"})
        flow.add(lambda chk, ctx: ctx.wait_for("ok"), name="ask")
        flow.trace(Mine())
        flow.run()
        names = exported[0]
        self.assertEqual(names[0], ("flow", "paused"))
        self.assertEqual([n for n in names if n[0] == "inc"], [("inc", "ok"), ("inc", "ok")])
        self.assertIn(("ask", "waiting"), names)

    def test_opentelemetry_is_optional(self):
        import importlib.util

        if importlib.util.find_spec("opentelemetry") is None:
            with self.assertRaises(ImportError) as cm:
                OpenTelemetryTracer()
            self.assertIn("pip install opentelemetry-api", str(cm.exception))
        else:  # pragma: no cover - only when the user has OTel installed
            OpenTelemetryTracer()


if __name__ == "__main__":
    unittest.main()
