"""HIGH: parallelism, map/loops, checkpoints, memory, sub-flows, streaming, async apps."""

import asyncio
import gc
import os
import tempfile
import threading
import time
import unittest
import weakref

from taskpilot import Checkpoint, Flow, Retry, StepError


class TestParallelism(unittest.TestCase):
    def test_independent_sync_steps_run_in_parallel(self):
        flow = Flow()
        for i in range(8):
            flow.add(lambda x, i=i: (time.sleep(0.2), x + i)[1], name=f"s{i}")
        t = time.perf_counter()
        r = flow.run(x=0)
        elapsed = time.perf_counter() - t
        self.assertLess(elapsed, 0.6, "8 x 0.2s steps should overlap")
        self.assertEqual(r.output, {f"s{i}": i for i in range(8)})

    def test_independent_async_steps_run_concurrently(self):
        flow = Flow()

        async def wait(x):
            await asyncio.sleep(0.2)
            return x

        for i in range(50):
            flow.add(wait, name=f"w{i}")
        t = time.perf_counter()
        flow.run(x=1)
        self.assertLess(time.perf_counter() - t, 0.6)

    def test_mixed_sync_and_async_diamond(self):
        flow = Flow()

        @flow.step
        def root(x):
            return x

        @flow.step
        async def left(root):
            await asyncio.sleep(0.1)
            return root + 1

        @flow.step
        def right(root):
            time.sleep(0.1)
            return root + 2

        @flow.step
        def join(left, right):
            return left * right

        t = time.perf_counter()
        self.assertEqual(flow.run(x=1).output, 6)
        self.assertLess(time.perf_counter() - t, 0.19 if os.name != "nt" else 0.4)  # serial would be 0.2s+

    def test_max_workers_limits_threads(self):
        active = []
        peak = [0]
        lock = threading.Lock()
        flow = Flow(max_workers=2)

        def work(x):
            with lock:
                active.append(1)
                peak[0] = max(peak[0], len(active))
            time.sleep(0.05)
            with lock:
                active.pop()
            return x

        for i in range(6):
            flow.add(work, name=f"w{i}")
        flow.run(x=1)
        self.assertLessEqual(peak[0], 2)

    def test_fail_fast_cancels_slow_siblings(self):
        flow = Flow()

        @flow.step
        async def slow():
            await asyncio.sleep(5)

        @flow.step
        def bad():
            raise RuntimeError("stop")

        t = time.perf_counter()
        with self.assertRaises(StepError):
            flow.run()
        self.assertLess(time.perf_counter() - t, 1)


class TestMap(unittest.TestCase):
    def test_map_sync_keeps_order(self):
        flow = Flow()
        flow.add(lambda n: n * n, name="square", map="n")
        self.assertEqual(flow.run(n=list(range(100))).output, [i * i for i in range(100)])

    def test_map_async_with_concurrency_limit(self):
        running = [0]
        peak = [0]
        flow = Flow()

        @flow.step(map="item", max_concurrency=3)
        async def process(item):
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            await asyncio.sleep(0.01)
            running[0] -= 1
            return item * 10

        self.assertEqual(flow.run(item=range(10)).output, [i * 10 for i in range(10)])
        self.assertLessEqual(peak[0], 3)

    def test_map_sync_concurrency_limit(self):
        lock = threading.Lock()
        running = [0]
        peak = [0]
        flow = Flow()

        @flow.step(map="item", max_concurrency=2)
        def process(item):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.01)
            with lock:
                running[0] -= 1
            return item

        flow.run(item=range(10))
        self.assertLessEqual(peak[0], 2)

    def test_map_with_other_params_and_ctx(self):
        flow = Flow()

        @flow.step(map="word")
        def tag(word, prefix, ctx):
            return f"{prefix}{ctx.item}:{word}"

        self.assertEqual(flow.run(word=["a", "b"], prefix="#").output, ["#0:a", "#1:b"])

    def test_map_retry_per_item(self):
        failures = {}
        lock = threading.Lock()
        flow = Flow()

        @flow.step(map="n", retry=Retry(3, backoff=0))
        def maybe(n):
            with lock:
                failures[n] = failures.get(n, 0) + 1
                if n % 2 and failures[n] < 2:
                    raise ConnectionError
            return n

        self.assertEqual(flow.run(n=range(6)).output, list(range(6)))

    def test_map_failure_raises(self):
        flow = Flow()
        flow.add(lambda n: 1 / n, name="inv", map="n")
        with self.assertRaises(StepError) as cm:
            flow.run(n=[1, 2, 0, 4])
        self.assertIsInstance(cm.exception.cause, ZeroDivisionError)

    def test_map_empty_and_not_iterable(self):
        flow = Flow()
        flow.add(lambda n: n, name="each", map="n")
        self.assertEqual(flow.run(n=[]).output, [])
        with self.assertRaises(StepError) as cm:
            flow.run(n=5)
        self.assertIn("iterable", str(cm.exception))

    def test_map_with_timeout_uses_async_path(self):
        flow = Flow()
        flow.add(lambda n: n + 1, name="each", map="n", timeout=1)
        self.assertEqual(flow.run(n=[1, 2]).output, [2, 3])

    def test_fan_out_fan_in(self):
        flow = Flow()

        @flow.step
        def chunks(text):
            return text.split(".")

        @flow.step(map="chunks")
        async def lengths(chunks):
            return len(chunks.strip())

        @flow.step
        def total(lengths):
            return sum(lengths)

        self.assertEqual(flow.run(text="ab. cde. f").output, 6)


class TestRepeat(unittest.TestCase):
    def test_repeat_until_good_enough(self):
        flow = Flow()

        @flow.step(repeat_until=lambda draft: len(draft) >= 5, max_repeats=10)
        def draft(ctx):
            return (ctx.previous or "") + "x"

        r = flow.run()
        self.assertEqual(r.output, "xxxxx")

    def test_repeat_stops_at_max(self):
        flow = Flow()
        counter = []

        @flow.step(repeat_until=lambda v: False, max_repeats=4)
        def never(ctx):
            counter.append(ctx.iteration)
            return ctx.iteration

        self.assertEqual(flow.run().output, 4)
        self.assertEqual(counter, [1, 2, 3, 4])

    def test_repeat_with_async(self):
        flow = Flow()

        @flow.step(repeat_until=lambda v: v > 3)
        async def grow(ctx):
            return (ctx.previous or 0) + 1

        self.assertEqual(flow.run().output, 4)


class TestCheckpoint(unittest.TestCase):
    def test_resume_after_crash(self):
        calls = []
        crash = {"on": True}
        flow = Flow("job")

        @flow.step
        def download(url):
            calls.append("download")
            return {"rows": [1, 2, 3], "tuple": (1, 2)}

        @flow.step
        def process(download):
            calls.append("process")
            if crash["on"]:
                raise RuntimeError("power cut")
            return sum(download["rows"])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "job.ckpt")
            with self.assertRaises(StepError):
                flow.run(url="u", checkpoint=path)
            self.assertTrue(os.path.exists(path))
            crash["on"] = False
            r = flow.run(url="u", checkpoint=path)
            self.assertEqual(r.output, 6)
            self.assertTrue(r.steps["download"].resumed)
            self.assertEqual(r["download"]["tuple"], (1, 2))  # exact types survive
            self.assertEqual(calls, ["download", "process", "process"])
            self.assertFalse(os.path.exists(path), "checkpoint removed after success")

    def test_checkpoint_ignored_for_different_inputs(self):
        calls = []
        flow = Flow("job")
        flow.add(lambda x: calls.append(x) or x, name="a")
        flow.add(lambda a: 1 / 0, name="b")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.ckpt")
            flow.run(x=1, checkpoint=path, raise_errors=False)
            flow.run(x=2, checkpoint=path, raise_errors=False)
            self.assertEqual(calls, [1, 2])

    def test_corrupt_checkpoint_starts_fresh(self):
        flow = Flow("job")
        flow.add(lambda x: x, name="a")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.ckpt")
            with open(path, "w") as fh:
                fh.write("{not json")
            self.assertEqual(flow.run(x=1, checkpoint=Checkpoint(path, keep=True)).output, 1)
            self.assertTrue(os.path.exists(path))


class TestMemory(unittest.TestCase):
    def test_lean_releases_intermediate_values(self):
        flow = Flow()
        flow.add(lambda x: bytearray(1000), name="a")
        flow.add(lambda a: len(a), name="b")
        flow.add(lambda b: b + 1, name="c", keep=True)
        flow.add(lambda c: c * 2, name="d")
        r = flow.run(x=1, lean=True)
        self.assertTrue(r.steps["a"].released)
        self.assertFalse(r.steps["c"].released)
        self.assertEqual(r.output, 2002)
        with self.assertRaises(KeyError) as cm:
            r["a"]
        self.assertIn("lean", str(cm.exception))

    def test_results_freed_without_gc(self):
        class Big:
            pass

        flow = Flow()
        flow.add(lambda: Big(), name="make")
        gc.disable()
        try:
            r = flow.run()
            ref = weakref.ref(r["make"])
            del r
            self.assertIsNone(ref(), "run result must not be kept alive by reference cycles")
        finally:
            gc.enable()


class TestComposition(unittest.TestCase):
    def test_subflow_as_step(self):
        inner = Flow("clean")
        inner.add(lambda text: text.strip().lower(), name="stripped")
        inner.add(lambda stripped, sep=" ": stripped.split(sep), name="tokens")

        outer = Flow("outer")
        outer.add(lambda raw: raw + "  ", name="text")
        outer.add(inner.as_step("tokens"))
        outer.add(lambda tokens: len(tokens), name="count")
        self.assertEqual(outer.run(raw="  Hello World").output, 2)

    def test_add_existing_step_to_another_flow(self):
        a = Flow()

        @a.step(retry=2)
        def inc(x):
            return x + 1

        b = Flow()
        b.add(inc, name="inc2")
        self.assertEqual(b["inc2"].retry.attempts, 2)
        self.assertEqual(b.run(x=1).output, 2)

    def test_prompt_step_with_fake_llm(self):
        from taskpilot import FakeLLM

        llm = FakeLLM(fn=lambda msgs, tools: "SUMMARY of " + msgs[-1]["content"])
        flow = Flow()
        flow.add(lambda url: "long article", name="article")
        flow.prompt("summary", llm, "Summarize: {article}")
        self.assertEqual(flow.run(url="x").output, "SUMMARY of Summarize: long article")

    def test_prompt_json(self):
        from taskpilot import FakeLLM

        llm = FakeLLM(['```json\n{"sentiment": "positive"}\n```'])
        flow = Flow()
        flow.prompt("mood", llm, "Classify: {text}", json=True)
        self.assertEqual(flow.run(text="great").output, {"sentiment": "positive"})


class TestStreamingAndAsync(unittest.TestCase):
    def make(self):
        flow = Flow()
        flow.add(lambda x: x + 1, name="a")
        flow.add(lambda a: a * 2, name="b")
        return flow

    def test_stream_yields_events_then_result(self):
        events = list(self.make().stream(x=1))
        self.assertEqual(events[0].type, "flow_start")
        self.assertEqual(events[-1].type, "flow_end")
        self.assertEqual(events[-1].result.output, 4)
        ends = [e.step for e in events if e.type == "step_end"]
        self.assertEqual(ends, ["a", "b"])

    def test_stream_raises_step_errors(self):
        flow = Flow()
        flow.add(lambda: 1 / 0, name="bad")
        with self.assertRaises(StepError):
            list(flow.stream())

    def test_astream(self):
        async def main():
            return [e async for e in self.make().astream(x=1)]

        events = asyncio.run(main())
        self.assertEqual(events[-1].result.output, 4)

    def test_arun_inside_user_event_loop(self):
        flow = self.make()

        async def main():
            results = await asyncio.gather(*(flow.arun(x=i) for i in range(20)))
            return [r.output for r in results]

        self.assertEqual(asyncio.run(main()), [(i + 1) * 2 for i in range(20)])

    def test_sync_run_called_inside_running_loop(self):
        flow = self.make()

        async def main():
            return flow.run(x=1).output  # like calling from Jupyter

        self.assertEqual(asyncio.run(main()), 4)

    def test_step_can_use_asyncio_run_and_nested_flows(self):
        inner = self.make()
        flow = Flow()

        @flow.step
        def uses_asyncio(x):
            async def double():
                return x * 2

            return asyncio.run(double())

        @flow.step
        def uses_flow(uses_asyncio):
            return inner.run(x=uses_asyncio).output

        self.assertEqual(flow.run(x=1).output, 6)

    def test_recursive_same_flow_in_threads_does_not_deadlock(self):
        flow = Flow(max_workers=2)

        @flow.step
        def a(depth):
            return depth

        @flow.step
        def b(depth):
            if depth <= 0:
                return 0
            return flow.run(depth=depth - 1).output["b"] + 1

        t = time.perf_counter()
        self.assertEqual(flow.run(depth=3).output["b"], 3)
        self.assertLess(time.perf_counter() - t, 5)

    def test_many_threads_running_same_flow(self):
        flow = self.make()
        out = []
        lock = threading.Lock()

        def worker(i):
            for j in range(50):
                v = flow.run(x=i * 100 + j).output
                with lock:
                    out.append(v == (i * 100 + j + 1) * 2)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(out), 400)
        self.assertTrue(all(out))


if __name__ == "__main__":
    unittest.main()
