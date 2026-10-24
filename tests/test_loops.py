"""Multi-step loops (flow.loop), from easy to complex."""

import asyncio
import os
import random
import tempfile
import threading
import time
import unittest

from taskpilot import FakeLLM, Flow, FlowDefinitionError, LoopLimit, Retry, StepError


def essay_flow(target=8, step=3, max_iterations=5, on_max="stop"):
    flow = Flow("essay")

    @flow.step
    def draft(topic, feedback=None, ctx=None):
        return f"{topic} v{ctx.iteration}" + (f" +{feedback['note']}" if feedback else "")

    @flow.step
    def review(draft, ctx):
        return {"score": step * ctx.iteration, "note": f"r{ctx.iteration}"}

    @flow.step
    def publish(draft, review):
        return f"{draft} [{review['score']}]"

    flow.loop("draft", "review", until=lambda review: review["score"] >= target,
              feed={"feedback": "review"}, max_iterations=max_iterations, on_max=on_max)
    return flow


class Easy(unittest.TestCase):
    def test_loop_until_good(self):
        r = essay_flow().run(topic="AI")
        self.assertEqual(r.output, "AI v3 +r2 [9]")
        self.assertEqual(r.loops["draft_loop"], {"rounds": 3, "stopped_by": "until"})

    def test_history_keeps_every_round(self):
        r = essay_flow().run(topic="AI")
        self.assertEqual([h["score"] for h in r.history["review"]], [3, 6, 9])
        self.assertEqual(r.history["draft"][0], "AI v1")

    def test_first_round_is_enough(self):
        r = essay_flow(target=1).run(topic="AI")
        self.assertEqual(r.loops["draft_loop"]["rounds"], 1)
        self.assertEqual(r.output, "AI v1 [3]")

    def test_reviewer_is_not_an_output_when_steps_follow_the_loop(self):
        r = essay_flow().run(topic="AI")
        self.assertEqual(r.sinks, ("publish",))
        self.assertIsInstance(r.output, str)
        alone = Flow()
        alone.add(lambda n=0: n + 1, name="a")
        alone.add(lambda a: a, name="b")
        alone.loop("a", "b", until=lambda b: b >= 2, feed={"n": "b"})
        self.assertEqual(alone.run().output, 2, "with nothing after the loop, its end is the output")

    def test_named_loop_and_describe(self):
        flow = Flow()
        flow.add(lambda n=0: n + 1, name="inc")
        flow.add(lambda inc: inc, name="check")
        flow.loop("inc", "check", until=lambda check: check >= 3, feed={"n": "check"}, name="counter")
        self.assertEqual(flow.run().output, 3)
        self.assertIn("loop counter: inc -> check", flow.describe())
        self.assertIn('check -. "repeat until" .-> inc', flow.to_mermaid())


class Medium(unittest.TestCase):
    def test_max_iterations_stop_keeps_last_values(self):
        r = essay_flow(target=100, max_iterations=4).run(topic="AI")
        self.assertEqual(r.loops["draft_loop"], {"rounds": 4, "stopped_by": "max_iterations"})
        self.assertEqual(r["review"]["score"], 12)

    def test_max_iterations_raise(self):
        with self.assertRaises(StepError) as cm:
            essay_flow(target=100, max_iterations=2, on_max="raise").run(topic="AI")
        self.assertIsInstance(cm.exception.cause, LoopLimit)

    def test_until_can_read_inputs(self):
        flow = Flow()
        flow.add(lambda total=0: total + 5, name="add")
        flow.add(lambda add: add, name="seen")
        flow.loop("add", "seen", until=lambda seen, goal: seen >= goal, feed={"total": "seen"})
        self.assertEqual(flow.run(goal=17).output, 20)

    def test_until_error_fails_clearly(self):
        flow = Flow()
        flow.add(lambda: 1, name="a")
        flow.add(lambda a: a, name="b")
        flow.loop("a", "b", until=lambda b: b["missing"])
        with self.assertRaises(StepError) as cm:
            flow.run()
        self.assertIsInstance(cm.exception.cause, TypeError)

    def test_definition_errors(self):
        def base():
            f = Flow()
            f.add(lambda x, fb=None: x, name="a")
            f.add(lambda a: a, name="b")
            f.add(lambda b: b, name="c")
            f.add(lambda x: x, name="side")
            return f

        cases = [
            (dict(start="a", end="nope", until=lambda b: True), "no such step"),
            (dict(start="b", end="a", until=lambda b: True), "does not come after"),
            (dict(start="a", end="side", until=lambda b: True), "does not come after"),
            (dict(start="a", end="b", until=lambda c: True), "only runs after the loop"),
            (dict(start="a", end="b", until=lambda b: True, feed={"zzz": "b"}), "no step in the loop"),
            (dict(start="a", end="b", until=lambda b: True, feed={"fb": "nope"}), "inside or before"),
            (dict(start="a", end="b", until=lambda b: True, max_iterations=0), "max_iterations"),
            (dict(start="a", end="b", until=lambda b: True, on_max="explode"), "on_max"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                f = base()
                with self.assertRaises(FlowDefinitionError) as cm:
                    f.loop(**kwargs)
                    f.validate()
                self.assertIn(message, str(cm.exception))

    def test_feed_target_needs_default(self):
        f = Flow()
        f.add(lambda x, fb: x, name="a")
        f.add(lambda a: a, name="b")
        f.loop("a", "b", until=lambda b: True, feed={"fb": "b"})
        with self.assertRaises(FlowDefinitionError) as cm:
            f.validate()
        self.assertIn("give it a default", str(cm.exception))

    def test_overlapping_loops_rejected(self):
        f = Flow()
        f.add(lambda: 1, name="a")
        f.add(lambda a: a, name="b")
        f.add(lambda b: b, name="c")
        f.loop("a", "b", until=lambda b: True)
        f.loop("b", "c", until=lambda c: True)
        with self.assertRaises(FlowDefinitionError) as cm:
            f.validate()
        self.assertIn("can't overlap", str(cm.exception))

    def test_skipped_end_leaves_loop(self):
        flow = Flow()
        flow.add(lambda: 1, name="a")
        flow.add(lambda a: a, name="b", when=lambda a: False)
        flow.add(lambda b=None: "after", name="c")
        flow.loop("a", "b", until=lambda b: False, max_iterations=5)
        r = flow.run()
        self.assertEqual(r.output, "after")
        self.assertEqual(r.loops["a_loop"]["rounds"], 1)


class High(unittest.TestCase):
    def test_parallel_and_async_steps_inside_loop(self):
        flow = Flow("checks")

        @flow.step
        def draft(feedback=None, ctx=None):
            return ctx.iteration

        @flow.step
        async def grammar(draft):
            await asyncio.sleep(0.05)
            return draft >= 2

        @flow.step
        def facts(draft):
            time.sleep(0.05)
            return draft >= 3

        @flow.step
        def review(grammar, facts):
            return grammar and facts

        flow.loop("draft", "review", until=lambda review: review, feed={"feedback": "review"})
        t = time.perf_counter()
        r = flow.run()
        self.assertEqual(r.loops["draft_loop"]["rounds"], 3)
        self.assertLess(time.perf_counter() - t, 0.4, "grammar and facts run in parallel each round")

    def test_steps_outside_loop_wait_for_it(self):
        order = []
        lock = threading.Lock()
        flow = Flow()

        @flow.step
        def a(n=0):
            return n + 1

        @flow.step
        def side(a):
            with lock:
                order.append(("side", a))  # reads a mid-loop step, but only after the loop
            return a

        @flow.step
        def b(a):
            return a

        flow.loop("a", "b", until=lambda b: b >= 3, feed={"n": "b"})
        r = flow.run()
        self.assertEqual(order, [("side", 3)])
        self.assertEqual(r["side"], 3)

    def test_map_and_retry_inside_loop(self):
        calls = {"n": 0}
        lock = threading.Lock()
        flow = Flow()
        flow.add(lambda size=1: list(range(size)), name="batch")

        @flow.step(map="batch", retry=Retry(3, backoff=0))
        def work(batch):
            with lock:
                calls["n"] += 1
                fail = calls["n"] == 2
            if fail:
                raise ConnectionError("blip")
            return batch * 2

        flow.add(lambda work: len(work) + 1, name="total")
        flow.loop("batch", "total", until=lambda total: total > 4, feed={"size": "total"})
        r = flow.run()
        # sizes 1, 2, 3, 4 -> totals 2, 3, 4, 5; stops when total > 4
        self.assertEqual(r["work"], [0, 2, 4, 6])
        self.assertEqual(r.loops["batch_loop"]["rounds"], 4)
        self.assertEqual(calls["n"], 1 + 2 + 3 + 4 + 1, "one retried item")

    def test_checkpoint_resume_reuses_finished_loop(self):
        calls = {"draft": 0}
        crash = {"on": True}
        flow = Flow("ck")

        @flow.step
        def draft(feedback=None, ctx=None):
            calls["draft"] += 1
            return ctx.iteration

        @flow.step
        def review(draft):
            return draft

        @flow.step
        def publish(review):
            if crash["on"]:
                raise RuntimeError("network down")
            return f"published {review}"

        flow.loop("draft", "review", until=lambda review: review >= 3, feed={"feedback": "review"})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.json")
            with self.assertRaises(StepError):
                flow.run(checkpoint=path)
            crash["on"] = False
            r = flow.run(checkpoint=path)
        self.assertEqual(r.output, "published 3")
        self.assertEqual(calls["draft"], 3, "the loop did not run again on resume")

    def test_crash_inside_loop_reruns_loop_from_start(self):
        boom = {"on": True}
        flow = Flow("ck2")
        flow.add(lambda x: x, name="prep")

        @flow.step
        def step_a(prep, n=0):
            return n + 1

        @flow.step
        def step_b(step_a):
            if boom["on"] and step_a == 2:
                raise RuntimeError("crash in round 2")
            return step_a

        flow.loop("step_a", "step_b", until=lambda step_b: step_b >= 3, feed={"n": "step_b"})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.json")
            with self.assertRaises(StepError):
                flow.run(x=1, checkpoint=path)
            boom["on"] = False
            r = flow.run(x=1, checkpoint=path)
        self.assertTrue(r.steps["prep"].resumed)
        self.assertEqual(r.output, 3)

    def test_lean_and_stream_with_loops(self):
        flow = essay_flow()
        r = flow.run(topic="X", lean=True)
        self.assertEqual(r.output, "X v3 +r2 [9]")
        rounds = [e.data["round"] for e in flow.stream(topic="X") if e.type == "loop_round"]
        self.assertEqual(rounds, [1, 2, 3])

    def test_two_loops_one_after_another(self):
        flow = Flow()
        flow.add(lambda n=0: n + 1, name="a1")
        flow.add(lambda a1: a1, name="a2")
        flow.add(lambda a2, m=0: a2 + m + 10, name="b1")
        flow.add(lambda b1: b1, name="b2")
        flow.loop("a1", "a2", until=lambda a2: a2 >= 2, feed={"n": "a2"})
        flow.loop("b1", "b2", until=lambda b2: b2 >= 30, feed={"m": "b2"})
        r = flow.run()
        self.assertEqual(r.loops["a1_loop"]["rounds"], 2)
        self.assertEqual(r.output, 36)  # 12, 2+12+10=24, 2+24+10=36


class Complex(unittest.TestCase):
    def test_evaluator_optimizer_with_llm(self):
        def brain(messages, tools):
            text = messages[-1]["content"]
            if text.startswith("Write"):
                return "short poem" if "feedback" not in text else "a much longer and vivid poem"
            return '{"ok": true}' if "vivid" in text else '{"ok": false, "tip": "be vivid"}'

        llm = FakeLLM(fn=brain)
        flow = Flow("poet")

        @flow.step
        def poem(theme, review=None):
            hint = f" feedback: {review['tip']}" if review else ""
            return llm.ask(f"Write a poem about {theme}.{hint}")

        @flow.step
        def judge(poem):
            return llm.json(f"Judge this poem: {poem}")

        flow.loop("poem", "judge", until=lambda judge: judge["ok"], feed={"review": "judge"})
        r = flow.run(theme="monsoon")
        self.assertEqual(r["poem"], "a much longer and vivid poem")
        self.assertEqual(r.loops["poem_loop"]["rounds"], 2)

    def test_random_counting_loops_match_reference(self):
        rnd = random.Random(5)
        for case in range(40):
            step = rnd.randint(1, 5)
            goal = rnd.randint(1, 40)
            limit = rnd.randint(1, 15)
            flow = Flow(f"c{case}")
            flow.add(lambda total=0, step=step: total + step, name="add")
            flow.add(lambda add: add, name="check")
            flow.loop("add", "check", until=lambda check, goal=goal: check >= goal, feed={"total": "check"},
                      max_iterations=limit)
            r = flow.run()
            rounds = min(limit, -(-goal // step))
            self.assertEqual(r.output, rounds * step, (step, goal, limit))
            self.assertEqual(r.loops["add_loop"]["rounds"], rounds)

    def test_many_threads_run_loop_flows(self):
        flow = essay_flow()
        results = []
        lock = threading.Lock()

        def worker(i):
            for _ in range(25):
                out = flow.run(topic=f"T{i}").output
                with lock:
                    results.append(out == f"T{i} v3 +r2 [9]")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 150)
        self.assertTrue(all(results))

    def test_async_entrypoint_with_loop(self):
        async def main():
            return await asyncio.gather(*(essay_flow().arun(topic=str(i)) for i in range(20)))

        outs = [r.output for r in asyncio.run(main())]
        self.assertEqual(outs, [f"{i} v3 +r2 [9]" for i in range(20)])


if __name__ == "__main__":
    unittest.main()
