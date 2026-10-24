"""MEDIUM: reliability features – retry, timeout, cache, branching, errors."""

import asyncio
import threading
import time
import unittest

from taskpilot import (
    Cache,
    CycleError,
    Flow,
    FlowDefinitionError,
    MissingInputError,
    Retry,
    Skip,
    StepError,
    StepTimeout,
)


class TestErrorsAreClear(unittest.TestCase):
    def test_missing_input_names_step_and_suggests(self):
        flow = Flow()
        flow.add(lambda subject: subject, name="topic")
        with self.assertRaises(MissingInputError) as cm:
            flow.run(subjct="ai")
        msg = str(cm.exception)
        self.assertIn("'subject'", msg)
        self.assertIn("'topic'", msg)
        self.assertIn("Did you mean 'subjct'", msg)

    def test_cycle_is_reported_with_path(self):
        flow = Flow()
        flow.add(lambda b: b, name="a")
        flow.add(lambda c: c, name="b")
        flow.add(lambda a: a, name="c")
        with self.assertRaises(CycleError) as cm:
            flow.validate()
        self.assertEqual(len(cm.exception.cycle), 4)
        self.assertEqual(cm.exception.cycle[0], cm.exception.cycle[-1])

    def test_duplicate_step_name(self):
        flow = Flow()
        flow.add(lambda: 1, name="a")
        with self.assertRaises(FlowDefinitionError):
            flow.add(lambda: 2, name="a")

    def test_bad_options_fail_at_definition_time(self):
        flow = Flow()
        with self.assertRaises(FlowDefinitionError):
            flow.add(lambda x: x, name="a", map="nope")
        with self.assertRaises(FlowDefinitionError):
            flow.add(lambda x: x, name="b", on_error="explode")
        with self.assertRaises(FlowDefinitionError):
            flow.add(lambda x: x, name="c", timeout=-1)
        with self.assertRaises(FlowDefinitionError):
            flow.add(lambda x: x, name="not valid")
        with self.assertRaises(FlowDefinitionError):
            flow.add(lambda x: x, name="d", retry="lots")

    def test_unknown_after(self):
        flow = Flow()
        flow.add(lambda: 1, name="setup")
        flow.add(lambda: 2, name="work", after=["setpu"])
        with self.assertRaises(FlowDefinitionError) as cm:
            flow.validate()
        self.assertIn("Did you mean 'setup'", str(cm.exception))

    def test_step_error_carries_partial_result(self):
        flow = Flow()
        flow.add(lambda x: x + 1, name="ok")

        @flow.step
        def boom(ok):
            raise ValueError("bad value")

        with self.assertRaises(StepError) as cm:
            flow.run(x=1)
        err = cm.exception
        self.assertEqual(err.step, "boom")
        self.assertIsInstance(err.cause, ValueError)
        self.assertEqual(err.result["ok"], 2)
        self.assertEqual(err.result.failed, ["boom"])

    def test_raise_errors_false_returns_failed_result(self):
        flow = Flow()
        flow.add(lambda: 1 / 0, name="bad")
        flow.add(lambda: "fine", name="good")
        r = flow.run(raise_errors=False)
        self.assertFalse(r.ok)
        self.assertEqual(r.failed, ["bad"])
        self.assertEqual(r["good"], "fine")
        self.assertIsInstance(r.steps["bad"].error, ZeroDivisionError)


class TestRetry(unittest.TestCase):
    def test_retry_until_success(self):
        calls = []
        flow = Flow()

        @flow.step(retry=Retry(4, backoff=0))
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("down")
            return "up"

        r = flow.run()
        self.assertEqual(r.output, "up")
        self.assertEqual(r.steps["flaky"].attempts, 3)

    def test_retry_gives_up(self):
        flow = Flow()
        flow.add(lambda: 1 / 0, name="bad", retry=Retry(3, backoff=0))
        with self.assertRaises(StepError) as cm:
            flow.run()
        self.assertEqual(cm.exception.attempts, 3)

    def test_retry_only_on_listed_errors(self):
        calls = []
        flow = Flow()

        @flow.step(retry=Retry(5, backoff=0, on=ConnectionError))
        def bad():
            calls.append(1)
            raise KeyError("not retryable")

        with self.assertRaises(StepError):
            flow.run()
        self.assertEqual(len(calls), 1)

    def test_retry_int_shorthand_and_async(self):
        calls = []
        flow = Flow()

        @flow.step(retry=Retry(2, backoff=0))
        async def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError
            return "ok"

        self.assertEqual(flow.run().output, "ok")
        self.assertEqual(Retry.coerce(5).attempts, 5)

    def test_retry_events(self):
        seen = []
        flow = Flow()
        state = {"n": 0}

        @flow.step(retry=Retry(3, backoff=0))
        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise ValueError("x")
            return 1

        flow.on("step_retry", lambda e: seen.append(e.attempt))
        flow.run()
        self.assertEqual(seen, [1, 2])

    def test_backoff_grows_and_caps(self):
        r = Retry(10, backoff=1, factor=2, max_delay=5, jitter=0)
        self.assertEqual([r.delay(i) for i in range(1, 6)], [1, 2, 4, 5, 5])


class TestTimeout(unittest.TestCase):
    def test_sync_timeout(self):
        flow = Flow()
        flow.add(lambda: time.sleep(2), name="slow", timeout=0.05)
        t = time.perf_counter()
        with self.assertRaises(StepError) as cm:
            flow.run()
        self.assertLess(time.perf_counter() - t, 1.0)
        self.assertIsInstance(cm.exception.cause, StepTimeout)

    def test_async_timeout_with_retry(self):
        flow = Flow()

        @flow.step(timeout=0.02, retry=Retry(2, backoff=0))
        async def slow():
            await asyncio.sleep(1)

        with self.assertRaises(StepError) as cm:
            flow.run()
        self.assertEqual(cm.exception.attempts, 2)

    def test_flow_timeout(self):
        from taskpilot import FlowTimeout

        flow = Flow(timeout=0.05)

        @flow.step
        async def slow():
            await asyncio.sleep(1)

        with self.assertRaises(FlowTimeout):
            flow.run()
        r = flow.run(raise_errors=False)
        self.assertIsInstance(r.error, FlowTimeout)
        self.assertFalse(r.ok)


class TestCache(unittest.TestCase):
    def test_cache_reuses_results(self):
        calls = []
        flow = Flow()

        @flow.step(cache=True)
        def expensive(x):
            calls.append(x)
            return x * 2

        self.assertEqual(flow.run(x=2).output, 4)
        r = flow.run(x=2)
        self.assertEqual(r.output, 4)
        self.assertTrue(r.steps["expensive"].cached)
        flow.run(x=3)
        self.assertEqual(calls, [2, 3])

    def test_cache_ttl(self):
        c = Cache(ttl=0.01)
        c.set("k", 1)
        self.assertEqual(c.get("k"), 1)
        time.sleep(0.02)
        from taskpilot.policies import MISSING

        self.assertIs(c.get("k"), MISSING)

    def test_cache_lru_limit(self):
        c = Cache(maxsize=2)
        c.set("a", 1)
        c.set("b", 2)
        c.get("a")
        c.set("c", 3)
        self.assertEqual(len(c), 2)
        from taskpilot.policies import MISSING

        self.assertIs(c.get("b"), MISSING)
        self.assertEqual(c.get("a"), 1)

    def test_unpicklable_args_still_work(self):
        flow = Flow()
        lock = threading.Lock()
        flow.add(lambda thing: "ok", name="s", cache=True)
        self.assertEqual(flow.run(thing=lock).output, "ok")


class TestBranching(unittest.TestCase):
    def make_router(self):
        flow = Flow("router")

        @flow.step
        def kind(question):
            return "math" if any(c.isdigit() for c in question) else "chat"

        @flow.step(when=lambda kind: kind == "math")
        def math_answer(question):
            return eval(question, {"__builtins__": {}})  # test input only

        @flow.step(when=lambda kind: kind == "chat")
        def chat_answer(question):
            return f"Let's talk about {question}"

        @flow.step
        def final(math_answer=None, chat_answer=None):
            return math_answer if math_answer is not None else chat_answer

        return flow

    def test_when_routes(self):
        flow = self.make_router()
        r = flow.run(question="2+3")
        self.assertEqual(r.output, 5)
        self.assertEqual(r.skipped, ["chat_answer"])
        self.assertEqual(flow.run(question="cats").output, "Let's talk about cats")

    def test_skip_propagates_to_dependents_without_defaults(self):
        flow = Flow()
        flow.add(lambda: 1, name="a", when=lambda: False)
        flow.add(lambda a: a, name="b")
        flow.add(lambda b: b, name="c")
        r = flow.run()
        self.assertEqual(r.skipped, ["a", "b", "c"])
        self.assertIn("needs 'a'", r.steps["b"].reason)
        self.assertTrue(r.ok)

    def test_raise_skip_inside_step(self):
        flow = Flow()

        @flow.step
        def email(user):
            if not user.get("email"):
                raise Skip("no email")
            return "sent"

        r = flow.run(user={})
        self.assertEqual(r.steps["email"].reason, "no email")
        self.assertIsNone(r.output)

    def test_on_error_skip_and_continue(self):
        flow = Flow()
        flow.add(lambda: 1 / 0, name="bad", on_error="skip")
        flow.add(lambda bad="default": bad, name="after_bad")
        r = flow.run()
        self.assertTrue(r.ok)
        self.assertEqual(r.output, "default")

        flow2 = Flow()
        flow2.add(lambda: 1 / 0, name="bad", on_error="continue")
        flow2.add(lambda: "independent", name="other")
        r2 = flow2.run()
        self.assertFalse(r2.ok)
        self.assertEqual(r2["other"], "independent")

    def test_fallback_value_and_function(self):
        flow = Flow()
        flow.add(lambda: 1 / 0, name="a", fallback=0)
        flow.add(lambda: [][1], name="b", fallback=lambda exc: type(exc).__name__)
        r = flow.run()
        self.assertEqual(r["a"], 0)
        self.assertEqual(r["b"], "IndexError")
        self.assertTrue(r.steps["a"].fallback)
        self.assertTrue(r.ok)

    def test_after_orders_without_data(self):
        order = []
        flow = Flow()
        flow.add(lambda: order.append("second"), name="work", after=["setup"])
        flow.add(lambda: order.append("first"), name="setup")
        flow.run()
        self.assertEqual(order, ["first", "second"])

    def test_bind_maps_parameter(self):
        flow = Flow()
        flow.add(lambda x: x * 3, name="tripled")
        flow.add(lambda value: value + 1, name="plus", bind={"value": "tripled"})
        self.assertEqual(flow.run(x=2).output, 7)


class TestContextAndEvents(unittest.TestCase):
    def test_ctx_shared_state(self):
        flow = Flow()

        @flow.step
        def a(ctx):
            ctx.set("seen", ["a"])
            return 1

        @flow.step
        def b(a, ctx):
            ctx.update("seen", lambda s: s + ["b"])
            return ctx.step

        r = flow.run(state={"start": True})
        self.assertEqual(r.state, {"start": True, "seen": ["a", "b"]})
        self.assertEqual(r["b"], "b")

    def test_events_in_order(self):
        flow = Flow()
        flow.add(lambda: 1, name="a")
        flow.add(lambda a: 2, name="b")
        types = []
        flow.on("*", lambda e: types.append((e.type, e.step)))
        flow.run()
        self.assertEqual(types[0], ("flow_start", None))
        self.assertEqual(types[-1], ("flow_end", None))
        self.assertLess(types.index(("step_end", "a")), types.index(("step_start", "b")))

    def test_custom_events_and_logs(self):
        flow = Flow()

        @flow.step
        def work(ctx):
            ctx.log("halfway", pct=50)
            ctx.emit("progress", pct=100)

        seen = []
        flow.on("log", lambda e: seen.append(e.message))
        flow.on("progress", lambda e: seen.append(e.pct))
        flow.run()
        self.assertEqual(seen, ["halfway", 100])

    def test_broken_listener_does_not_break_run(self):
        flow = Flow()
        flow.add(lambda: 1, name="a")
        flow.on("step_end", lambda e: 1 / 0)
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(flow.run().output, 1)

    def test_on_as_decorator_and_off(self):
        flow = Flow()
        flow.add(lambda: 1, name="a")
        seen = []

        @flow.on("step_end")
        def h(e):
            seen.append(e.step)

        flow.run()
        flow.off("step_end", h)
        flow.run()
        self.assertEqual(seen, ["a"])

    def test_verbose_prints_trace(self):
        import contextlib
        import io

        flow = Flow("loud")
        flow.add(lambda: 1, name="a")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            flow.run(verbose=True)
        self.assertIn("loud", buf.getvalue())
        self.assertIn("a", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
