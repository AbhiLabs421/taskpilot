"""EASY: the basics every user touches on day one."""

import asyncio
import unittest

from taskpilot import Flow, RunResult, Step


class TestBasics(unittest.TestCase):
    def test_single_step(self):
        flow = Flow("one")

        @flow.step
        def double(x):
            return x * 2

        self.assertEqual(flow.run(x=21).output, 42)

    def test_steps_wired_by_parameter_name(self):
        flow = Flow("greet")

        @flow.step
        def name(raw):
            return raw.strip().title()

        @flow.step
        def hello(name):
            return f"Hello, {name}!"

        result = flow.run(raw="  asha ")
        self.assertEqual(result.output, "Hello, Asha!")
        self.assertEqual(result["name"], "Asha")
        self.assertTrue(result.ok)

    def test_definition_order_does_not_matter(self):
        flow = Flow()

        @flow.step
        def b(a):
            return a + 1

        @flow.step
        def a(x):
            return x * 10

        self.assertEqual(flow.run(x=1).output, 11)
        self.assertEqual(flow.plan().order, ("a", "b"))

    def test_inputs_as_dict_or_keywords(self):
        flow = Flow()
        flow.add(lambda a, b: a + b, name="add")
        self.assertEqual(flow.run({"a": 1}, b=2).output, 3)
        self.assertEqual(flow.run({"a": 1, "b": 2}).output, 3)

    def test_default_parameter_used_when_input_missing(self):
        flow = Flow()

        @flow.step
        def greet(name, greeting="Namaste"):
            return f"{greeting}, {name}"

        self.assertEqual(flow.run(name="Ravi").output, "Namaste, Ravi")
        self.assertEqual(flow.run(name="Ravi", greeting="Hi").output, "Hi, Ravi")

    def test_async_step(self):
        flow = Flow()

        @flow.step
        async def fetch(x):
            await asyncio.sleep(0.001)
            return x + 1

        @flow.step
        def use(fetch):
            return fetch * 2

        self.assertEqual(flow.run(x=1).output, 4)

    def test_step_named_like_its_input(self):
        flow = Flow()

        @flow.step
        def text(text):
            return text.strip()

        @flow.step
        def words(text):
            return text.split()

        self.assertEqual(flow.run(text="  a b ").output, ["a", "b"])

    def test_multiple_outputs_become_dict(self):
        flow = Flow()
        flow.add(lambda x: x + 1, name="plus")
        flow.add(lambda x: x - 1, name="minus")
        self.assertEqual(flow.run(x=5).output, {"plus": 6, "minus": 4})

    def test_decorator_returns_step_that_is_still_callable(self):
        flow = Flow()

        @flow.step
        def square(x):
            return x * x

        self.assertIsInstance(square, Step)
        self.assertEqual(square(x=3), 9)

    def test_decorator_with_name(self):
        flow = Flow()

        @flow.step("total")
        def compute(x):
            return x

        @flow.step(name="shown")
        def other(total):
            return total

        self.assertEqual(list(flow.steps), ["total", "shown"])

    def test_empty_flow(self):
        result = Flow().run()
        self.assertIsNone(result.output)
        self.assertTrue(result.ok)

    def test_result_helpers(self):
        flow = Flow("r")
        flow.add(lambda x: x, name="a")
        r = flow.run(x=1)
        self.assertIsInstance(r, RunResult)
        self.assertEqual(r.values, {"a": 1})
        self.assertEqual(r.get("missing", "d"), "d")
        self.assertIn("a", r)
        self.assertIn("a", r.summary())
        self.assertEqual(r.to_dict()["steps"]["a"]["status"], "success")
        self.assertEqual(r.succeeded, ["a"])

    def test_chain(self):
        def load(path):
            return [3, 1, 2]

        def sort_it(data):
            return sorted(data)

        def first(items):
            return items[0]

        pipeline = Flow.chain(load, sort_it, first)
        self.assertEqual(pipeline.run(path="x").output, 1)

    def test_describe_and_mermaid(self):
        flow = Flow("d")
        flow.add(lambda x: x, name="a")
        flow.add(lambda a: a, name="b", retry=3)
        self.assertIn("b <- a", flow.describe())
        self.assertIn("retry=3", flow.describe())
        mm = flow.to_mermaid()
        self.assertIn("a --> b", mm)
        self.assertIn("in_x --> a", mm)

    def test_inputs_listing(self):
        flow = Flow()
        flow.add(lambda a, b=1: a, name="s")
        self.assertEqual(flow.inputs, ["a", "b"])
        self.assertEqual(flow.required_inputs, ["a"])

    def test_remove_and_contains(self):
        flow = Flow()
        flow.add(lambda x: x, name="a")
        self.assertIn("a", flow)
        flow.remove("a")
        self.assertNotIn("a", flow)
        self.assertEqual(len(flow), 0)


if __name__ == "__main__":
    unittest.main()
