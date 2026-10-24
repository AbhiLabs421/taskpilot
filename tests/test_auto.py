"""AutoFlow: plain-language request -> planned and executed workflow. Easy to complex."""

import asyncio
import contextlib
import io
import json
import os
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from taskpilot import Agent, AutoFlow, FakeLLM, Flow, FlowDefinitionError, TaskPilotError, tool
from taskpilot.web import WebApp


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}: {body}"


def plan_json(*steps, answer=None, reason=None):
    d = {"steps": [dict(id=s[0], tool=s[1], args=s[2]) for s in steps]}
    if answer:
        d["answer"] = answer
    if reason:
        d["reason"] = reason
    return json.dumps(d)


CHAIN = plan_json(("s1", "add_numbers", {"a": 125, "b": 75}), ("s2", "subtract_numbers", {"a": "{{s1}}", "b": 40}))


def make(replies, **kw):
    llm = FakeLLM(replies)
    return AutoFlow(llm, [add_numbers, subtract_numbers, send_email], **kw), llm


class Easy(unittest.TestCase):
    def test_plan_and_run(self):
        auto, _ = make([CHAIN])
        r = auto.run("add 125 and 75, then take away 40")
        self.assertTrue(r.ok)
        self.assertEqual(r.output, 160)
        self.assertEqual([s.tool for s in r.plan.steps], ["add_numbers", "subtract_numbers"])

    def test_describe_in_plain_words(self):
        auto, _ = make([CHAIN])
        text = auto.plan("x").describe()
        self.assertIn("1. s1: add_numbers(a=125, b=75)  (can start right away)", text)
        self.assertIn("subtract_numbers(a=<result of s1>, b=40)  (after s1)", text)
        self.assertIn("Answer: result of s2", text)

    def test_references_keep_types_and_fill_text(self):
        auto, _ = make([plan_json(("s1", "add_numbers", {"a": 2, "b": 3}),
                                  ("s2", "send_email", {"to": "x@y.z", "body": "Total is {{s1}} rupees"}))])
        r = auto.run("x")
        self.assertEqual(r.result["s1"], 5)
        self.assertEqual(r.output, "sent to x@y.z: Total is 5 rupees")

    def test_the_model_is_told_about_tools(self):
        auto, llm = make([CHAIN])
        auto.plan("x")
        system = llm.calls[0]["messages"][0]["content"]
        for name in ("add_numbers", "subtract_numbers", "send_email", "ask_ai", "ask_human"):
            self.assertIn(name, system)


class Medium(unittest.TestCase):
    def test_bad_plan_is_repaired(self):
        bad = plan_json(("s1", "multiply", {"a": 1, "b": 2}))
        auto, llm = make([bad, CHAIN])
        r = auto.run("x")
        self.assertEqual(r.output, 160)
        self.assertIn("unknown tool 'multiply'", llm.calls[1]["messages"][-1]["content"])

    def test_problems_are_found(self):
        auto, _ = make([])
        cases = [
            (plan_json(("s1", "add_numbers", {"a": 1})), "needs argument 'b'"),
            (plan_json(("s1", "add_numbers", {"a": 1, "b": 2, "c": 3})), "no argument 'c'"),
            (plan_json(("s1", "add_numbers", {"a": "{{s2}}", "b": 2}), ("s2", "add_numbers", {"a": 1, "b": 1})), "earlier step"),
            (plan_json(("s1", "add_numbers", {"a": 1, "b": 2}), ("s1", "add_numbers", {"a": 1, "b": 2})), "used twice"),
            (plan_json(("Bad Id", "add_numbers", {"a": 1, "b": 2})), "lowercase"),
            (plan_json(("ctx", "add_numbers", {"a": 1, "b": 2})), "lowercase"),
            (plan_json(("s1", "add_numbers", {"a": 1, "b": 2}), answer="s9"), "not a step id"),
            (json.dumps({"nope": 1}), "missing required key 'steps'"),
        ]
        for raw, message in cases:
            with self.subTest(message=message):
                problems = auto.check(json.loads(raw))
                self.assertTrue(any(message in p for p in problems), problems)

    def test_gives_up_after_repairs(self):
        bad = plan_json(("s1", "nope", {}))
        auto, _ = make([bad, bad, bad], repair=2)
        with self.assertRaises(TaskPilotError) as cm:
            auto.plan("x")
        self.assertIn("Could not make a plan that passes its checks", str(cm.exception))

    def test_impossible_request(self):
        auto, _ = make([json.dumps({"steps": [], "reason": "no tool can book flights"})])
        with self.assertRaises(TaskPilotError) as cm:
            auto.plan("book a flight")
        self.assertIn("book flights", str(cm.exception))

    def test_approval_is_enforced_even_if_the_model_forgot(self):
        auto, _ = make([plan_json(("s1", "send_email", {"to": "boss@x.com", "body": "hi"}))], approve=["send_email"])
        r = auto.run("email the boss")
        self.assertTrue(r.paused)
        self.assertEqual(r.pending[0].key, "approve_s1")
        self.assertIn("send_email", r.pending[0].question)
        self.assertIn("[needs approval]", r.plan.describe())
        self.assertEqual(r.resume({"approve_s1": "yes"}).output, "sent to boss@x.com: hi")
        again = r.plan.run()
        rejected = r.plan.resume(again, approve_s1=False)
        self.assertEqual(rejected.steps["s1"].status, "skipped")
        self.assertIn("not approved", rejected.steps["s1"].reason)

    def test_ask_human_step(self):
        auto, _ = make([plan_json(("s1", "ask_human", {"question": "Which month?"}),
                                  ("s2", "send_email", {"to": "a@b.c", "body": "Report for {{s1}}"}))])
        r = auto.run("send the report, ask me which month")
        self.assertEqual(r.pending[0].question, "Which month?")
        self.assertEqual(r.resume(s1="October").output, "sent to a@b.c: Report for October")

    def test_save_and_load_without_the_model(self):
        auto, _ = make([CHAIN])
        plan = auto.plan("x")
        with tempfile.TemporaryDirectory() as d:
            path = plan.save(os.path.join(d, "monthly.json"))
            again, llm2 = make([])
            self.assertEqual(again.load(path).run().get("s2"), 160)
            self.assertEqual(len(llm2.calls), 0, "a saved plan runs without the model")
            smaller = AutoFlow(FakeLLM([]), [add_numbers])
            with self.assertRaises(FlowDefinitionError):
                smaller.load(path)

    def test_bad_configuration(self):
        with self.assertRaises(FlowDefinitionError):
            AutoFlow(FakeLLM([]), [add_numbers], approve=["delete_everything"])
        with self.assertRaises(FlowDefinitionError):
            AutoFlow(FakeLLM([]), [add_numbers, add_numbers])
        with self.assertRaises(ValueError):
            AutoFlow(FakeLLM([]), [add_numbers]).plan("  ")

    def test_confirm_can_stop_the_run(self):
        auto, _ = make([CHAIN])
        seen = []
        r = auto.run("x", confirm=lambda plan: seen.append(plan) or False)
        self.assertIsNone(r.result)
        self.assertEqual(len(seen), 1)


class High(unittest.TestCase):
    def test_independent_steps_run_in_parallel(self):
        @tool
        def slow_add(a: float, b: float) -> float:
            """Add slowly."""
            time.sleep(0.2)
            return a + b

        plan = plan_json(*[(f"s{i}", "slow_add", {"a": i, "b": 1}) for i in range(1, 6)],
                         ("s6", "slow_add", {"a": "{{s1}}", "b": "{{s5}}"}), answer="s6")
        auto = AutoFlow(FakeLLM([plan]), [slow_add])
        t = time.perf_counter()
        r = auto.run("x")
        self.assertEqual(r.output, 8)
        self.assertLess(time.perf_counter() - t, 0.75, "five independent steps overlap")

    def test_failed_steps_are_retried(self):
        calls = {"n": 0}

        @tool
        def flaky(x: int) -> int:
            """Sometimes fails."""
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("blip")
            return x

        auto = AutoFlow(FakeLLM([plan_json(("s1", "flaky", {"x": 7}))]), [flaky], retry=2)
        self.assertEqual(auto.run("x").output, 7)

    def test_ask_ai_uses_the_model(self):
        llm = FakeLLM([plan_json(("s1", "add_numbers", {"a": 2, "b": 2}),
                                 ("s2", "ask_ai", {"prompt": "Write a thank-you note mentioning {{s1}}"})),
                       "Thank you for the 4!"])
        auto = AutoFlow(llm, [add_numbers])
        r = auto.run("add then thank")
        self.assertEqual(r.output, "Thank you for the 4!")
        self.assertEqual(llm.calls[-1]["messages"][-1]["content"], "Write a thank-you note mentioning 4")

    def test_agents_flows_and_async_tools_as_tools(self):
        helper = Agent(FakeLLM(["agent says 42"]), name="helper", system="Answers questions.")
        totals = Flow("totals", description="Add a and b.")
        totals.add(lambda a: a, name="first")
        totals.add(lambda first, b: first + b, name="sum")

        @tool
        async def fetch_rate(code: str) -> float:
            """Exchange rate."""
            await asyncio.sleep(0)
            return {"USD": 88.0}[code]

        plan = plan_json(("s1", "helper", {"task": "what is the answer?"}), ("s2", "totals", {"a": 1, "b": 2}),
                         ("s3", "fetch_rate", {"code": "USD"}))
        r = AutoFlow(FakeLLM([plan]), [helper, totals, fetch_rate]).run("x")
        self.assertEqual(r.result["s1"], "agent says 42")
        self.assertEqual(r.result["s2"], 3)
        self.assertEqual(r.result["s3"], 88.0)

    def test_field_references(self):
        @tool
        def invoice(item: str) -> dict:
            """Make an invoice."""
            return {"total": 250, "lines": [{"price": 100}, {"price": 150}]}

        auto = AutoFlow(FakeLLM([plan_json(("s1", "invoice", {"item": "tea"}),
                                           ("s2", "add_numbers", {"a": "{{s1.total}}", "b": "{{s1.lines.0.price}}"}))]),
                        [invoice, add_numbers])
        self.assertEqual(auto.run("x").output, 350)

    def test_plan_is_a_normal_flow(self):
        auto, _ = make([CHAIN])
        plan = auto.plan("x")
        self.assertIn("s1 --> s2", plan.to_mermaid())
        self.assertEqual(asyncio.run(plan.arun()).get("s2"), 160)


class Complex(unittest.TestCase):
    def test_web_app_end_to_end(self):
        auto, _ = make([plan_json(("s1", "add_numbers", {"a": 1, "b": 2}),
                                  ("s2", "send_email", {"to": "a@b.c", "body": "{{s1}}"}))], approve=["send_email"])
        httpd = WebApp(auto, "Test Desk").server("127.0.0.1", 0)
        t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        try:
            with urllib.request.urlopen(base + "/") as resp:
                self.assertIn("Test Desk", resp.read().decode())
            with urllib.request.urlopen(base + "/api/tools") as resp:
                tools = json.loads(resp.read())["tools"]
            self.assertTrue(any(t["name"] == "send_email" and t["approval"] for t in tools))
            code, plan = post("/api/plan", {"request": "add and email"})
            self.assertEqual(code, 200)
            self.assertTrue(plan["steps"][1]["approval"])
            code, state = post("/api/run", {"id": plan["id"]})
            self.assertEqual(state["status"], "paused")
            self.assertEqual(state["pending"][0]["kind"], "approval")
            code, state = post("/api/answer", {"id": plan["id"], "answers": {"approve_s2": "yes"}})
            self.assertEqual(state["status"], "ok")
            self.assertEqual(state["output"], "sent to a@b.c: 3")
            self.assertEqual(post("/api/run", {"id": "nope"})[0], 400)
            self.assertEqual(post("/api/answer", {"id": plan["id"], "answers": {}})[0], 400)
            self.assertEqual(post("/api/plan", {"request": ""})[0], 400)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_cli_ask_with_tools_file(self):
        from taskpilot.cli import main

        reply = plan_json(("s1", "add_numbers", {"a": 2, "b": 3}), ("s2", "send_email", {"to": "x@y.z", "body": "{{s1}}"}))
        code = textwrap.dedent(f'''
            from taskpilot import FakeLLM, tool

            @tool
            def add_numbers(a: float, b: float) -> float:
                """Add two numbers."""
                return a + b

            @tool
            def send_email(to: str, body: str) -> str:
                """Send an email."""
                return "sent " + body

            tools = [add_numbers, send_email]
            approve = ["send_email"]
            llm = FakeLLM([{reply!r}])
        ''')
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "office.py")
            with open(path, "w") as fh:
                fh.write(code)
            saved = os.path.join(d, "p.json")
            out, err = io.StringIO(), io.StringIO()
            with mock.patch("builtins.input", side_effect=["y"]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(main(["ask", path, "add 2 and 3 and email it", "--yes", "--save-plan", saved]), 0)
            self.assertEqual(out.getvalue().strip(), "sent 5")
            self.assertIn("[needs approval]", err.getvalue())
            out = io.StringIO()
            with mock.patch("builtins.input", side_effect=["n"]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["ask", path, "--plan", saved, "--yes"]), 0)
            self.assertEqual(out.getvalue().strip(), "", "a rejected email gives no answer")

    def test_many_users_at_once(self):
        def brain(messages, tools):
            import re

            n = int(re.search(r"minus one (\d+)", messages[1]["content"]).group(1))
            return plan_json(("s1", "add_numbers", {"a": n, "b": n}), ("s2", "subtract_numbers", {"a": "{{s1}}", "b": 1}))

        auto = AutoFlow(FakeLLM(fn=brain), [add_numbers, subtract_numbers])
        results, lock = [], threading.Lock()

        def user(n):
            out = auto.run(f"double and minus one {n}").output
            with lock:
                results.append(out == 2 * n - 1)

        threads = [threading.Thread(target=user, args=(n,)) for n in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 30)
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()
