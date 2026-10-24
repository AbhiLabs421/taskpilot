"""Reliability: evals, guards (check / verify / plan review), memory. Easy to complex."""

import json
import os
import tempfile
import threading
import unittest

from taskpilot import (Agent, AutoFlow, CheckFailed, ConversationMemory, Disagreement, FakeLLM, FallbackLLM, Flow,
                       LLMError, MemoryStore, Retry, StepError, Unclear, VerificationError, tool)
from taskpilot.evals import Case, evaluate, wilson
from taskpilot.memory import estimate_tokens


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def get_order(order_id: str) -> dict:
    """Look up an order."""
    return {"B2": {"item": "phone case", "status": "in transit"}}.get(order_id, {"error": "no such order"})


class EasyEvals(unittest.TestCase):
    def test_wilson_interval(self):
        low, high = wilson(20, 20)
        self.assertGreater(low, 0.8)
        self.assertLess(low, 0.9)
        self.assertEqual(high, 1.0)
        low, _ = wilson(3000, 3000)
        self.assertGreater(low, 0.998, "thousands of passes are needed to claim 99.9%")

    def test_case_judging(self):
        self.assertIsNone(Case("x", 160).judge(160.0))
        self.assertIn("expected 160", Case("x", 160).judge(161))
        self.assertIsNone(Case("x", 100 / 3).judge(33.333333333333336))
        self.assertIsNone(Case("x", lambda o: "tea" in o).judge("a tea set"))
        self.assertEqual(Case("x", lambda o: "needs tea").judge("x"), "needs tea")
        self.assertIsNone(Case("x", "ok").judge("  ok "))

    def test_evaluate_counts_crashes_as_failures(self):
        def target(q):
            if q == "boom":
                raise RuntimeError("broken")
            return len(q)

        report = evaluate(target, [Case("abc", 3), Case("boom", 4), Case("ab", 3)], runs=2)
        self.assertEqual((report.passed, report.total), (2, 6))
        self.assertIn("RuntimeError: broken", report.summary())
        self.assertEqual(report.to_dict()["cases"]["boom"], {"passed": 0, "total": 2})


class MediumGuards(unittest.TestCase):
    def test_step_check_retries_then_fails_clearly(self):
        calls = {"n": 0}
        flow = Flow()

        @flow.step(retry=Retry(3, backoff=0), check=lambda total: total > 0 or "total must be positive")
        def total(a, b):
            calls["n"] += 1
            return a - b if calls["n"] == 1 else a + b

        self.assertEqual(flow.run(a=1, b=5).output, 6)
        bad = Flow()
        bad.add(lambda: -1, name="neg", check=lambda v: v > 0)
        with self.assertRaises(StepError) as cm:
            bad.run()
        self.assertIsInstance(cm.exception.cause, CheckFailed)
        boom = Flow()
        boom.add(lambda: {"x": 1}, name="s", check=lambda v: v["missing"] > 0)
        with self.assertRaises(StepError) as cm:
            boom.run()
        self.assertIn("KeyError", str(cm.exception.cause))
        fb = Flow()
        fb.add(lambda: -1, name="neg", check=lambda v: v > 0, fallback=0)
        self.assertEqual(fb.run().output, 0)

    def test_step_check_in_map_and_dynamic(self):
        flow = Flow()
        flow.add(lambda n: n * 2, name="dbl", map="n", check=lambda v: v < 10)
        with self.assertRaises(StepError):
            flow.run(n=[1, 2, 9])
        flow2 = Flow()
        flow2.add(lambda n: -n, name="neg", dynamic=True, check=lambda v: v >= 0, fallback="rejected")
        flow2.add(lambda ctx: ctx.call("neg", n=3), name="main")
        self.assertEqual(flow2.run().output, "rejected")

    def test_agent_check_fix_loop(self):
        llm = FakeLLM(["wrong", "still wrong", "right"])
        r = Agent(llm, check=lambda a: a == "right" or "say right", max_fixes=2).run("x")
        self.assertEqual((r.output, r.verified, r.fixes), ("right", True, 2))
        r = Agent(FakeLLM(["wrong"]), check=lambda a: a == "right", max_fixes=1).run("x")
        self.assertEqual((r.verified, r.fixes, len(r.problems)), (False, 1, 2))
        with self.assertRaises(VerificationError):
            Agent(FakeLLM(["wrong"]), check=lambda a: False, max_fixes=0, strict=True).run("x")
        self.assertIsNone(Agent(FakeLLM(["hi"])).run("x").verified, "no checks -> not verified either way")

    def test_agent_verify_catches_invented_facts(self):
        def brain(messages, tools):
            t = messages[-1]["content"]
            if "check an AI assistant" in t:
                bad = "25 Oct" in t
                return json.dumps({"supported": not bad, "problems": ["arrival date not in evidence"] if bad else []})
            if not any(m.get("role") == "tool" for m in messages):
                return {"tool": "get_order", "args": {"order_id": "B2"}}
            if "has problems" in t:
                return "A phone case, in transit. The arrival date is not available."
            return "A phone case, arriving 25 Oct."

        r = Agent(FakeLLM(fn=brain), tools=[get_order], verify=True).run("What and when?")
        self.assertTrue(r.verified)
        self.assertEqual(r.fixes, 1)
        self.assertIn("not available", r.output)
        checker_prompt = next(c for c in r.messages if "has problems" in str(c.get("content")))
        self.assertIn("arrival date not in evidence", checker_prompt["content"])

    def test_verify_with_a_separate_checker_and_checker_failure(self):
        checker = FakeLLM(['{"supported": true, "problems": []}'])
        r = Agent(FakeLLM(["fine"]), verify=checker).run("x")
        self.assertTrue(r.verified)
        self.assertEqual(len(checker.calls), 1)
        broken = FakeLLM(["not json"])
        r = Agent(FakeLLM(["fine"]), verify=broken, max_fixes=0).run("x")
        self.assertFalse(r.verified, "a checker that fails never counts as a pass")

    def test_tool_output_is_capped(self):
        @tool
        def huge() -> str:
            """Returns a lot."""
            return "x" * 50000

        r = Agent(FakeLLM([{"tool": "huge"}, "done"]), tools=[huge], max_tool_chars=1000).run("x")
        self.assertLess(len(r.steps[0].output), 1100)
        self.assertIn("full result was 50000 characters", r.steps[0].output)


class HighPlanReview(unittest.TestCase):
    PLAN_BAD = json.dumps({"steps": [{"id": "s1", "tool": "add", "args": {"a": 12500, "b": 75000}}]})
    PLAN_GOOD = json.dumps({"steps": [{"id": "s1", "tool": "add", "args": {"a": 125000, "b": 75000}}]})

    def test_review_sends_problems_back_to_the_planner(self):
        def brain(messages, tools):
            t = messages[-1]["content"]
            if "You review a plan" in t:
                import re

                if re.search(r"a=12500\b", t):
                    return json.dumps({"ok": False, "problems": ["12500 should be 125000"]})
                return json.dumps({"ok": True, "problems": []})
            if "12500 should be 125000" in t:
                return self.PLAN_GOOD
            return self.PLAN_BAD

        llm = FakeLLM(fn=brain)
        auto = AutoFlow(llm, [add], review=True)
        r = auto.run("Add 125000 and 75000")
        self.assertEqual(r.output, 200000)
        review_prompt = next(c["messages"][-1]["content"] for c in llm.calls if "You review a plan" in c["messages"][-1]["content"])
        self.assertIn("not written in the request", review_prompt)
        self.assertIn("12500", review_prompt)

    def test_ambiguous_request_asks_a_question(self):
        def brain(messages, tools):
            if "You review a plan" in messages[-1]["content"]:
                return json.dumps({"ok": False, "problems": ["2000 could be daily or total"], "ambiguous": True,
                                   "question": "Is 2000 spent per day or in total?"})
            return self.PLAN_GOOD

        with self.assertRaises(Unclear) as cm:
            AutoFlow(FakeLLM(fn=brain), [add], review=True).plan("30 days, 450 a day, spent 2000")
        self.assertEqual(cm.exception.question, "Is 2000 spent per day or in total?")

    def test_review_never_passes_on_reviewer_failure(self):
        replies = [self.PLAN_GOOD, "garbage", "garbage", "garbage"] * 3
        auto = AutoFlow(FakeLLM(replies), [add], review=True, repair=1)
        with self.assertRaises(Exception):
            auto.plan("Add 125000 and 75000")

    def test_review_off_by_default_for_autoflow(self):
        llm = FakeLLM([self.PLAN_BAD])
        self.assertEqual(AutoFlow(llm, [add]).run("Add 125000 and 75000").output, 87500)
        self.assertEqual(len(llm.calls), 1)


class HighStaysUp(unittest.TestCase):
    PLAN_A = json.dumps({"steps": [{"id": "s1", "tool": "add", "args": {"a": 1, "b": 2}}]})
    PLAN_A2 = json.dumps({"steps": [{"id": "x1", "tool": "add", "args": {"a": 1.0, "b": 2}}]})
    PLAN_B = json.dumps({"steps": [{"id": "s1", "tool": "add", "args": {"a": 1, "b": 3}}]})

    def test_fallback_llm_switches_models(self):
        def down(messages, tools):
            raise LLMError("HTTP 503: down")

        backup = FakeLLM(["from backup"])
        llm = FallbackLLM([FakeLLM(fn=down), backup])
        self.assertEqual(llm.ask("x"), "from backup")
        self.assertIs(llm.last, backup)
        self.assertEqual(Agent(llm).run("x").output, "from backup")
        with self.assertRaises(LLMError) as cm:
            FallbackLLM([FakeLLM(fn=down), FakeLLM(fn=down)]).ask("x")
        self.assertIn("Every model failed", str(cm.exception))
        with self.assertRaises(ValueError):
            FallbackLLM([])

    def test_agent_escalates_to_a_stronger_model(self):
        weak = FakeLLM(["wrong"])
        strong = FakeLLM(["right"])
        r = Agent(weak, check=lambda a: a == "right" or "must be right", max_fixes=1, escalate=strong).run("x")
        self.assertEqual((r.output, r.verified, r.escalated), ("right", True, True))
        self.assertEqual(len(r.problems), 2, "the weak model's problems are kept for the record")
        r = Agent(FakeLLM(["right"]), check=lambda a: a == "right", escalate=strong).run("x")
        self.assertFalse(r.escalated)
        with self.assertRaises(VerificationError):
            Agent(weak, check=lambda a: False, max_fixes=0, escalate=FakeLLM(["nope"]), strict=True).run("x")

    def test_planner_consensus(self):
        planners = [FakeLLM([self.PLAN_A]), FakeLLM([self.PLAN_A2]), FakeLLM([self.PLAN_B])]
        auto = AutoFlow(FakeLLM([]), [add], consensus=planners)
        self.assertEqual(auto.run("add 1 and 2").output, 3, "two of three agree (ids and 1 vs 1.0 don't matter)")
        split = AutoFlow(FakeLLM([]), [add], consensus=[FakeLLM([self.PLAN_A]), FakeLLM([self.PLAN_B])])
        with self.assertRaises(Disagreement):
            split.plan("add 1 and something")
        self.assertTrue(issubclass(Disagreement, Unclear))

    def test_planner_escalates(self):
        weak = FakeLLM([json.dumps({"steps": [{"id": "s1", "tool": "nope", "args": {}}]})])
        strong = FakeLLM([self.PLAN_A])
        auto = AutoFlow(weak, [add], repair=1, escalate=strong)
        self.assertEqual(auto.run("add 1 and 2").output, 3)
        self.assertEqual(len(strong.calls), 1)


class ComplexMemory(unittest.TestCase):
    def test_conversation_compacts_and_keeps_facts(self):
        summarizer = FakeLLM(fn=lambda m, t: "- Ravi ordered A7 (tea set, 2499)\n- promised a refund")
        mem = ConversationMemory(summarizer, max_tokens=200, keep_recent=4)
        for i in range(12):
            mem.add([{"role": "user", "content": f"message {i} " + "words " * 30},
                     {"role": "assistant", "content": f"reply {i} " + "words " * 30}])
        self.assertGreaterEqual(mem.compactions, 1)
        self.assertIn("promised a refund", mem.summary)
        self.assertLessEqual(len(mem.messages), 4)
        ctx = mem.context()
        self.assertIn("Summary of our earlier conversation", ctx[0]["content"])
        self.assertEqual(ctx[2]["role"], "user", "kept turns start at a user message")
        prompt = summarizer.calls[0]["messages"][-1]["content"]
        self.assertIn("Keep every fact, number, name", prompt)

    def test_compaction_never_orphans_tool_results(self):
        mem = ConversationMemory(None, max_tokens=200, keep_recent=3)
        msgs = []
        for i in range(10):
            msgs += [{"role": "user", "content": "q" * 200},
                     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_order", "arguments": {}}}]},
                     {"role": "tool", "content": "r" * 200}, {"role": "assistant", "content": "a" * 200}]
        mem.add(msgs)
        self.assertTrue(mem.messages[0]["role"] == "user")
        self.assertTrue(mem.summary, "without a model the old turns are shortened, not lost")

    def test_agent_with_compacting_memory_across_runs(self):
        seen = []

        def brain(messages, tools):
            seen.append(messages)
            if "Summarise this conversation" in messages[-1]["content"]:
                return "- user's name is Asha; order B2"
            return "ok " + messages[-1]["content"][:10]

        llm = FakeLLM(fn=brain)
        agent = Agent(llm, memory=ConversationMemory(llm, max_tokens=300, keep_recent=2))
        agent.run("My name is Asha and my order is B2. " + "details " * 60)
        for i in range(4):
            agent.run(f"question {i} " + "more " * 40)
        last = seen[-1]
        self.assertTrue(any("user's name is Asha" in str(m.get("content")) for m in last))
        self.assertLess(estimate_tokens(last), 700)

    def test_memory_store(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "facts.json")
            store = MemoryStore(path)
            store.remember("Ravi prefers emails in Hindi", key="ravi.language")
            store.remember("Asha's order B2 ships by courier", tags=["asha"])
            store.remember("Ravi prefers emails in English", key="ravi.language")
            self.assertEqual(len(store), 2, "same key replaces the old fact")
            again = MemoryStore(path)
            self.assertEqual(again.recall("what language does Ravi like for emails")[0]["text"], "Ravi prefers emails in English")
            self.assertEqual(again.recall("asha")[0]["tags"], ["asha"])
            self.assertEqual(again.recall("unrelated words"), [])
            self.assertTrue(again.forget("ravi.language"))
            remember, recall = again.as_tools()
            self.assertEqual(remember.invoke({"fact": "Imran is a VIP", "key": "imran.vip"}), "remembered")
            self.assertIn("Imran is a VIP", recall.invoke({"query": "is imran vip"}))
            self.assertEqual(recall.invoke({"query": "zzz"}), "nothing saved about that")

    def test_memory_store_is_thread_safe(self):
        store = MemoryStore()
        threads = [threading.Thread(target=lambda i=i: [store.remember(f"fact {i}-{j}") for j in range(50)]) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(store), 400)

    def test_eval_of_a_guarded_agent(self):
        def brain(messages, tools):
            t = messages[-1]["content"]
            if "check an AI assistant" in t:
                return json.dumps({"supported": "invented" not in t, "problems": ["invented fact"]})
            if "has problems" in t:
                return "not available"
            return "invented answer"

        guarded = evaluate(lambda q: Agent(FakeLLM(fn=brain), verify=True).run(q).output,
                           [Case("q", lambda o: "invented" not in o)], runs=5)
        plain = evaluate(lambda q: Agent(FakeLLM(fn=brain)).run(q).output,
                         [Case("q", lambda o: "invented" not in o)], runs=5)
        self.assertEqual((guarded.passed, plain.passed), (5, 0))


if __name__ == "__main__":
    unittest.main()
