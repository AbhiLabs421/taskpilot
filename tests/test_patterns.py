"""Ready-made patterns, from easy to complex."""

import asyncio
import threading
import time
import unittest

from taskpilot import (
    Agent, FakeLLM, Flow, FlowDefinitionError, GroupChat, Handoff, MapReduce, Parallel, Retry,
    ReviewLoop, Router, Sequential, StepError, Supervisor, Vote, tool,
)
from taskpilot.tracing import MemoryTracer


class EasyFlows(unittest.TestCase):
    def test_sequential(self):
        flow = Sequential(lambda text: text.strip(), str.upper, lambda s: s + "!")
        self.assertEqual(flow.run(text="  hi ").output, "HI!")

    def test_builtins_work_as_steps(self):
        self.assertEqual(Sequential(str.strip, str.upper, len).run({"self": " hey "}).output, 3)
        self.assertEqual(MapReduce(abs, sum).run(items=[-1, -2, 3]).output, 6)
        flow = Flow()
        flow.add(sorted, name="ordered")
        self.assertEqual(flow.run(iterable=[3, 1, 2]).output, [1, 2, 3])

    def test_parallel(self):
        def words(text):
            return len(text.split())

        def chars(text):
            return len(text)

        self.assertEqual(Parallel(words, chars).run(text="a bc").output, {"words": 2, "chars": 4})
        total = Parallel(words, chars, combine=lambda r: r["words"] + r["chars"])
        self.assertEqual(total.run(text="a bc").output, 6)

    def test_router(self):
        flow = Router(lambda q: "math" if any(c.isdigit() for c in q) else "chat",
                      {"math": lambda q: eval(q, {"__builtins__": {}}), "chat": lambda q: f"hi {q}"})
        r = flow.run(q="2*21")
        self.assertEqual(r.output, 42)
        self.assertEqual(r["route"], "math")
        self.assertEqual(flow.run(q="you").output, "hi you")

    def test_map_reduce(self):
        flow = MapReduce(lambda n: n * n, sum)
        self.assertEqual(flow.run(items=[1, 2, 3]).output, 14)
        self.assertEqual(MapReduce(lambda n: -n).run(items=[1, 2]).output, [-1, -2])

    def test_review_loop(self):
        def write(topic, feedback=None):
            return f"{topic} essay" + (" with examples" if feedback else "")

        def critique(text):
            return {"ok": "examples" in text, "tip": "add examples"}

        r = ReviewLoop(write, critique, until=lambda r: r["ok"]).run(topic="AI")
        self.assertEqual(r.output, "AI essay with examples")
        self.assertEqual(len(r.history["draft"]), 2)

    def test_vote(self):
        answers = iter(["42", "41", "42", " 42 ", "40"])
        lock = threading.Lock()

        def guess(question):
            with lock:
                return next(answers)

        r = Vote(guess, n=5).run(question="6*7?")
        self.assertEqual(r.output.strip(), "42")
        self.assertEqual(len(r["answers"]), 5)


class MediumFlows(unittest.TestCase):
    def test_router_default_and_unknown(self):
        flow = Router(lambda q: q, {"a": lambda q: "A"}, default=lambda q: "default")
        self.assertEqual(flow.run(q="zzz").output, "default")
        strict = Router(lambda q: q, {"a": lambda q: "A"})
        with self.assertRaises(StepError) as cm:
            strict.run(q="zzz")
        self.assertIn("expected one of ['a']", str(cm.exception))

    def test_router_input_named_key_is_safe(self):
        flow = Router(lambda key: key, {"x": lambda key: "X", "y": lambda key: "Y"})
        self.assertEqual(flow.run(key="y").output, "Y")
        self.assertEqual(flow.inputs, ["key"])

    def test_definition_errors(self):
        with self.assertRaises(FlowDefinitionError):
            Parallel()
        with self.assertRaises(FlowDefinitionError):
            Router(lambda q: q, {})
        with self.assertRaises(FlowDefinitionError):
            Router(lambda q: q, {"not valid": lambda q: q})
        with self.assertRaises(FlowDefinitionError):
            Router(lambda q: q, {"route": lambda q: q})
        with self.assertRaises(FlowDefinitionError):
            ReviewLoop(lambda topic: topic, lambda d: d, until=bool)
        with self.assertRaises(FlowDefinitionError):
            Vote(lambda: 1, n=0)
        with self.assertRaises(FlowDefinitionError):
            MapReduce(lambda: 1)

    def test_map_reduce_policies(self):
        calls = {"n": 0}
        lock = threading.Lock()

        def flaky(n, factor):
            with lock:
                calls["n"] += 1
                if calls["n"] == 2:
                    raise ConnectionError
            return n * factor

        flow = MapReduce(flaky, sum, retry=Retry(2, backoff=0), max_concurrency=2)
        self.assertEqual(flow.run(items=[1, 2, 3], factor=10).output, 60)

    def test_review_loop_max_and_custom_feedback_name(self):
        flow = ReviewLoop(lambda topic, notes=None: topic, lambda d: {"ok": False}, until=lambda r: r["ok"],
                          feedback="notes", max_iterations=3)
        r = flow.run(topic="x")
        self.assertEqual(r.loops["improve"], {"rounds": 3, "stopped_by": "max_iterations"})

    def test_patterns_are_normal_flows(self):
        flow = Parallel(lambda x: x + 1, lambda x: x * 2)
        mem = flow.trace(MemoryTracer())
        r = asyncio.run(flow.arun(x=3))
        self.assertEqual(r.output, {"branch_1": 4, "branch_2": 6})
        self.assertEqual(len(mem.last()), 3)
        outer = Flow("outer")
        outer.add(lambda raw: int(raw), name="x")
        outer.add(flow.as_step("both"))
        self.assertEqual(outer.run(raw="5").output, {"branch_1": 6, "branch_2": 10})


class HighFlows(unittest.TestCase):
    def test_parallel_really_overlaps(self):
        def a(x):
            time.sleep(0.15)
            return 1

        def b(x):
            time.sleep(0.15)
            return 2

        t = time.perf_counter()
        Parallel(a, b).run(x=0)
        self.assertLess(time.perf_counter() - t, 0.28)

    def test_router_with_llm_classifier_and_review_loop_with_llm(self):
        llm = FakeLLM(fn=lambda m, t: "billing" if "refund" in m[-1]["content"] else "tech")
        flow = Router(lambda msg: llm.ask(f"Classify: {msg}").strip(), {
            "billing": lambda msg: "billing team",
            "tech": lambda msg: "tech team",
        })
        self.assertEqual(flow.run(msg="I want a refund").output, "billing team")

        def brain(m, t):
            text = m[-1]["content"]
            if text.startswith("Write"):
                return "draft with numbers 1 2 3" if "numbers" in text else "plain draft"
            return '{"ok": true}' if "numbers" in text else '{"ok": false, "tip": "use numbers"}'

        writer = FakeLLM(fn=brain)
        loop = ReviewLoop(
            lambda topic, feedback=None: writer.ask(f"Write about {topic}" + (f" ({feedback['tip']})" if feedback else "")),
            lambda draft: writer.json(f"Review: {draft}"),
            until=lambda r: r["ok"],
        )
        self.assertEqual(loop.run(topic="stats").output, "draft with numbers 1 2 3")

    def test_vote_custom_pick_and_parallel_calls(self):
        def slow(q):
            time.sleep(0.1)
            return len(q)

        t = time.perf_counter()
        r = Vote(slow, n=6, pick=max).run(q="abc")
        self.assertEqual(r.output, 3)
        self.assertLess(time.perf_counter() - t, 0.35)


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return f"order {order_id}: refunded"


class Agents(unittest.TestCase):
    def test_supervisor_delegates(self):
        researcher = Agent(FakeLLM(["Fact: mangoes are India's national fruit."]), name="researcher",
                           system="Finds facts.")
        writer = Agent(FakeLLM(fn=lambda m, t: "Poem: " + m[-1]["content"]), name="writer", system="Writes poems.")
        boss = Supervisor(FakeLLM([
            {"tool": "researcher", "args": {"task": "fact about mangoes"}},
            {"tool": "writer", "args": {"task": "poem using: national fruit"}},
            "Final: Poem: poem using: national fruit",
        ]), [researcher, writer])
        result = boss.run("Write a poem about mangoes with a fact")
        self.assertEqual(result.tools_used, ["researcher", "writer"])
        self.assertIn("national fruit", result.steps[0].output)
        self.assertTrue(result.output.startswith("Final"))
        self.assertIn("researcher", boss.tools["researcher"].description)

    def test_handoff_routes_to_specialist_with_tools(self):
        triage = Agent(FakeLLM([{"tool": "transfer_to_billing", "args": {"reason": "money"}}]), name="triage",
                       system="Sends people to the right team.")
        billing = Agent(FakeLLM([{"tool": "lookup_order", "args": {"order_id": "A7"}}, "Your order A7 was refunded."]),
                        tools=[lookup_order], name="billing", system="Handles refunds.")
        tech = Agent(FakeLLM(["tech answer"]), name="tech")
        result = Handoff([triage, billing, tech]).run("Where is my refund for A7?")
        self.assertEqual(result.output, "Your order A7 was refunded.")
        self.assertEqual(result.path, ["triage", "billing"])
        self.assertEqual(result.agent, "billing")
        self.assertEqual(result.steps[0].output, "order A7: refunded")
        self.assertIn("handed the conversation to billing: money", result.messages[1]["content"])

    def test_after_handoff_the_user_speaks_last(self):
        triage = Agent(FakeLLM([{"tool": "transfer_to_math", "args": {"reason": "numbers"}}]), name="triage")
        math = Agent(FakeLLM(["", "100"]), name="math")  # first reply empty, like reasoning models can
        result = Handoff([triage, math]).run("25 * 4?")
        self.assertEqual(result.output, "100")
        first = math.llm.calls[0]["messages"]
        self.assertEqual(first[-1], {"role": "user", "content": "25 * 4?"})
        self.assertIn("triage handed the conversation to math: numbers", first[0]["content"])

    def test_handoff_limits(self):
        ping = Agent(FakeLLM([{"tool": "transfer_to_pong"}]), name="ping")
        pong = Agent(FakeLLM(fn=lambda m, t: {"tool": "transfer_to_ping"} if t else "stopped bouncing"), name="pong")
        result = Handoff([ping, pong], max_handoffs=3).run("x")
        self.assertEqual(len(result.path), 4)
        self.assertEqual(result.output, "stopped bouncing")
        with self.assertRaises(FlowDefinitionError):
            Handoff([ping])
        with self.assertRaises(FlowDefinitionError):
            Handoff([ping, pong], start="nobody")

    def test_group_chat_round_robin_and_stop_word(self):
        pm = Agent(FakeLLM(["We need a login page."]), name="pm")
        dev = Agent(FakeLLM(["Built it with OAuth."]), name="dev")
        qa = Agent(FakeLLM(["Tested, all good. DONE"]), name="qa")
        chat = GroupChat([pm, dev, qa], rounds=10)
        result = chat.run("Plan the login feature")
        self.assertEqual(result.speakers, ["user", "pm", "dev", "qa"])
        self.assertEqual(result.stopped_by, "done")
        self.assertIn("pm: We need a login page.", dev.llm.calls[0]["messages"][-1]["content"])

    def test_group_chat_only_finishers_can_end_it(self):
        adder = Agent(FakeLLM(["75 DONE"]), name="adder")           # tries to end too early
        checker = Agent(FakeLLM(["correct. DONE"]), name="checker")
        result = GroupChat([adder, checker], finishers=["checker"]).run("50 + 25?")
        self.assertEqual(result.speakers, ["user", "adder", "checker"])
        self.assertEqual(result.stopped_by, "done")
        self.assertIn("Do not write the word DONE", adder.llm.calls[0]["messages"][-1]["content"])
        self.assertIn("end your message with the word DONE", checker.llm.calls[0]["messages"][-1]["content"])
        with self.assertRaises(FlowDefinitionError):
            GroupChat([adder, checker], finishers=["nobody"])

    def test_group_chat_custom_speaker_until_and_rounds(self):
        a = Agent(FakeLLM(["a says"]), name="a")
        b = Agent(FakeLLM(["b says"]), name="b")
        r = GroupChat([a, b], rounds=4, speaker=lambda msgs, names: "b").run("go")
        self.assertEqual(r.speakers, ["user", "b", "b", "b", "b"])
        self.assertEqual(r.stopped_by, "rounds")
        r = GroupChat([a, b], until=lambda msgs: len(msgs) >= 3).run("go")
        self.assertEqual(r.stopped_by, "until")
        manager = FakeLLM(["b", "a"])
        r = GroupChat([a, b], rounds=2, speaker=manager).run("go")
        self.assertEqual(r.speakers, ["user", "b", "a"])
        with self.assertRaises(FlowDefinitionError):
            GroupChat([a, b], speaker=123)


class AgentsAndFlows(unittest.TestCase):
    def make_refund_flow(self):
        refund = Flow("process_refund", description="Validate and process a refund.")

        @refund.step
        def order(order_id: str):
            return {"A7": 2499, "B2": 299}.get(order_id.upper(), 0)

        @refund.step
        def decision(order, reason: str = "unspecified"):
            return f"refund {order} ({reason})" if order else "no such order"

        return refund

    def test_flow_as_tool_schema_and_call(self):
        t = self.make_refund_flow().as_tool()
        spec = t.spec()["function"]
        self.assertEqual(spec["name"], "process_refund")
        self.assertEqual(spec["description"], "Validate and process a refund.")
        self.assertEqual(spec["parameters"]["required"], ["order_id"])
        self.assertEqual(t.invoke({"order_id": "a7", "reason": "broken"}), "refund 2499 (broken)")
        self.assertEqual(t.invoke({"order_id": "b2"}), "refund 299 (unspecified)")
        self.assertIn("missing required", t.invoke({}))

    def test_agent_runs_a_workflow_as_a_tool(self):
        llm = FakeLLM([{"tool": "process_refund", "args": {"order_id": "A7", "reason": "broken"}}, "Refund done."])
        result = Agent(llm, tools=[self.make_refund_flow().as_tool(), lookup_order]).run("refund A7, it broke")
        self.assertEqual(result.tools_used, ["process_refund"])
        self.assertEqual(result.steps[0].output, "refund 2499 (broken)")

    def test_paused_workflow_tool_tells_the_model(self):
        gate = Flow("gate")
        gate.add(lambda amount, ctx: ctx.wait_for("ok"), name="ask")
        self.assertIn("waiting for a person", gate.as_tool().invoke({"amount": "5"}))

    def test_agent_as_step_with_prompt_fields(self):
        writer = Agent(FakeLLM(fn=lambda m, t: "notes: " + m[-1]["content"]), name="writer")
        flow = Flow()
        flow.add(lambda raw: raw.strip(), name="topic")
        flow.add(writer.as_step("notes", "Research {topic} for {audience}"))
        r = flow.run(raw=" tea ", audience="kids")
        self.assertEqual(r.output, "notes: Research tea for kids")
        self.assertEqual(flow.inputs, ["raw", "audience"])


class ComplexAgents(unittest.TestCase):
    def test_teams_compose_into_flows_and_other_teams(self):
        triage = Agent(FakeLLM([{"tool": "transfer_to_billing"}]), name="triage")
        billing = Agent(FakeLLM(["refund approved"]), name="billing")
        support = Handoff([triage, billing], name="support")
        boss = Supervisor(FakeLLM([{"tool": "support", "args": {"task": "refund?"}}, "Done: refund approved"]), [support])
        self.assertEqual(boss.run("help customer").output, "Done: refund approved")

        flow = Flow("tickets")
        flow.add(support.as_step("reply"), name="reply")
        flow.add(lambda reply: reply.upper(), name="shout")
        r = flow.run(task="refund?")
        self.assertEqual(r.output, "REFUND APPROVED")

    def test_team_async_and_parallel_tickets(self):
        a = Agent(FakeLLM(["first"]), name="a")
        b = Agent(FakeLLM(["second DONE"]), name="b")
        chat = GroupChat([a, b])

        async def main():
            return await asyncio.gather(*(chat.arun(f"t{i}") for i in range(8)))

        results = asyncio.run(main())
        self.assertTrue(all(r.stopped_by == "done" for r in results))
        self.assertEqual(chat("x"), "second DONE")

    def test_map_reduce_over_agents(self):
        grader = Agent(FakeLLM(fn=lambda m, t: str(len(m[-1]["content"]))), name="grader")
        flow = MapReduce(lambda essay: int(grader(essay)), lambda scores: sum(scores) / len(scores))
        self.assertEqual(flow.run(items=["aa", "aaaa"]).output, 3.0)


if __name__ == "__main__":
    unittest.main()
