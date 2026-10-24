"""LIVE: talks to a real Ollama server. Skipped unless OLLAMA_API_KEY or TASKPILOT_LIVE is set.

    OLLAMA_API_KEY=... python -m unittest tests.test_live_ollama
    TASKPILOT_LIVE=1 OLLAMA_HOST=http://localhost:11434 python -m unittest tests.test_live_ollama
"""

import os
import unittest

from taskpilot import Agent, AutoFlow, Flow, Workspace, GroupChat, Handoff, Ollama, ReviewLoop, Router, Supervisor, tool

LIVE = bool(os.environ.get("OLLAMA_API_KEY") or os.environ.get("TASKPILOT_LIVE"))
MODEL = os.environ.get("TASKPILOT_MODEL", "gpt-oss:120b")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers exactly."""
    return a * b


@unittest.skipUnless(LIVE, "set OLLAMA_API_KEY (or TASKPILOT_LIVE=1) to run live tests")
class TestLiveOllama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = Ollama(MODEL, options={"temperature": 0})

    def test_ask(self):
        answer = self.llm.ask("What is the capital of India? Reply with just the city name.")
        self.assertIn("delhi", answer.lower())

    def test_json(self):
        schema = {
            "type": "object",
            "properties": {"country": {"type": "string"}, "capital": {"type": "string"}},
            "required": ["country", "capital"],
        }
        data = self.llm.json("Give the country and capital for Japan.", schema=schema)
        self.assertEqual(data["capital"].lower(), "tokyo")

    def test_stream(self):
        pieces = list(self.llm.stream("Count from 1 to 5 separated by spaces. Only the numbers."))
        self.assertIn("5", "".join(pieces))

    def test_agent_uses_tool(self):
        agent = Agent(self.llm, tools=[multiply], system="Use tools for arithmetic. Answer with the number only.")
        result = agent.run("What is 1234 * 5678?")
        self.assertIn("multiply", result.tools_used)
        self.assertIn("7006652", result.output.replace(",", ""))

    def test_flow_with_parallel_llm_steps(self):
        flow = Flow("live")

        @flow.step
        def topics(subject):
            return [f"{subject} for kids", f"{subject} for experts"]

        @flow.step(map="topics", retry=2)
        def lines(topics):
            return self.llm.ask(f"Write one short sentence about {topics}.")

        @flow.step
        def report(lines):
            return "\n".join(lines)

        result = flow.run(subject="black holes")
        self.assertEqual(len(result["lines"]), 2)
        self.assertTrue(all(line.strip() for line in result["lines"]))


    # ------------------------------------------------------------ patterns
    def test_router_with_llm(self):
        flow = Router(
            lambda msg: self.llm.ask(f"Is this about 'billing' or 'tech'? Reply with one word.\n\n{msg}").strip().lower().strip("."),
            {"billing": lambda msg: "billing team", "tech": lambda msg: "tech team"},
            default=lambda msg: "general",
        )
        self.assertEqual(flow.run(msg="I was charged twice on my credit card").output, "billing team")

    def test_review_loop_with_llm(self):
        def write(topic, feedback=None):
            extra = f" Reviewer said: {feedback['tip']}" if feedback else ""
            return self.llm.ask(f"Write one sentence about {topic}.{extra}")

        def critique(text):
            return self.llm.json(
                f"Does this sentence contain a number written with digits? Sentence: {text}",
                schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "tip": {"type": "string"}},
                        "required": ["ok", "tip"]},
            )

        def has_digit(r):
            return r["ok"]

        r = ReviewLoop(write, critique, until=has_digit, max_iterations=4).run(topic="the Moon")
        self.assertTrue(r["review"]["ok"] or r.loops["improve"]["stopped_by"] == "max_iterations")
        self.assertTrue(r.output)

    def test_supervisor_handoff_groupchat(self):
        calc = Agent(self.llm, tools=[multiply], name="calculator", system="Does exact arithmetic with tools.")
        boss = Supervisor(self.llm, [calc], system="Always delegate arithmetic. Answer with the number only.")
        result = boss.run("What is 321 * 123?")
        self.assertIn("calculator", result.tools_used)
        self.assertIn("39483", result.output.replace(",", ""))

        triage = Agent(self.llm, name="triage", system="You never answer questions yourself. Always transfer math questions to math.")
        math = Agent(self.llm, tools=[multiply], name="math", system="Answer math with the multiply tool. Number only.")
        h = Handoff([triage, math]).run("What is 25 * 4?")
        self.assertEqual(h.path[-1], "math")
        self.assertIn("100", h.output)

        idea = Agent(self.llm, name="ideas", system="Suggest one short product name in one line. Never write the word DONE; the critic decides.")
        critic = Agent(self.llm, name="critic", system="If the last suggested name is good, reply 'Approved' and end with DONE.")
        chat = GroupChat([idea, critic], rounds=4).run("Name a tea brand.")
        self.assertEqual(chat.speakers[:3], ["user", "ideas", "critic"])
        self.assertIn(chat.stopped_by, ("done", "rounds"))


    # ------------------------------------------------------------ AutoFlow
    def test_autoflow_plans_and_runs(self):
        @tool
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        @tool
        def subtract_numbers(a: float, b: float) -> float:
            """Subtract b from a."""
            return a - b

        sent = []

        @tool
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email."""
            sent.append((to, body))
            return f"sent to {to}"

        auto = AutoFlow(self.llm, [add_numbers, subtract_numbers, send_email], approve=["send_email"])
        r = auto.run("Sales 125000 aur 75000 thi, returns 40000. Net sales nikaalo aur boss@company.com ko email karo")
        self.assertTrue(r.paused, r.plan.describe())
        self.assertIn("160000", r.pending[0].question.replace(",", ""))
        done = r.resume({r.pending[0].key: "yes"})
        self.assertTrue(done.ok)
        # The model may pick the email or the net sales as "the answer"; what matters is what happened.
        self.assertIn(done.output, (160000, "sent to boss@company.com"))
        self.assertEqual(sent[0][0], "boss@company.com")
        self.assertIn("160000", sent[0][1].replace(",", ""))

        r = auto.run("What is (10 + 20) - 5? Also separately add 2 and 3.")
        values = [v for v in r.result.values.values()]
        self.assertIn(25, values)
        self.assertIn(5, values)


    # ------------------------------------------------------------ Workspace
    def test_workspace_suggest_review_and_approval(self):
        sent = []

        @tool
        def get_order(order_id: str) -> dict:
            """Look up an order by id (customer, email, item)."""
            return {"customer": "Ravi", "email": "ravi@example.com", "item": "tea set"}

        @tool
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email."""
            sent.append(to)
            return f"sent to {to}"

        ws = Workspace(self.llm, [get_order, send_email], approve=["send_email"])
        ws.create_agent("support", "Polite support agent. Look up orders before writing.", ["get_order", "send_email"])
        ws.create_agent("reviewer", "Check customer emails: polite, correct, under 80 words.")
        request = "Write a short thank-you email to the customer of order A7 and send it. The reviewer must check it first."
        tip = ws.suggest(request)
        self.assertIn(tip["pattern"], ("review", "sequential", "plan"), tip)
        run = ws.start(request, "review", ["support", "reviewer"])
        self.assertEqual(sent, [], "nothing is sent before a person approves")
        self.assertEqual(run.status, "waiting", run.trace)
        ws.approve_action(run.id, run.actions[0]["id"], True)
        self.assertEqual(sent, ["ravi@example.com"])

    # ------------------------------------------------------------ MCP
    def test_agent_uses_mcp_tools(self):
        import sys
        import tempfile

        from taskpilot.mcp import MCPClient

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
        script = os.path.join(tempfile.mkdtemp(), "crm_server.py")
        with open(script, "w") as fh:
            fh.write(
                "import sys\n"
                f"sys.path.insert(0, {src!r})\n"
                "from taskpilot.mcp import MCPServer\n\n"
                "def find_customer(email: str) -> dict:\n"
                "    'Find a customer by email (name, tier).'\n"
                "    if email != 'ravi@x.com':\n"
                "        raise KeyError('no customer with that email')\n"
                "    return {'name': 'Ravi', 'tier': 'gold'}\n\n"
                "MCPServer([find_customer], name='crm').serve_stdio()\n"
            )
        with MCPClient.stdio([sys.executable, script]) as crm:
            agent = Agent(self.llm, tools=crm.tools(prefix="crm_"), system="Use the tools. Be brief.")
            r = agent.run("What is the tier of customer ravi@x.com?")
            self.assertIn("crm_find_customer", r.tools_used)
            self.assertIn("gold", r.output.lower())
            missing = agent.run("What is the tier of customer nobody@x.com?")
            self.assertNotIn("gold", missing.output.lower())


if __name__ == "__main__":
    unittest.main()
