"""Workspace + web page: tools, agents, workflow types, approvals. Easy to complex."""

import json
import os
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.error
import urllib.request

from taskpilot import Agent, FakeLLM, FlowDefinitionError, TaskPilotError, Workspace, tool
from taskpilot.web import WebApp


@tool
def get_order(order_id: str) -> dict:
    """Look up an order."""
    return {"A7": {"customer": "Ravi", "email": "ravi@example.com", "amount": 2499}}.get(order_id, {"error": "none"})


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    SENT.append((to, body))
    return f"sent to {to}"


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


SENT = []


def brain(messages, tools):
    """A scripted model good enough to drive every workflow type."""
    text = messages[-1]["content"]
    names = [t["function"]["name"] for t in tools or []]
    done_tool = any(m.get("role") == "tool" for m in messages)
    if "Choose how to do" in text:
        return '{"pattern": "review", "agents": ["writer", "checker"], "why": "needs a check"}'
    if "Which one helper" in text:
        return '{"agent": "checker", "why": "it checks"}'
    if "Work to check" in text:
        return "OK" if "polite" in text else "Please be more polite."
    if "send_email" in names and "approved by the reviewer" in text and not done_tool:
        return {"tool": "send_email", "args": {"to": "ravi@example.com", "body": "Thanks, Ravi (polite)"}}
    if "send_email" in names and "send it now" in text and not done_tool:
        return {"tool": "send_email", "args": {"to": "ravi@example.com", "body": "hello"}}
    if "reviewer said" in text.lower():
        return "Thanks, Ravi (polite)"
    if "Combine them" in text:
        return "combined answer"
    return "draft: " + text.splitlines()[0][:40]


def workspace(**kw):
    SENT.clear()
    kw.setdefault("review", False)
    kw.setdefault("verify", False)
    ws = Workspace(FakeLLM(fn=brain), [get_order, send_email, add_numbers], approve=["send_email"], **kw)
    ws.create_agent("writer", "You write customer emails.", ["get_order", "send_email"])
    ws.create_agent("checker", "You check emails.")
    return ws


class Easy(unittest.TestCase):
    def test_overview(self):
        ws = workspace()
        o = ws.overview()
        self.assertEqual([t["name"] for t in o["tools"]], ["get_order", "send_email", "add_numbers"])
        self.assertTrue(next(t for t in o["tools"] if t["name"] == "send_email")["approval"])
        self.assertEqual([a["name"] for a in o["agents"]], ["writer", "checker"])
        self.assertEqual(o["patterns"][0]["key"], "plan")
        self.assertEqual(len(o["patterns"]), 10)

    def test_single_and_sequential(self):
        ws = workspace()
        r = ws.start("Say hi", "single", ["checker"])
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.output, "draft: Say hi")
        r = ws.start("Say hi", "sequential", ["writer", "checker"])
        self.assertEqual([t["who"] for t in r.trace], ["writer", "checker"])
        self.assertIn("Work so far (from writer)", ws.llm.calls[-1]["messages"][-1]["content"])

    def test_code_agents(self):
        helper = Agent(FakeLLM(["hi"]), tools=[add_numbers], name="helper", system="Helps.")
        ws = Workspace(FakeLLM(["x"]), [], agents=[helper, {"name": "notes", "instructions": "Take notes."}], review=False, verify=False)
        self.assertEqual(ws.agents["helper"].source, "code")
        self.assertIn("add_numbers", ws.tools, "a code agent's tools join the workspace")
        self.assertEqual(ws.start("go", "single", ["helper"]).output, "hi")


class Medium(unittest.TestCase):
    def test_agent_rules(self):
        ws = workspace()
        bad = [("Writer Two", "x", []), ("writer", "x", []), ("helper", "x", ["delete_db"]), ("send_email", "x", []),
               ("ok_name", "x" * 5000, [])]
        for name, instr, tools in bad:
            with self.subTest(name=name):
                with self.assertRaises(FlowDefinitionError):
                    ws.create_agent(name, instr, tools)
        ws.update_agent("checker", instructions="Check strictly.", tools=["get_order"])
        self.assertEqual(ws.agents["checker"].tools, ["get_order"])
        ws.delete_agent("checker")
        self.assertNotIn("checker", ws.agents)
        with self.assertRaises(FlowDefinitionError) as cm:
            ws.delete_agent("writr")
        self.assertIn("Did you mean 'writer'", str(cm.exception))

    def test_code_agents_are_read_only(self):
        ws = Workspace(FakeLLM(["x"]), [], agents=[{"name": "boss", "instructions": "Lead."}], review=False, verify=False)
        with self.assertRaises(FlowDefinitionError):
            ws.delete_agent("boss")
        with self.assertRaises(FlowDefinitionError):
            ws.update_agent("boss", instructions="other")

    def test_choice_checks(self):
        ws = workspace()
        self.assertIn("at least 2", ws.check_choice("review", ["writer"])[0])
        self.assertIn("at most 2", ws.check_choice("review", ["writer", "checker", "writer"])[-1])
        self.assertIn("unknown agent", ws.check_choice("single", ["ghost"])[0])
        self.assertIn("unknown workflow type", ws.check_choice("magic", [])[0])
        with self.assertRaises(FlowDefinitionError):
            ws.start("x", "parallel", ["writer"])
        with self.assertRaises(ValueError):
            ws.start("  ", "single", ["writer"])

    def test_suggest_and_fallback(self):
        ws = workspace()
        self.assertEqual(ws.suggest("write and check an email"),
                         {"pattern": "review", "agents": ["writer", "checker"], "why": "needs a check", "items": []})
        bad = Workspace(FakeLLM(['{"pattern": "review", "agents": ["ghost"], "why": "x"}']), [add_numbers], review=False, verify=False)
        tip = bad.suggest("x")
        self.assertEqual(tip["pattern"], "plan")
        self.assertIn("did not fit", tip["why"])

    def test_agent_actions_wait_for_approval(self):
        ws = workspace()
        r = ws.start("send it now", "single", ["writer"])
        self.assertEqual(r.status, "waiting")
        self.assertEqual(SENT, [], "nothing is sent before approval")
        self.assertEqual(r.actions[0]["tool"], "send_email")
        r = ws.approve_action(r.id, "a1", True)
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.actions[0]["status"], "done")
        self.assertEqual(SENT, [("ravi@example.com", "hello")])
        with self.assertRaises(TaskPilotError):
            ws.approve_action(r.id, "a1", True)
        r2 = ws.start("send it now", "single", ["writer"])
        r2 = ws.approve_action(r2.id, "a1", False)
        self.assertEqual(r2.actions[0]["status"], "rejected")
        self.assertEqual(len(SENT), 1)

    def test_store_keeps_agents_and_workflows(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ws.json")
            ws = workspace(store=path)
            run = ws.start("Say hi", "single", ["checker"])
            ws.save_workflow("Greeting", run.id)
            again = Workspace(FakeLLM(fn=brain), [get_order, send_email, add_numbers], approve=["send_email"], store=path, review=False, verify=False)
            self.assertEqual(list(again.agents), ["writer", "checker"])
            self.assertEqual(again.run_workflow("Greeting").output, "draft: Say hi")
            again.delete_workflow("Greeting")
            self.assertEqual(again.workflows, {})
            fewer = Workspace(FakeLLM(["x"]), [get_order], store=path, review=False, verify=False)
            self.assertEqual(fewer.agents["writer"].tools, ["get_order"], "tools removed from code are dropped")


class High(unittest.TestCase):
    def test_review_drafts_without_actions_then_acts_after_ok(self):
        ws = workspace()
        r = ws.start("Thank Ravi for order A7 by email", "review", ["writer", "checker"])
        self.assertEqual(r.status, "waiting")
        whos = [t["who"] for t in r.trace]
        self.assertEqual(whos, ["writer", "checker", "writer", "checker", "writer (after the check)"])
        self.assertEqual(r.output, "Thanks, Ravi (polite)")
        drafting_tools = [c["tools"] for c in ws.llm.calls if c["messages"][-1]["content"].startswith("Thank Ravi")]
        self.assertTrue(all("send_email" not in [t["function"]["name"] for t in (ts or [])] for ts in drafting_tools[:2]))
        self.assertEqual(SENT, [])
        ws.approve_action(r.id, "a1", True)
        self.assertEqual(SENT, [("ravi@example.com", "Thanks, Ravi (polite)")])

    def test_review_never_acts_when_not_approved(self):
        ws = Workspace(FakeLLM(fn=lambda m, t: "Please be more polite." if "Work to check" in m[-1]["content"] else "draft"),
                       [send_email], approve=["send_email"], review=False, verify=False)
        ws.create_agent("writer", "w", ["send_email"])
        ws.create_agent("checker", "c")
        r = ws.start("email", "review", ["writer", "checker"], rounds=2)
        self.assertEqual(r.actions, [])
        self.assertIn("nothing was done", r.trace[-1]["text"])

    def test_parallel_router_map(self):
        def slow(messages, tools):
            text = messages[-1]["content"]
            if "Which one helper" in text:
                return '{"agent": "agent_b", "why": "fits"}'
            if "Combine" in text:
                return "combined"
            time.sleep(0.2)
            return "done " + text[-6:]

        ws = Workspace(FakeLLM(fn=slow), [], review=False, verify=False)
        for n in ("agent_a", "agent_b", "agent_c"):
            ws.create_agent(n, f"helper {n}")
        t = time.perf_counter()
        r = ws.start("question", "parallel", ["agent_a", "agent_b", "agent_c"])
        self.assertLess(time.perf_counter() - t, 0.5, "three agents at once")
        self.assertEqual(r.output, "combined")
        r = ws.start("question", "router", ["agent_a", "agent_b"])
        self.assertEqual(r.trace[0]["text"], "chose agent_b: fits")
        t = time.perf_counter()
        r = ws.start("Summarise", "map", ["agent_a"], items=["one", "two", "three"])
        self.assertLess(time.perf_counter() - t, 0.5)
        self.assertEqual(r.output.count("\n"), 2)
        bad = ws.start("Summarise", "map", ["agent_a"])
        self.assertEqual(bad.status, "failed")
        self.assertIn("one item per line", bad.error)

    def test_team_types(self):
        ws = workspace()
        for pattern, agents in [("supervisor", ["writer"]), ("handoff", ["writer", "checker"]), ("group_chat", ["writer", "checker"])]:
            with self.subTest(pattern=pattern):
                r = ws.start("Say hi", pattern, agents)
                self.assertIn(r.status, ("ok", "waiting"))
                self.assertTrue(r.trace)

    def test_plan_type_and_saved_plan_without_ai(self):
        plan = json.dumps({"steps": [{"id": "s1", "tool": "add_numbers", "args": {"a": 2, "b": 3}}]})
        ws = Workspace(FakeLLM([plan]), [add_numbers], review=False, verify=False)
        r = ws.start("add 2 and 3", "plan")
        self.assertEqual((r.status, r.output), ("ok", 5))
        ws.save_workflow("Add", r.id)
        calls = len(ws.llm.calls)
        again = ws.run_workflow("Add")
        self.assertEqual(again.output, 5)
        self.assertEqual(len(ws.llm.calls), calls, "a saved plan runs without the model")

    def test_errors_are_reported_not_raised(self):
        def broken(messages, tools):
            raise RuntimeError("model is down")

        ws = Workspace(FakeLLM(fn=broken), [], review=False, verify=False)
        ws.create_agent("writer", "w")
        r = ws.start("x", "single", ["writer"])
        self.assertEqual(r.status, "failed")
        self.assertIn("model is down", r.error)


class Complex(unittest.TestCase):
    def test_web_api_end_to_end(self):
        ws = workspace()
        httpd = WebApp(ws, "Desk").server("127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        try:
            with urllib.request.urlopen(base + "/") as resp:
                page = resp.read().decode()
            self.assertIn("Desk", page)
            self.assertIn("AI decides (recommended)", page)
            code, o = call("/api/workspace")
            self.assertEqual([a["name"] for a in o["agents"]], ["writer", "checker"])
            code, o = call("/api/agents", {"name": "translator", "instructions": "Translate.", "tools": []})
            self.assertEqual(code, 200)
            self.assertIn("translator", [a["name"] for a in o["agents"]])
            self.assertEqual(call("/api/agents", {"name": "Bad Name", "instructions": "x"})[0], 400)
            code, tip = call("/api/suggest", {"request": "write and check"})
            self.assertEqual(tip["pattern"], "review")
            code, run = call("/api/start", {"request": "Thank Ravi for order A7 by email", "pattern": tip["pattern"], "agents": tip["agents"]})
            self.assertEqual(run["status"], "waiting")
            self.assertEqual(run["actions"][0]["status"], "waiting")
            code, run = call("/api/action", {"id": run["id"], "action": "a1", "approve": True})
            self.assertEqual(run["actions"][0]["status"], "done")
            self.assertEqual(run["status"], "ok")
            code, o = call("/api/workflows", {"name": "Thanks mail", "id": run["id"]})
            self.assertEqual(o["workflows"][0]["name"], "Thanks mail")
            code, again = call("/api/workflows/run", {"name": "Thanks mail"})
            self.assertEqual(again["status"], "waiting")
            code, h = call("/api/history")
            self.assertGreaterEqual(len(h["runs"]), 2)
            code, got = call("/api/get", {"id": run["id"]})
            self.assertEqual(got["id"], run["id"])
            self.assertEqual(call("/api/start", {"request": "x", "pattern": "review", "agents": ["writer"]})[0], 400)
            self.assertEqual(call("/api/nope", {})[0], 404)
            code, o = call("/api/agents/delete", {"name": "translator"})
            self.assertNotIn("translator", [a["name"] for a in o["agents"]])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_cli_loads_a_tools_file_with_agents(self):
        from taskpilot.cli import _load_workspace

        code = textwrap.dedent('''
            from taskpilot import FakeLLM, tool

            @tool
            def ping(x: str) -> str:
                """Ping."""
                return x

            tools = [ping]
            approve = ["ping"]
            llm = FakeLLM(["hi"])
            agents = [{"name": "pinger", "tools": ["ping"], "instructions": "Pings."}]
        ''')
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.py")
            with open(path, "w") as fh:
                fh.write(code)
            ws = _load_workspace(path, os.path.join(d, "ws.json"))
            self.assertEqual(list(ws.agents), ["pinger"])
            self.assertEqual(ws.approve, {"ping"})
            ws.create_agent("echo", "Echo.", ["ping"])
            self.assertTrue(os.path.exists(os.path.join(d, "ws.json")))

    def test_many_users_at_once(self):
        ws = workspace()
        results, lock = [], threading.Lock()

        def user(i):
            r = ws.start(f"Say hi {i}", "sequential", ["writer", "checker"])
            with lock:
                results.append(r.status == "ok" and len(r.trace) == 2)

        threads = [threading.Thread(target=user, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 20)
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()
