"""Human-in-the-loop: pause with ctx.wait_for, resume later. Easy to complex."""

import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest

from taskpilot import Flow, FlowDefinitionError, Retry, StepError, TaskPilotError
from taskpilot.cli import main as cli_main


def approval_flow(counter=None):
    counter = counter if counter is not None else {}
    flow = Flow("publish")

    @flow.step
    def draft(topic):
        counter["draft"] = counter.get("draft", 0) + 1
        return f"Post about {topic}"

    @flow.step
    def approval(draft, ctx):
        return ctx.wait_for("approve", question="Publish this?", draft=draft)

    @flow.step
    def publish(draft, approval):
        return f"LIVE: {draft}" if approval else "rejected"

    return flow


class Easy(unittest.TestCase):
    def test_pause_then_resume(self):
        flow = approval_flow()
        r = flow.run(topic="AI")
        self.assertTrue(r.paused)
        self.assertEqual(r.status, "paused")
        self.assertFalse(r.ok)
        self.assertEqual(r.pending[0].key, "approve")
        self.assertEqual(r.pending[0].question, "Publish this?")
        self.assertEqual(r.pending[0].data, {"draft": "Post about AI"})
        self.assertIsNone(r.output)
        r2 = flow.resume(r, approve=True)
        self.assertTrue(r2.ok)
        self.assertEqual(r2.output, "LIVE: Post about AI")

    def test_answer_decides_branch(self):
        flow = approval_flow()
        self.assertEqual(flow.resume(flow.run(topic="AI"), approve=False).output, "rejected")

    def test_finished_steps_not_rerun(self):
        counter = {}
        flow = approval_flow(counter)
        flow.resume(flow.run(topic="AI"), {"approve": True})
        self.assertEqual(counter["draft"], 1)

    def test_answered_tells_if_wait_for_would_pause(self):
        flow = Flow()

        @flow.step
        def ask(ctx):
            return "known" if ctx.answered("name") else ctx.wait_for("name")

        r = flow.run()
        self.assertTrue(r.paused)
        self.assertEqual(flow.resume(r, name="Asha").output, "known")


class Medium(unittest.TestCase):
    def test_save_and_resume_from_file(self):
        flow = approval_flow()
        with tempfile.TemporaryDirectory() as d:
            path = flow.run(topic="AI").save(os.path.join(d, "sub", "run.json"))
            with open(path) as fh:
                self.assertEqual(json.load(fh)["pending"][0]["key"], "approve")
            r = approval_flow().resume(path, approve=True)  # a fresh Flow object, like a new process
        self.assertEqual(r.output, "LIVE: Post about AI")

    def test_typo_in_answer_key(self):
        flow = approval_flow()
        with self.assertRaises(FlowDefinitionError) as cm:
            flow.resume(flow.run(topic="AI"), aprove=True)
        self.assertIn("Did you mean 'approve'", str(cm.exception))

    def test_wrong_flow_or_bad_file(self):
        r = approval_flow().run(topic="AI")
        other = Flow("other")
        with self.assertRaises(FlowDefinitionError):
            other.resume(r, approve=True)
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.json")
            with open(bad, "w") as fh:
                fh.write("{}")
            with self.assertRaises(TaskPilotError):
                approval_flow().resume(bad, approve=True)
        with self.assertRaises(TypeError):
            approval_flow().resume(42)

    def test_waiting_is_never_retried_or_replaced_by_fallback(self):
        calls = []
        flow = Flow()

        @flow.step(retry=Retry(5, backoff=0), fallback="fallback!")
        def ask(ctx):
            calls.append(1)
            return ctx.wait_for("q")

        r = flow.run()
        self.assertTrue(r.paused)
        self.assertEqual(len(calls), 1)
        self.assertEqual(flow.resume(r, q="answer").output, "answer")

    def test_events_and_summary(self):
        flow = approval_flow()
        seen = []
        flow.on("step_wait", lambda e: seen.append((e.step, e.key)))
        r = flow.run(topic="AI")
        self.assertEqual(seen, [("approval", "approve")])
        self.assertIn("waiting", r.summary())
        self.assertEqual(r.to_dict()["status"], "paused")

    def test_raises_still_work_while_paused_elsewhere(self):
        flow = Flow()
        flow.add(lambda ctx: ctx.wait_for("q"), name="ask")
        flow.add(lambda: 1 / 0, name="bad")
        with self.assertRaises(StepError):
            flow.run()


class High(unittest.TestCase):
    def test_independent_branches_finish_before_pausing(self):
        flow = Flow()
        flow.add(lambda ctx: ctx.wait_for("ok"), name="gate")
        flow.add(lambda: "report ready", name="report")
        flow.add(lambda gate, report: f"{report} / {gate}", name="send")
        r = flow.run()
        self.assertEqual(r["report"], "report ready")
        self.assertEqual(r.steps["send"].status, "pending")
        self.assertEqual(flow.resume(r, ok="sent").output, "report ready / sent")

    def test_two_questions_answered_in_two_rounds(self):
        flow = Flow()
        flow.add(lambda ctx: ctx.wait_for("legal"), name="legal")
        flow.add(lambda ctx: ctx.wait_for("budget"), name="budget")
        flow.add(lambda legal, budget: legal and budget, name="go")
        r = flow.run()
        self.assertEqual(sorted(p.key for p in r.pending), ["budget", "legal"])
        r = flow.resume(r, legal=True)
        self.assertEqual([p.key for p in r.pending], ["budget"])
        r = flow.resume(r, budget=True)
        self.assertTrue(r.ok)
        self.assertTrue(r.output)

    def test_human_inside_a_loop_one_key_per_round(self):
        flow = Flow()

        @flow.step
        def draft(note=None, ctx=None):
            return f"draft {ctx.iteration}" + (f" ({note})" if note else "")

        @flow.step
        def feedback(draft, ctx):
            return ctx.wait_for(f"review-{ctx.iteration}", question=f"OK? {draft}")

        flow.loop("draft", "feedback", until=lambda feedback: feedback == "ok", feed={"note": "feedback"})
        r = flow.run()
        self.assertEqual(r.pending[0].key, "review-1")
        r = flow.resume(r, **{"review-1": "shorter please"})
        self.assertEqual(r.pending[0].key, "review-2")
        self.assertEqual(r.pending[0].question, "OK? draft 2 (shorter please)")
        r = flow.resume(r, **{"review-2": "ok"})
        self.assertTrue(r.ok)
        self.assertEqual(r.loops["draft_loop"]["rounds"], 2)

    def test_map_items_can_ask(self):
        flow = Flow()

        @flow.step(map="item")
        def check(item, ctx):
            if item == "risky":
                return ctx.wait_for("risky-ok")
            return True

        r = flow.run(item=["a", "risky", "b"])
        self.assertTrue(r.paused)
        self.assertEqual(flow.resume(r, **{"risky-ok": False}).output, [True, False, True])

    def test_async_steps_and_aresume(self):
        flow = Flow()

        @flow.step
        async def ask(ctx):
            await asyncio.sleep(0)
            return ctx.wait_for("color")

        async def main():
            r = await flow.arun()
            return (await flow.aresume(r, color="blue")).output

        self.assertEqual(asyncio.run(main()), "blue")

    def test_state_survives_pause(self):
        flow = Flow()

        @flow.step
        def a(ctx):
            ctx.set("audit", ["a"])
            return 1

        @flow.step
        def b(a, ctx):
            ok = ctx.wait_for("ok")
            ctx.update("audit", lambda v: v + ["b"])
            return ok

        with tempfile.TemporaryDirectory() as d:
            path = flow.run().save(os.path.join(d, "r.json"))
            r = flow.resume(path, ok=True)
        self.assertEqual(r.state["audit"], ["a", "b"])


class Complex(unittest.TestCase):
    def test_resume_in_another_process(self):
        code = textwrap.dedent('''
            from taskpilot import Flow
            flow = Flow("expense")

            @flow.step
            def amount(raw):
                return round(float(raw), 2)

            @flow.step(when=lambda amount: amount > 1000)
            def manager(amount, ctx):
                return ctx.wait_for("manager", question=f"Approve {amount}?")

            @flow.step
            def decision(amount, manager=None):
                return "auto-approved" if manager is None else ("approved" if manager else "denied")
        ''')
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "expense.py")
            with open(src, "w") as fh:
                fh.write(code)
            paused = os.path.join(d, "paused.json")
            env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
            run = [sys.executable, "-m", "taskpilot"]
            p1 = subprocess.run(run + ["run", src, "raw=5000", "--save", paused], capture_output=True, text=True, env=env)
            self.assertEqual(p1.returncode, 3, p1.stderr)
            self.assertIn("Approve 5000.0?", p1.stderr)
            p2 = subprocess.run(run + ["resume", src, paused, "manager=true"], capture_output=True, text=True, env=env)
            self.assertEqual(p2.returncode, 0, p2.stderr)
            self.assertEqual(p2.stdout.strip(), "approved")
            p3 = subprocess.run(run + ["run", src, "raw=10"], capture_output=True, text=True, env=env, cwd=d)
            self.assertEqual(p3.stdout.strip(), "auto-approved")

    def test_many_concurrent_paused_runs(self):
        flow = approval_flow()
        results = []
        lock = threading.Lock()

        def worker(i):
            r = flow.run(topic=f"T{i}")
            r = flow.resume(r, approve=i % 2 == 0)
            with lock:
                results.append((i, r.output))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, out in results:
            self.assertEqual(out, f"LIVE: Post about T{i}" if i % 2 == 0 else "rejected")

    def test_unpicklable_value_reports_cleanly(self):
        flow = Flow()
        flow.add(lambda: threading.Lock(), name="lock")
        flow.add(lambda lock, ctx: ctx.wait_for("x"), name="ask")
        r = flow.run()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TaskPilotError) as cm:
                r.save(os.path.join(d, "r.json"))
            self.assertIn("step 'lock'", str(cm.exception))
            self.assertEqual(os.listdir(d), [], "no half-written file left behind")
        self.assertEqual(flow.resume(r, x=1).output, 1)  # in-memory resume still works

    def test_cli_resume_wrong_key(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "f.py")
            with open(src, "w") as fh:
                fh.write("from taskpilot import Flow\nflow = Flow('f')\nflow.add(lambda ctx: ctx.wait_for('yes'), name='ask')\n")
            paused = os.path.join(d, "p.json")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["run", src, "--save", paused]), 3)
                self.assertEqual(cli_main(["resume", src, paused, "yse=1"]), 2)
            self.assertIn("Did you mean 'yes'", err.getvalue())


if __name__ == "__main__":
    unittest.main()
