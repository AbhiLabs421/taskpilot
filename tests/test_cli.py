"""The `taskpilot` command line."""

import contextlib
import io
import os
import tempfile
import textwrap
import unittest

from taskpilot.cli import main

FLOW_FILE = textwrap.dedent('''
    from taskpilot import Flow

    flow = Flow("cli")

    @flow.step
    def total(numbers, bonus=0):
        return sum(numbers) + bonus

    @flow.step
    def label(total, unit):
        return f"{total} {unit}"
''')


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "pipe.py")
        with open(self.path, "w") as fh:
            fh.write(FLOW_FILE)

    def tearDown(self):
        self.dir.cleanup()

    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_run_parses_json_inputs(self):
        code, out, _ = self.call("run", self.path, "numbers=[1,2,3]", "unit=kg", "bonus=4")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "10 kg")

    def test_run_with_named_flow_and_json_output(self):
        code, out, _ = self.call("run", f"{self.path}:flow", "numbers=[1]", "unit=m", "--json")
        self.assertEqual(code, 0)
        self.assertIn('"ok": true', out)

    def test_missing_input_is_reported(self):
        code, _, err = self.call("run", self.path, "numbers=[1]")
        self.assertEqual(code, 2)
        self.assertIn("Missing input 'unit'", err)

    def test_show_graph_version(self):
        self.assertIn("label <- total", self.call("show", self.path)[1])
        self.assertIn("total --> label", self.call("graph", self.path)[1])
        self.assertIn("taskpilot", self.call("version")[1])


if __name__ == "__main__":
    unittest.main()
