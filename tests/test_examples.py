"""The beginner examples must always run (no network needed)."""

import os
import subprocess
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def run_example(name, *args):
    env = {k: v for k, v in os.environ.items() if k not in ("OLLAMA_API_KEY", "OPENAI_API_KEY")}
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    return subprocess.run([sys.executable, os.path.join(ROOT, "examples", name), *args],
                          capture_output=True, text=True, env=env, timeout=120)


class TestBeginnerExamples(unittest.TestCase):
    def test_calc(self):
        p = run_example("calc.py")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "{'add': 14, 'subtract': 6}")

    def test_calculator_tutorial(self):
        p = run_example("10_calculator_tutorial.py")
        self.assertEqual(p.returncode, 0, p.stderr)
        for expected in ["{'add': 14, 'subtract': 6}", "[3, 7, 11] -> total 21", "[70, 40, 10, -20] -> final -20",
                         "resume: saved 1100", "[3, 7, 30]", "5 + 3 - 2 = 6", "Answer: 11"]:
            self.assertIn(expected, p.stdout)

    def test_connect_platforms(self):
        p = run_example("12_connect_platforms.py")
        self.assertEqual(p.returncode, 0, p.stderr)
        for expected in ["tools from the platform: ['find_customer', 'create_ticket']", "MCP tool: {'name': 'Ravi', 'tier': 'gold'}",
                         "REST tool: {'ticket': 'T-1001'", "A2A agent: Report:", "used: ['support', 'reporter']"]:
            self.assertIn(expected, p.stdout)

    def test_hello(self):
        p = run_example("01_hello.py")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("NAMASTE, ASHA!", p.stdout)


if __name__ == "__main__":
    unittest.main()
