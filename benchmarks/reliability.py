"""How often is the AI workflow right? Measured on a real model.

    OLLAMA_API_KEY=... python benchmarks/reliability.py                # today's defaults
    OLLAMA_API_KEY=... python benchmarks/reliability.py --guards       # with the reliability guards
    OLLAMA_API_KEY=... python benchmarks/reliability.py --runs 5

Two suites:
* planner – business calculations planned by AutoFlow (exact numbers expected)
* agent   – questions answered from order data, including two traps where the
            right answer is "I don't know / no such order" (hallucination checks)
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from taskpilot import Agent, AutoFlow, FallbackLLM, Ollama, Unclear, tool  # noqa: E402
from taskpilot.testing import ChaosLLM  # noqa: E402
from taskpilot.evals import Case, evaluate  # noqa: E402
from taskpilot.llm import from_env  # noqa: E402

ORDERS = {
    "A7": {"customer": "Ravi", "email": "ravi@example.com", "item": "tea set", "amount": 2499, "status": "delivered"},
    "B2": {"customer": "Asha", "email": "asha@example.com", "item": "phone case", "amount": 299, "status": "in transit"},
    "C3": {"customer": "Imran", "email": "imran@example.com", "item": "desk lamp", "amount": 1500, "status": "cancelled"},
}


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id (like A7): customer, email, item, amount, status."""
    return ORDERS.get(order_id.strip().upper(), {"error": f"no order with id {order_id}"})


@tool
def order_amount(order_id: str) -> float:
    """The amount (rupees) of an order by id. Fails if the order does not exist."""
    return ORDERS[order_id.strip().upper()]["amount"]


PLANNER_CASES = [
    Case("Add 125 and 75, then subtract 40", 160),
    Case("Sales 125000 aur 75000 thi, returns 40000. Net sales kitni hai?", 160000),
    Case("What is 18% of 2499?", 449.82),
    Case("Multiply 37 by 12 and then add 100", 544),
    Case("Divide 1000 by 8, then subtract 25", 100),
    Case("Order A7 aur B2 ka total amount kitna hai?", 2798),
    Case("What is the amount of order C3 after a 10% discount?", 1350),
    Case("3 items cost 249.5 each, plus 50 shipping. Total?", 798.5),
    Case("Ek mahine mein 30 din, har din 450 rupaye kamaye aur 2000 kharch kiye. Kitna bacha?", 11500),
    Case("What is (500 - 120) * 3?", 1140),
    Case("Order B2 ka amount double karo aur 1 ghatao", 597),
    Case("Total of orders A7, B2 and C3, then take off a 5% discount", 4083.1),
    Case("Divide 100 by 3", 100 / 3),
    Case("Increase 2000 by 15 percent", 2300),
    Case("Average of 10, 20 and 60", 30),
]


def has(*words):
    return lambda out: all(w in str(out).lower() for w in words) or f"missing {words} in {str(out)[:80]!r}"


NO_SUCH = ("not found", "no order", "doesn't exist", "does not exist", "couldn't find", "could not find", "no such",
           "unable to find", "isn't", "not exist", "no record")
MONTHS = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{1,2}/\d{1,2}\b|tomorrow|\bin \d+ days?", re.I)


def no_such(out):
    text = str(out).lower()
    return any(p in text for p in NO_SUCH) or f"should say the order does not exist: {text[:80]!r}"


def no_invented_date(out):
    text = str(out).lower()
    if "phone case" not in text:
        return f"missing 'phone case': {text[:80]!r}"
    if MONTHS.search(text):
        return f"invented an arrival date: {text[:100]!r}"
    return True


AGENT_CASES = [
    Case("Which item did the customer of order A7 buy?", has("tea set")),
    Case("What is the email of the customer of order B2?", has("asha@example.com")),
    Case("What is the status of order C3?", has("cancel")),
    Case("Who placed order A7?", has("ravi")),
    Case("How much did order C3 cost?", lambda o: "1500" in str(o).replace(",", "") or f"no 1500 in {str(o)[:80]!r}"),
    Case("Has order B2 been delivered yet?", has("transit")),
    Case("What is the status of order Z9?", no_such, name="TRAP: order Z9 does not exist"),
    Case("What did Asha order (order B2), and on which date will it arrive?", no_invented_date, name="TRAP: no arrival date in the data"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--guards", action="store_true", help="turn on the reliability guards")
    ap.add_argument("--only", choices=["planner", "agent"])
    ap.add_argument("--reviewers", default="gemma4:31b,nemotron-3-super",
                    help="comma-separated Ollama models that review plans (different from the planner)")
    ap.add_argument("--model", default=None, help="model doing the work (default: TASKPILOT_MODEL or gpt-oss:120b)")
    ap.add_argument("--checker", default="nemotron-3-super", help="model that verifies agent answers")
    ap.add_argument("--escalate", default="gpt-oss:120b", help="stronger model used when checks keep failing")
    ap.add_argument("--chaos", type=float, default=0.0,
                    help="make the working model misbehave: chance of a changed number (outages at half this rate)")
    args = ap.parse_args()
    llm = from_env(args.model)
    if hasattr(llm, "options"):
        llm.options.setdefault("temperature", 0)
    strong = Ollama(args.escalate, options={"temperature": 0})
    worker = llm
    if args.chaos:
        worker = ChaosLLM(llm, numbers=args.chaos, outage=args.chaos / 2, seed=7)
        print(f"chaos: {args.chaos:.0%} of replies get a changed number, {args.chaos / 2:.0%} fail like an outage")
    if args.guards:
        worker = FallbackLLM([worker, llm])  # an outage falls through to the healthy model
    print(f"model: {llm!r} | guards: {'on' if args.guards else 'off'} | runs per case: {args.runs}")
    if args.guards:
        print(f"plan reviewers: {args.reviewers} | answer checker: {args.checker} | escalate to: {args.escalate}")
    print()

    if args.only in (None, "planner"):
        extra = {"review": [Ollama(m, options={"temperature": 0}) for m in args.reviewers.split(",")],
                 "escalate": strong} if args.guards else {}
        auto = AutoFlow(worker, [add, subtract, multiply, divide, order_amount], allow_human=False, **extra)
        report = evaluate(lambda q: auto.run(q).output, PLANNER_CASES, runs=args.runs, name="planner (AutoFlow)",
                          abstain_on=(Unclear,))
        print(report.summary(), "\n")

    if args.only in (None, "agent"):
        extra = {"verify": Ollama(args.checker, options={"temperature": 0}), "escalate": strong} if args.guards else {}
        system = "You answer questions about orders. Use the tools. Be brief."
        def ask(q):
            r = Agent(worker, tools=[get_order], system=system, **extra).run(q)
            # An answer that failed verification is shown flagged, never as a trusted answer.
            return r.output if r.verified is not False else "UNVERIFIED: " + r.output

        report = evaluate(ask, AGENT_CASES, runs=args.runs, name="agent (order questions)",
                          abstain=lambda o: str(o).startswith("UNVERIFIED"))
        print(report.summary())


if __name__ == "__main__":
    main()
