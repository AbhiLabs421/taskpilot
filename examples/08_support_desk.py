"""Support desk: from a customer's complaint to a checked reply, with a manager's OK for big refunds.

    export OLLAMA_API_KEY=...
    python examples/08_support_desk.py "My order A7 arrived broken, I want my 2499 rupees back"
    python examples/08_support_desk.py --approve yes      # answer the manager question

What it shows in ~50 lines: parallel steps, an LLM returning JSON, a
write -> check loop, a human approval that pauses the run, and tracing.
"""

import sys

from taskpilot import Flow, Ollama
from taskpilot.tracing import ConsoleTracer

llm = Ollama("gpt-oss:120b", options={"temperature": 0})
ORDERS = {"A7": {"item": "ceramic tea set", "amount": 2499}, "B2": {"item": "phone case", "amount": 299}}

flow = Flow("support_desk", verbose=True)


@flow.step(retry=3)
def ticket(message):
    """Understand the complaint (the LLM answers in JSON)."""
    return llm.json(
        f"Read this customer message and extract the details.\n\n{message}",
        schema={"type": "object", "required": ["category", "order_id"],
                "properties": {"category": {"type": "string", "enum": ["refund", "delivery", "other"]},
                               "order_id": {"type": "string"}}},
    )


@flow.step
def mood(message):
    """Runs at the same time as `ticket`: both only need the message."""
    return llm.ask(f"One word for the customer's mood: {message}").strip(" .").lower()


@flow.step
def order(ticket):
    return ORDERS.get(ticket["order_id"].upper(), {"item": "unknown", "amount": 0})


@flow.step
def reply(ticket, order, mood, feedback=None):
    note = f" Fix this: {feedback['tip']}" if feedback else ""
    return llm.ask(f"Write a 2-sentence reply to a {mood} customer about a {ticket['category']} "
                   f"for their {order['item']} (Rs {order['amount']}). Be warm and specific.{note}")


@flow.step
def check(reply):
    return llm.json(f"Is this support reply polite, specific and under 60 words?\n\n{reply}",
                    schema={"type": "object", "required": ["ok", "tip"],
                            "properties": {"ok": {"type": "boolean"}, "tip": {"type": "string"}}})


flow.loop("reply", "check", until=lambda check: check["ok"], feed={"feedback": "check"}, max_iterations=3)


@flow.step(when=lambda order: order["amount"] > 1000)
def manager(order, reply, ctx):
    """Big refunds wait for a person. The run pauses here."""
    return ctx.wait_for("refund_ok", question=f"Approve refund of Rs {order['amount']}?", reply=reply)


@flow.step
def send(reply, manager=True):
    return f"SENT: {reply}" if manager else "Escalated to a senior agent."


if __name__ == "__main__":
    flow.trace(ConsoleTracer())
    if sys.argv[1:2] == ["--approve"]:
        result = flow.resume("support_run.json", refund_ok=sys.argv[2].lower() in ("y", "yes", "true"))
    else:
        message = " ".join(sys.argv[1:]) or "My order A7 arrived broken, I want my 2499 rupees back!"
        result = flow.run(message=message)
    if result.paused:
        result.save("support_run.json")
        print(f"\nPAUSED: {result.pending[0].question}  ->  run again with: --approve yes")
    else:
        print("\n" + result.output)
