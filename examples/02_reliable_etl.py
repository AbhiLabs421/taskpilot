"""Reliability: retry, timeout, cache, parallel map, branching, checkpoint, lean memory.

    python examples/02_reliable_etl.py
"""

import random
import time

from taskpilot import Flow, Retry

flow = Flow("orders-etl", verbose=True)


@flow.step(retry=Retry(5, backoff=0.05), timeout=2)
def orders(day):
    """Download orders (a flaky API)."""
    if random.random() < 0.5:
        raise ConnectionError("API hiccup")      # retried automatically
    return [{"id": i, "amount": random.randint(-5, 500)} for i in range(20)]


@flow.step(map="orders", max_concurrency=8, cache=True)
def checked(orders):
    """Runs once per order, in parallel."""
    time.sleep(0.01)                             # pretend I/O
    return {**orders, "valid": orders["amount"] >= 0}


@flow.step
def good(checked):
    return [o for o in checked if o["valid"]]


@flow.step(when=lambda good: len(good) >= 15)
def full_report(good):
    return f"{len(good)} orders, total {sum(o['amount'] for o in good)}"


@flow.step(when=lambda good: len(good) < 15)
def alert(good):
    return f"Only {len(good)} valid orders - please check the source!"


@flow.step
def message(full_report=None, alert=None):       # defaults = "ok if that branch was skipped"
    return full_report or alert


if __name__ == "__main__":
    result = flow.run(day="2026-10-23", checkpoint="orders.ckpt", lean=True)
    print(result.output)
