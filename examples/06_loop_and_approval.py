"""A writing workflow: improve a draft in a loop, then wait for a human to approve.

    python examples/06_loop_and_approval.py             # runs, pauses, saves run.json
    python examples/06_loop_and_approval.py yes         # resumes with the answer
"""

import os
import sys

from taskpilot import Flow

flow = Flow("newsletter", verbose=True)


@flow.step
def draft(topic, feedback=None, ctx=None):
    text = f"{topic}: draft {ctx.iteration}"
    return text + (f" (fixed: {feedback['tip']})" if feedback else "")


@flow.step
def review(draft, ctx):
    good = ctx.iteration >= 3                   # pretend the 3rd draft is good
    return {"ok": good, "tip": "shorter intro" if not good else ""}


flow.loop("draft", "review", until=lambda review: review["ok"], feed={"feedback": "review"}, max_iterations=5)


@flow.step
def approval(draft, review, ctx):
    return ctx.wait_for("publish", question=f"Publish this?\n  {draft}")


@flow.step
def publish(draft, approval):
    return f"Sent: {draft}" if approval else "Kept as draft"


if __name__ == "__main__":
    saved = "run.json"
    if len(sys.argv) > 1 and os.path.exists(saved):
        result = flow.resume(saved, publish=sys.argv[1].lower() in ("y", "yes", "true"))
        os.remove(saved)
        print(result.output)
    else:
        result = flow.run(topic="Monsoon offers")
        print(f"\nWaiting: {result.pending[0].question}")
        result.save(saved)
        print(f"Saved to {saved}. Run again with 'yes' or 'no'.")
