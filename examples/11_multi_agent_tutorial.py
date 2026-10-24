"""Many agents, one job: eight ways to build a multi-agent workflow.

    export OLLAMA_API_KEY=...        # or set OPENAI_API_KEY and use OpenAICompatible below
    python examples/11_multi_agent_tutorial.py

Three small agents, each good at one thing:
    adder       - can only add (has the add tool)
    subtractor  - can only subtract (has the subtract tool)
    checker     - checks someone else's arithmetic
"""

import re
import time

from taskpilot import Agent, Flow, GroupChat, Handoff, Ollama, ReviewLoop, Router, Supervisor, tool

llm = Ollama("gpt-oss:120b", options={"temperature": 0})
# llm = OpenAICompatible("gpt-4o-mini")      # OpenAI works the same way


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


NUMBER_ONLY = "Always use your tool. Reply with the final number only."
adder = Agent(llm, tools=[add_numbers], name="adder", system=f"You add numbers. {NUMBER_ONLY}")
subtractor = Agent(llm, tools=[subtract_numbers], name="subtractor", system=f"You subtract numbers. {NUMBER_ONLY}")
checker = Agent(llm, name="checker", system="You check arithmetic. Reply exactly 'correct' or 'wrong: <right answer>'.")


def number(text):
    """Agents reply with text; pull the number out of it."""
    found = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(found[-1]) if found else None


def title(text):
    print(f"\n=== {text} ===")


# ----------------------------------------------------- 1. agents in a row
title("1. Ek ke baad ek: adder -> subtractor   (a + b) - c")
flow = Flow("pipeline")
flow.add(adder.as_step("total", "Add {a} and {b}."))                 # agent = ek step
flow.add(subtractor.as_step("final", "Subtract {c} from {total}."))  # {total} = pichhle agent ka jawab
flow.add(number, name="answer", bind={"text": "final"})              # normal function bhi step
r = flow.run(a=125, b=75, c=40)
print("total:", r["total"], "| final:", r["final"], "| answer:", r.output)


# ----------------------------------------------------- 2. agents at the same time
title("2. Saath-saath: adder aur subtractor ek hi waqt")
flow = Flow("parallel")
flow.add(adder.as_step("sum", "Add {a} and {b}."))
flow.add(subtractor.as_step("difference", "Subtract {b} from {a}."))
t = time.perf_counter()
r = flow.run(a=500, b=120)
print(r.output, f"| took {time.perf_counter() - t:.1f}s (dono agents parallel)")


# ----------------------------------------------------- 3. pick the right agent
title("3. Router: sawaal dekh ke sahi agent chuno")
desk = Router(
    lambda question: "subtract" if any(w in question.lower() for w in ("minus", "subtract", "less")) else "add",
    {"add": adder.as_step("add", "{question}"), "subtract": subtractor.as_step("subtract", "{question}")},
)
for q in ("What is 19 plus 23?", "What is 100 minus 58?"):
    r = desk.run(question=q)
    print(f"{q} -> agent '{r['route']}' -> {r.output}")


# ----------------------------------------------------- 4. a boss agent
title("4. Supervisor: boss agent khud kaam baante")
boss = Supervisor(llm, [adder, subtractor], system="Never calculate yourself. Reply with the final number only.")
r = boss.run("What is (125 + 75) - 40?")
print(r.output, "| boss ne kisko bheja:", r.tools_used)


# ----------------------------------------------------- 5. pass the conversation
title("5. Handoff: reception agent sahi specialist ko de deta hai")
reception = Agent(llm, name="reception",
                  system="You never calculate. Transfer addition to adder and subtraction to subtractor.")
team = Handoff([reception, adder, subtractor])
r = team.run("Please subtract 37 from 90.")
print(r.output, "| path:", " -> ".join(r.path))


# ----------------------------------------------------- 6. one solves, one checks
title("6. ReviewLoop: subtractor solve kare, checker jaanche, galat ho toh dobara")


def solve(question, feedback=None):
    hint = f" A checker said your last answer was {feedback}." if feedback else ""
    return subtractor(question + hint)


def check(answer, question):
    return checker(f"Question: {question}\nAnswer given: {answer}\nIs it correct?")


r = ReviewLoop(solve, check, until=lambda verdict: verdict.strip().lower().startswith("correct"),
               max_iterations=3).run(question="What is 1000 minus 1?")
print("answer:", r.output, "| checker:", r["review"], "| rounds:", r.loops["improve"]["rounds"])


# ----------------------------------------------------- 7. a person in between
title("7. Do agents ke beech insaan ki approval")
flow = Flow("approve_between")
flow.add(adder.as_step("total", "Add {a} and {b}."))


@flow.step
def approval(total, ctx):
    if number(total) <= 1000:
        return True
    return ctx.wait_for("go", question=f"Total {total} hai. Kya subtract karein?")


flow.add(subtractor.as_step("final", "Subtract {c} from {total}."), after=["approval"])
r = flow.run(a=900, b=300, c=200)
print("status:", r.status, "| sawaal:", r.pending[0].question)
r = flow.resume(r, go=True)              # adder dobara nahi chalta
print("after approval:", r["final"], "| adder resumed:", r.steps["total"].resumed)


# ----------------------------------------------------- 8. agents talk
title("8. GroupChat: teeno ek hi baat-cheet mein")


def checker_agrees(messages):            # baat-cheet kab khatam ho: checker "correct" bole
    last = messages[-1]
    return last["speaker"] == "checker" and last["content"].strip().lower().startswith("correct")


talk = GroupChat([adder, subtractor, checker], rounds=6, until=checker_agrees, finishers=["checker"])
r = talk.run("Compute 50 + 25 - 10. adder does the addition, subtractor the subtraction, then checker confirms.")
for m in r.messages[1:]:
    print(f"  {m['speaker']:>10}: {m['content'][:80]}")
print("stopped by:", r.stopped_by)
