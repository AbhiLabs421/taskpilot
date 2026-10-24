"""Teams of agents: a support desk that hands off, and a supervisor that delegates.

    export OLLAMA_API_KEY=...
    python examples/07_agent_teams.py
"""

from taskpilot import Agent, Handoff, Ollama, Supervisor, tool

llm = Ollama("gpt-oss:120b", options={"temperature": 0})


@tool
def order_status(order_id: str) -> str:
    """Status of an order by its id."""
    return {"A7": "refunded on 21 Oct", "B2": "out for delivery"}.get(order_id.upper(), "not found")


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers exactly."""
    return a * b


triage = Agent(llm, name="triage", system="You route customers. Never answer yourself: transfer to billing for "
                                          "orders, refunds or payments; to tech for app problems.")
billing = Agent(llm, tools=[order_status], name="billing", system="You handle orders and refunds. Be brief.")
tech = Agent(llm, name="tech", system="You fix app problems. Be brief.")
support = Handoff([triage, billing, tech])

calculator = Agent(llm, tools=[multiply], name="calculator", system="Exact arithmetic with tools.")
boss = Supervisor(llm, [calculator, support], system="Answer briefly.")

if __name__ == "__main__":
    answer = support.run("Where is the refund for order A7?")
    print(answer.output, "| path:", " -> ".join(answer.path))

    result = boss.run("A customer bought 3 items at 249.5 each. What is the total?")
    print(result.output, "| delegated to:", result.tools_used)
