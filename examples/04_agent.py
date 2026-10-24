"""A tool-using agent, then a two-agent team.

    export OLLAMA_API_KEY=...
    python examples/04_agent.py
"""

from taskpilot import Agent, Ollama, tool

llm = Ollama("gpt-oss:120b", options={"temperature": 0})


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers exactly."""
    return a * b


@tool
def exchange_rate(currency: str) -> float:
    """Rupees for one unit of a currency.

    Args:
        currency: ISO code like USD or EUR.
    """
    return {"USD": 88.2, "EUR": 102.5}.get(currency.upper(), 0.0)


calculator = Agent(llm, tools=[multiply, exchange_rate], name="calculator",
                   system="Use tools for every number. Be brief.")

if __name__ == "__main__":
    result = calculator.run("How many rupees is 249.99 USD?")
    print(result.output)
    print("tools used:", result.tools_used)

    writer = Agent(
        llm,
        tools=[calculator.as_tool(description="Exact currency conversion and math. Never guess rates; ask this.")],
        system="You write friendly one-line answers. Always ask the calculator for any number or exchange rate.",
    )
    team = writer.run("I have 120 EUR. In rupees, how much is that?")
    print(team.output)
    print("delegated to:", team.tools_used)
