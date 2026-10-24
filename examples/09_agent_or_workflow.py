"""One job, three ways: write the steps yourself, let an agent decide, or mix both.

    export OPENAI_API_KEY=sk-...                 # OpenAI
    # or: export OLLAMA_API_KEY=...              # Ollama Cloud, through its OpenAI-compatible /v1 API
    python examples/09_agent_or_workflow.py
"""

import os

from taskpilot import Agent, Flow, OpenAICompatible, tool

# ---- 1. Pick a model. Every client has the same methods (ask, json, stream, chat).
if os.environ.get("OPENAI_API_KEY"):
    llm = OpenAICompatible("gpt-4o-mini")                      # key from $OPENAI_API_KEY
else:
    llm = OpenAICompatible("gpt-oss:120b", base_url="https://ollama.com/v1", api_key=os.environ["OLLAMA_API_KEY"])
# Other choices, same code after this line:
#   OpenAICompatible("llama-3.3-70b", base_url="https://api.groq.com/openai/v1", api_key="...")
#   OpenAICompatible.azure("my-deployment", endpoint="https://NAME.openai.azure.com", api_key="...")
#   OpenAICompatible("local-model", base_url="http://localhost:1234/v1")       # LM Studio / vLLM
#   Ollama("gpt-oss:20b")                                                      # native Ollama API

ORDERS = {"A7": {"item": "tea set", "amount": 2499, "status": "delivered"},
          "B2": {"item": "phone case", "amount": 299, "status": "in transit"}}


@tool
def order_status(order_id: str) -> dict:
    """Look up an order by its id (like A7)."""
    return ORDERS.get(order_id.upper(), {"error": "no such order"})


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order. Only for delivered orders."""
    return f"Refunded Rs {amount} for {order_id.upper()}"


MESSAGE = "Order A7 came broken. Please refund me."

# ---- 2a. WORKFLOW: you decide the steps. Predictable, testable, cheap.
flow = Flow("refund_desk", description="Check an order and refund it if it was delivered.")


@flow.step
def found(order_id: str):
    return order_status(order_id)


@flow.step(when=lambda found: found.get("status") == "delivered")
def refunded(found, order_id):
    return issue_refund(order_id, found["amount"])


@flow.step
def answer(found, refunded=None):
    return refunded or f"Cannot refund yet: order is {found.get('status', 'unknown')}."


# ---- 2b. AGENT: the model reads the message and decides which tools to call.
agent = Agent(llm, tools=[order_status, issue_refund], name="support",
              system="You are a support agent. Check the order before refunding. Be brief.")

# ---- 2c. MIX: the agent gets the whole workflow as one tool (safe, fixed rules inside),
#          and a workflow can contain an agent as one step.
mixed_agent = Agent(llm, tools=[flow.as_tool()], system="Use the refund_desk tool for refunds. Be brief.")

reply_flow = Flow("reply")
reply_flow.add(agent.as_step("reply", "Customer says: {message}"))
reply_flow.add(lambda reply: reply.strip()[:300], name="short")


if __name__ == "__main__":
    print("1) workflow :", flow.run(order_id="A7").output)

    result = agent.run(MESSAGE)
    print("2) agent    :", result.output, "| tools it chose:", result.tools_used)

    result = mixed_agent.run(MESSAGE)
    print("3) mixed    :", result.output, "| tools it chose:", result.tools_used)

    print("4) agent as a step in a flow:", reply_flow.run(message="Where is order B2?").output)
