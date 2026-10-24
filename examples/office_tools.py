"""Tools (and a few starter agents) a developer writes once. Business users do the rest on the web page.

    export OLLAMA_API_KEY=...        # or OPENAI_API_KEY; TASKPILOT_MODEL picks the model
    taskpilot serve examples/office_tools.py        # web page at http://127.0.0.1:8000
    taskpilot ask examples/office_tools.py "Sales 125000 aur 75000, returns 40000. Net sales boss@company.com ko email karo"

On the page people can see the tools and agents, make new agents, choose a
workflow type (or let the AI suggest one), run, approve and save workflows.
Replace the tool bodies with your real systems (database, email, CRM...).
"""

from taskpilot import tool

ORDERS = {"A7": {"customer": "Ravi", "email": "ravi@example.com", "item": "tea set", "amount": 2499, "status": "delivered"},
          "B2": {"customer": "Asha", "email": "asha@example.com", "item": "phone case", "amount": 299, "status": "in transit"}}


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def get_order(order_id: str) -> dict:
    """Look up an order (customer, email, item, amount, status) by its id, like A7."""
    return ORDERS.get(order_id.upper(), {"error": f"no order {order_id}"})


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email to {to} sent: {subject}"


tools = [add_numbers, subtract_numbers, get_order, send_email]
approve = ["send_email"]          # always asks a person first: in plans AND when an agent uses it

agents = [
    {"name": "support", "tools": ["get_order", "send_email"],
     "instructions": "You are a polite support agent. Look up orders before answering. Keep emails short and warm."},
    {"name": "reviewer", "tools": [],
     "instructions": "You check customer emails for politeness, correct facts and length under 80 words."},
]
instructions = "Amounts are in rupees. Keep emails short and polite."
