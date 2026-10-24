"""The smallest taskpilot workflow: add and subtract two numbers.

    python examples/calc.py
    taskpilot run examples/calc.py a=10 b=4
"""

from taskpilot import Flow

flow = Flow("calculator")


@flow.step
def add(a, b):
    return a + b


@flow.step
def subtract(a, b):
    return a - b


if __name__ == "__main__":
    result = flow.run(a=10, b=4)
    print(result.output)        # {'add': 14, 'subtract': 6}
