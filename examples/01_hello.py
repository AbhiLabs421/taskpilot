"""The basics: steps are functions, parameters name what they need.

    python examples/01_hello.py
    taskpilot run examples/01_hello.py raw="  asha " --verbose
"""

from taskpilot import Flow

flow = Flow("hello")


@flow.step
def name(raw):                      # 'raw' is not a step, so it is an input
    return raw.strip().title()


@flow.step
def greeting(name):                 # receives the result of step 'name'
    return f"Namaste, {name}!"


@flow.step
async def shout(greeting):          # async steps work the same way
    return greeting.upper()


if __name__ == "__main__":
    result = flow.run(raw="  asha ")
    print(result.output)            # NAMASTE, ASHA!
    print(result["name"])           # Asha
    print(result.summary())
    print(flow.describe())
