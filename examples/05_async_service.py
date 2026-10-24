"""Use a flow inside an async app (FastAPI, aiohttp, bots...) and stream progress.

    python examples/05_async_service.py
"""

import asyncio

from taskpilot import Flow

flow = Flow("enrich")


@flow.step
async def profile(user_id):
    await asyncio.sleep(0.1)                      # e.g. a database call
    return {"id": user_id, "name": f"user{user_id}"}


@flow.step
async def orders(user_id):                        # runs at the same time as profile
    await asyncio.sleep(0.1)
    return [user_id * 10, user_id * 20]


@flow.step
def card(profile, orders):
    return f"{profile['name']}: {len(orders)} orders, total {sum(orders)}"


async def main():
    # 100 requests handled concurrently - takes ~0.1s, not 20s
    results = await asyncio.gather(*(flow.arun(user_id=i) for i in range(100)))
    print(results[7].output)

    async for event in flow.astream(user_id=1):   # live progress, e.g. for a websocket
        if event.type == "step_end":
            print(f"done: {event.step} in {event.duration * 1000:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
