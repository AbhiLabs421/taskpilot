"""An AI pipeline on Ollama: outline -> sections written in parallel -> final edit.

    export OLLAMA_API_KEY=...          # Ollama Cloud, or run `ollama serve` locally
    python examples/03_ai_writer.py "why the sky is blue"
"""

import sys

from taskpilot import Flow, Ollama

llm = Ollama("gpt-oss:120b", options={"temperature": 0.4})
flow = Flow("writer", verbose=True)


@flow.step(retry=3)
def outline(topic):
    return llm.json(
        f"Make an outline of 3 short section titles for a blog post about: {topic}",
        schema={"type": "object", "properties": {"sections": {"type": "array", "items": {"type": "string"}}}, "required": ["sections"]},
    )["sections"]


@flow.step(map="outline", retry=3)              # every section is written at the same time
def sections(outline, topic):
    return llm.ask(f"Write one tight paragraph titled '{outline}' for a post about {topic}.")


flow.prompt("post", llm, "Join these into a polished post with a title. Keep it under 300 words:\n\n{sections}")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "why the sky is blue"
    print(flow.run(topic=topic).output)
