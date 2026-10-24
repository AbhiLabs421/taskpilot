"""LLM layer: tools, agents, JSON repair, and real HTTP clients against a local fake server."""

import asyncio
import enum
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Literal, Optional

from taskpilot import Agent, FakeLLM, LLMError, Ollama, OpenAICompatible, Tool, ToolCall, tool
from taskpilot.llm import check_schema, extract_json


# --------------------------------------------------------------- fake server
class FakeServer:
    """A tiny HTTP server that imitates Ollama and OpenAI endpoints."""

    def __init__(self):
        self.requests = []
        self.fail_next = []  # status codes to return before succeeding
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body, ctype="application/json", headers=None):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append(("GET", self.path, None, dict(self.headers)))
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": "gpt-oss:20b"}, {"name": "gpt-oss:120b"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(("POST", self.path, body, dict(self.headers)))
                if outer.fail_next:
                    code = outer.fail_next.pop(0)
                    self._send(code, {"error": f"status {code}"}, headers={"Retry-After": "0"})
                    return
                if self.headers.get("Authorization") == "Bearer bad":
                    self._send(401, {"error": "unauthorized"})
                    return
                if self.path == "/api/chat":
                    self.ollama_chat(body)
                elif self.path == "/api/embed":
                    self._send(200, {"embeddings": [[0.1, 0.2] for _ in body["input"]]})
                elif self.path == "/v1/chat/completions" or (
                        self.path.startswith("/openai/deployments/") and "/chat/completions?api-version=" in self.path):
                    self.openai_chat(body)
                else:
                    self._send(404, {"error": "no route"})

            def ollama_chat(self, body):
                last = body["messages"][-1]
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for piece in ["Hel", "lo", "!"]:
                        self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
                    self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())
                    return
                if body.get("tools") and last["role"] == "user":
                    msg = {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "function": {"name": "multiply", "arguments": {"a": 23, "b": 19}}}]}
                elif last["role"] == "tool":
                    msg = {"role": "assistant", "content": f"The answer is {last['content']}"}
                elif body.get("format"):
                    msg = {"role": "assistant", "content": '{"city": "Delhi", "population": 32}'}
                else:
                    msg = {"role": "assistant", "content": "echo:" + last["content"], "thinking": "hmm"}
                self._send(200, {"model": body["model"], "message": msg, "done": True, "prompt_eval_count": 5, "eval_count": 7})

            def openai_chat(self, body):
                last = body["messages"][-1]
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for piece in ["a", "b"]:
                        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                if body.get("tools") and last["role"] == "user":
                    msg = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "t1", "type": "function", "function": {"name": "multiply", "arguments": '{"a": 6, "b": 7}'}}]}
                elif last["role"] == "tool":
                    msg = {"role": "assistant", "content": f"got {last['content']} for {last['tool_call_id']}"}
                else:
                    msg = {"role": "assistant", "content": "hi " + last["content"]}
                self._send(200, {"model": body["model"], "choices": [{"message": msg}], "usage": {"prompt_tokens": 3, "completion_tokens": 4}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: first number.
        b: second number.
    """
    return a * b


class TestOllamaClient(unittest.TestCase):
    def test_chat_ask_and_usage(self):
        with FakeServer() as srv:
            llm = Ollama("gpt-oss:20b", host=srv.url, api_key="k")
            resp = llm.chat("hi", system="be nice")
            self.assertEqual(resp.content, "echo:hi")
            self.assertEqual(resp.thinking, "hmm")
            self.assertEqual(resp.usage, {"prompt_tokens": 5, "completion_tokens": 7})
            method, path, body, headers = srv.requests[-1]
            self.assertEqual(body["messages"][0], {"role": "system", "content": "be nice"})
            self.assertEqual(headers["Authorization"], "Bearer k")
            self.assertEqual(llm.ask("x"), "echo:x")

    def test_tool_agent_over_http(self):
        with FakeServer() as srv:
            agent = Agent(Ollama("gpt-oss:120b", host=srv.url), tools=[multiply])
            result = agent.run("What is 23*19?")
            self.assertEqual(result.output, "The answer is 437")
            self.assertEqual(result.tools_used, ["multiply"])
            tool_msg = srv.requests[-1][2]["messages"][-1]
            self.assertEqual(tool_msg, {"role": "tool", "content": "437", "tool_name": "multiply"})

    def test_stream(self):
        with FakeServer() as srv:
            self.assertEqual(list(Ollama(host=srv.url).stream("x")), ["Hel", "lo", "!"])

    def test_json_structured_output(self):
        with FakeServer() as srv:
            schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            out = Ollama(host=srv.url).json("Biggest city?", schema=schema)
            self.assertEqual(out["city"], "Delhi")
            self.assertEqual(srv.requests[-1][2]["format"], schema)

    def test_retries_on_429_and_500(self):
        with FakeServer() as srv:
            srv.fail_next = [429, 503]
            self.assertEqual(Ollama(host=srv.url, retries=3).ask("x"), "echo:x")
            self.assertEqual(len(srv.requests), 3)

    def test_gives_up_with_clear_error(self):
        with FakeServer() as srv:
            srv.fail_next = [500, 500, 500]
            with self.assertRaises(LLMError) as cm:
                Ollama(host=srv.url, retries=1).ask("x")
            self.assertEqual(cm.exception.status, 500)

    def test_bad_key_message(self):
        with FakeServer() as srv:
            with self.assertRaises(LLMError) as cm:
                Ollama(host=srv.url, api_key="bad").ask("x")
            self.assertIn("API key", str(cm.exception))
            self.assertEqual(cm.exception.status, 401)

    def test_unreachable_host(self):
        with self.assertRaises(LLMError) as cm:
            Ollama(host="http://127.0.0.1:1", retries=0, timeout=2).ask("x")
        self.assertIn("Could not reach", str(cm.exception))

    def test_models_and_embed(self):
        with FakeServer() as srv:
            llm = Ollama(host=srv.url)
            self.assertIn("gpt-oss:120b", llm.models())
            self.assertEqual(len(llm.embed(["a", "b"])), 2)

    def test_host_defaults(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Ollama().host, "http://localhost:11434")
            self.assertEqual(Ollama(api_key="x").host, "https://ollama.com")
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "0.0.0.0:11434"}, clear=True):
            self.assertEqual(Ollama().host, "http://127.0.0.1:11434")

    def test_async_chat(self):
        with FakeServer() as srv:
            llm = Ollama(host=srv.url)

            async def main():
                return await asyncio.gather(*(llm.aask(f"q{i}") for i in range(5)))

            self.assertEqual(asyncio.run(main()), [f"echo:q{i}" for i in range(5)])


class TestOpenAICompatible(unittest.TestCase):
    def test_chat_tools_stream(self):
        with FakeServer() as srv:
            llm = OpenAICompatible("m", base_url=srv.url + "/v1", api_key="k")
            self.assertEqual(llm.ask("there"), "hi there")
            result = Agent(llm, tools=[multiply]).run("6*7?")
            self.assertEqual(result.output, "got 42 for t1")
            self.assertEqual(list(llm.stream("x")), ["a", "b"])


class TestOpenAIConfig(unittest.TestCase):
    def test_env_vars(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "http://x/v1/", "OPENAI_API_KEY": "sk-env"}, clear=True):
            llm = OpenAICompatible("m")
        self.assertEqual(llm.base_url, "http://x/v1")
        self.assertEqual(llm._http.headers["Authorization"], "Bearer sk-env")

    def test_azure_url_header_and_version(self):
        with FakeServer() as srv:
            llm = OpenAICompatible.azure("my-gpt", endpoint=srv.url + "/", api_key="az-key", api_version="2024-10-21")
            self.assertEqual(llm.ask("there"), "hi there")
            method, path, body, headers = srv.requests[-1]
            headers = {k.lower(): v for k, v in headers.items()}  # HTTP header names are case-insensitive
            self.assertEqual(path, "/openai/deployments/my-gpt/chat/completions?api-version=2024-10-21")
            self.assertEqual(headers["api-key"], "az-key")
            self.assertNotIn("authorization", headers)
            self.assertEqual(list(llm.stream("x")), ["a", "b"])

    def test_azure_needs_endpoint(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMError):
                OpenAICompatible.azure("d")

    def test_custom_header_and_query(self):
        with FakeServer() as srv:
            llm = OpenAICompatible("m", base_url=srv.url + "/v1", api_key="k", api_key_header="X-Key",
                                   headers={"OpenAI-Organization": "org1"})
            llm.ask("x")
            headers = {k.lower(): v for k, v in srv.requests[-1][3].items()}
            self.assertEqual(headers["x-key"], "k")
            self.assertEqual(headers["openai-organization"], "org1")


class TestTools(unittest.TestCase):
    def test_schema_from_hints_and_docstring(self):
        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        @tool
        def paint(wall: str, color: Color, coats: int = 2, mode: Literal["fast", "slow"] = "fast",
                  tags: Optional[List[str]] = None) -> str:
            """Paint a wall.

            Args:
                wall: which wall.
                color: paint color.
            Returns:
                status text.
            """
            return "done"

        spec = paint.spec()["function"]
        self.assertEqual(spec["name"], "paint")
        self.assertEqual(spec["description"], "Paint a wall.")
        props = spec["parameters"]["properties"]
        self.assertEqual(props["wall"], {"type": "string", "description": "which wall."})
        self.assertEqual(props["color"]["enum"], ["red", "blue"])
        self.assertEqual(props["coats"], {"type": "integer", "default": 2})
        self.assertEqual(props["mode"]["enum"], ["fast", "slow"])
        self.assertEqual(props["tags"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(spec["parameters"]["required"], ["wall", "color"])

    def test_invoke_coerces_and_reports_errors(self):
        self.assertEqual(multiply.invoke({"a": "6", "b": 7.0}), "42")
        self.assertIn("missing required", multiply.invoke({"a": 1}))
        self.assertIn("Error", Tool(lambda: 1 / 0, name="boom").invoke({}))

    def test_text_arguments_accept_numbers_and_objects(self):
        @tool
        def note(text: str) -> str:
            "Echo."
            return "got " + text

        self.assertEqual(note.invoke({"text": 5}), "got 5")
        self.assertEqual(note.invoke({"text": 160.0}), "got 160")
        self.assertEqual(note.invoke({"text": {"a": 1}}), 'got {"a": 1}')

    def test_async_tool(self):
        @tool
        async def fetch(url: str) -> dict:
            "Fetch a URL."
            await asyncio.sleep(0)
            return {"url": url}

        self.assertEqual(json.loads(fetch.invoke({"url": "x"})), {"url": "x"})

    def test_bad_tool_name(self):
        from taskpilot import ToolError

        with self.assertRaises(ToolError):
            Tool(lambda: 1, name="has space")


class TestAgent(unittest.TestCase):
    def test_multi_step_tool_loop(self):
        llm = FakeLLM([
            {"tool": "multiply", "args": {"a": 2, "b": 3}},
            {"tool": "multiply", "args": {"a": 6, "b": 7}},
            "Final: 42",
        ])
        result = Agent(llm, tools=[multiply]).run("go")
        self.assertEqual(result.output, "Final: 42")
        self.assertEqual([s.output for s in result.steps], ["6", "42"])
        self.assertEqual(result.turns, 3)
        self.assertFalse(result.stopped)

    def test_empty_reply_gets_one_nudge(self):
        llm = FakeLLM(["", "real answer"])
        self.assertEqual(Agent(llm).run("x").output, "real answer")
        self.assertEqual(llm.calls[1]["messages"][-1]["content"], "Please give your answer.")

    def test_parallel_tool_calls(self):
        @tool
        def slow(n: int) -> int:
            "Sleep then return n."
            time.sleep(0.2)
            return n

        llm = FakeLLM([{"tool_calls": [{"name": "slow", "arguments": {"n": i}} for i in range(5)]}, "done"])
        t = time.perf_counter()
        result = Agent(llm, tools=[slow]).run("x")
        self.assertLess(time.perf_counter() - t, 0.6)
        self.assertEqual([s.output for s in result.steps], ["0", "1", "2", "3", "4"])

    def test_unknown_tool_and_tool_crash_are_fed_back(self):
        def boom() -> str:
            "Explodes."
            raise RuntimeError("kaboom")

        llm = FakeLLM([{"tool": "nope"}, {"tool": "boom"}, "recovered"])
        result = Agent(llm, tools=[boom]).run("x")
        self.assertEqual(result.output, "recovered")
        self.assertIn("no tool named 'nope'", result.steps[0].output)
        self.assertIn("kaboom", result.steps[1].output)

    def test_max_steps_forces_final_answer(self):
        llm = FakeLLM(fn=lambda msgs, tools: {"tool": "multiply", "args": {"a": 1, "b": 1}} if tools else "forced answer")
        result = Agent(llm, tools=[multiply], max_steps=3).run("loop forever")
        self.assertEqual(result.output, "forced answer")
        self.assertTrue(result.stopped)
        self.assertEqual(len(result.steps), 3)

    def test_memory_and_trim(self):
        llm = FakeLLM(fn=lambda msgs, tools: f"seen {len(msgs)}")
        agent = Agent(llm, memory=True, max_history=4, system="sys")
        agent("one")
        agent("two")
        self.assertEqual(agent("three"), "seen 6")  # sys + 4 history + new
        self.assertEqual(agent.history[0]["role"], "user")
        agent.reset()
        self.assertEqual(agent("again"), "seen 2")

    def test_agent_as_tool(self):
        researcher = Agent(FakeLLM(["research notes"]), name="researcher")
        boss = Agent(FakeLLM([{"tool": "researcher", "args": {"task": "find"}}, "report based on notes"]),
                     tools=[researcher.as_tool()])
        result = boss.run("write report")
        self.assertEqual(result.steps[0].output, "research notes")

    def test_async_agent(self):
        llm = FakeLLM([{"tool": "multiply", "args": {"a": 3, "b": 3}}, "nine"])
        result = asyncio.run(Agent(llm, tools=[multiply]).arun("x"))
        self.assertEqual(result.output, "nine")
        self.assertEqual(result.steps[0].output, "9")


class TestJSON(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('Sure! ```json\n[1, 2]\n``` hope it helps'), [1, 2])
        self.assertEqual(extract_json('The result is {"ok": true} as asked'), {"ok": True})
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_check_schema(self):
        schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}}
        self.assertEqual(check_schema({"n": 1, "tags": ["a"]}, schema), [])
        self.assertTrue(check_schema({"n": "1"}, schema))
        self.assertTrue(check_schema({"tags": [1]}, schema))
        self.assertTrue(check_schema({"n": True}, schema), "bool is not an integer")

    def test_json_self_repair(self):
        llm = FakeLLM(["not json at all", '{"n": "wrong type"}', '{"n": 5}'])
        out = llm.json("give n", schema={"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}})
        self.assertEqual(out, {"n": 5})
        self.assertEqual(len(llm.calls), 3)
        self.assertIn("not valid", llm.calls[1]["messages"][-1]["content"])

    def test_json_gives_up(self):
        with self.assertRaises(LLMError):
            FakeLLM(["nope"]).json("x", repair=1)


class TestToolCallRepr(unittest.TestCase):
    def test_repr(self):
        self.assertIn("multiply", repr(ToolCall("multiply", {"a": 1})))


if __name__ == "__main__":
    unittest.main()
