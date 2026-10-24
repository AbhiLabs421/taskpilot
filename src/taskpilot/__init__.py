"""taskpilot – fast, dependency-free workflows and AI agents for Python.

::

    from taskpilot import Flow

    flow = Flow("hello")

    @flow.step
    def name(raw):
        return raw.strip().title()

    @flow.step
    def greeting(name):
        return f"Hello, {name}!"

    print(flow.run(raw="  asha ").output)   # Hello, Asha!
"""

from .checkpoint import Checkpoint
from .context import Context
from .engine import Loop, LoopLimit, Plan, Skip
from .errors import (
    CheckFailed,
    VerificationError,
    CycleError,
    FlowDefinitionError,
    FlowTimeout,
    TaskPilotError,
    LLMError,
    MissingInputError,
    StepError,
    StepTimeout,
    ToolError,
)
from .events import Event
from .flow import Flow
from .human import Request, WaitingForInput
from .llm import LLM, Agent, ChatResponse, FakeLLM, FallbackLLM, Ollama, OpenAICompatible, Tool, ToolCall, tool
from . import auto, connect, evals, mcp, memory, patterns, tracing
from .auto import AutoFlow, AutoPlan, AutoResult, Disagreement, Unclear
from .workspace import Workspace
from .connect import A2AAgent, http_tool, openapi_tools
from .mcp import MCPClient, MCPServer
from .memory import ConversationMemory, MemoryStore
from .patterns import GroupChat, Handoff, MapReduce, Parallel, ReviewLoop, Router, Sequential, Supervisor, Vote
from .policies import Cache, Retry
from .result import RunResult, StepResult
from .step import Step

__version__ = "0.1.0"

__all__ = [
    "Flow", "Step", "Loop", "LoopLimit", "Context", "Retry", "Cache", "Checkpoint", "Skip", "Event", "Plan",
    "RunResult", "StepResult", "Request", "WaitingForInput",
    "LLM", "Ollama", "OpenAICompatible", "FakeLLM", "FallbackLLM", "ChatResponse", "Agent", "Tool", "ToolCall", "tool",
    "AutoFlow", "AutoPlan", "AutoResult", "Unclear", "Disagreement", "Workspace", "ConversationMemory", "MemoryStore",
    "MCPClient", "MCPServer", "A2AAgent", "http_tool", "openapi_tools",
    "Sequential", "Parallel", "Router", "MapReduce", "ReviewLoop", "Vote", "Supervisor", "Handoff", "GroupChat",
    "TaskPilotError", "CheckFailed", "VerificationError", "FlowDefinitionError", "CycleError", "MissingInputError", "StepError",
    "StepTimeout", "FlowTimeout", "LLMError", "ToolError",
    "auto", "connect", "evals", "mcp", "memory", "patterns", "tracing", "__version__",
]
