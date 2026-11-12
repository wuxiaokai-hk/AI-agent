"""A minimal, readable agent runtime (~120 lines).

This is what you build by hand in lesson 01, cleaned up so lessons 02-07 can reuse it.
No framework magic: an agent is just
    while model asks for a tool: run tool, send result back
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import MODEL, get_client

GREY, BLUE, GREEN, RED, RESET = "\033[90m", "\033[94m", "\033[92m", "\033[91m", "\033[0m"


@dataclass
class Tool:
    name: str
    description: str          # the model reads this: write it like docs for a colleague
    input_schema: dict        # JSON Schema for the arguments
    fn: Callable[..., Any]
    needs_approval: bool = False  # human-in-the-loop for risky actions

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def tool(name: str, description: str, properties: dict, required: list[str] | None = None,
         needs_approval: bool = False):
    """Decorator: turn a plain Python function into a Tool."""
    def wrap(fn):
        schema = {"type": "object", "properties": properties, "required": required or list(properties)}
        return Tool(name, description, schema, fn, needs_approval)
    return wrap


@dataclass
class Agent:
    system: str
    tools: list[Tool] = field(default_factory=list)
    model: str = MODEL
    max_steps: int = 12          # hard stop so a confused agent can't loop forever
    max_tokens: int = 4096
    verbose: bool = True
    name: str = "agent"
    # Filled during runs: useful for evals (lesson 07) and cost tracking
    trajectory: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    def __post_init__(self):
        self.client = get_client()
        self._tools = {t.name: t for t in self.tools}

    def log(self, color: str, msg: str):
        if self.verbose:
            print(f"{color}[{self.name}] {msg}{RESET}")

    def call_model(self, messages: list[dict]):
        kwargs = {"tools": [t.spec() for t in self.tools]} if self.tools else {}
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=self.system,
            messages=messages, **kwargs,
        )
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        return resp

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        """Run one tool. Returns (result_text, is_error). Errors go BACK to the model,
        which can usually fix its own mistake (wrong column name, etc.)."""
        t = self._tools.get(name)
        if t is None:
            return f"Unknown tool {name!r}. Available: {list(self._tools)}", True
        if t.needs_approval:
            ok = input(f"{RED}Approve {name}({json.dumps(args)})? [y/N] {RESET}").strip().lower()
            if ok != "y":
                return "The human declined this action. Do not retry it; explain what you wanted to do.", True
        try:
            out = t.fn(**args)
            return (out if isinstance(out, str) else json.dumps(out, default=str)), False
        except Exception as e:  # noqa: BLE001 - we want every failure surfaced to the model
            return f"{type(e).__name__}: {e}", True

    def run(self, user_msg: str, messages: list[dict] | None = None) -> str:
        """Run the agent loop. Pass `messages` to continue an existing conversation."""
        messages = messages if messages is not None else []
        messages.append({"role": "user", "content": user_msg})
        self.log(BLUE, f"user: {user_msg[:200]}")

        for step in range(self.max_steps):
            resp = self.call_model(messages)
            messages.append({"role": "assistant", "content": resp.content})

            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    self.log(GREY, f"thinks/says: {block.text.strip()[:300]}")

            if resp.stop_reason != "tool_use":
                final = "".join(b.text for b in resp.content if b.type == "text")
                self.log(GREEN, f"done in {step + 1} model calls | tokens {self.usage}")
                self.messages = messages
                return final

            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                self.log(BLUE, f"-> {block.name}({json.dumps(block.input)[:200]})")
                out, is_err = self.execute(block.name, block.input)
                self.log(RED if is_err else GREY, f"<- {out[:300]}")
                self.trajectory.append({"tool": block.name, "args": block.input, "error": is_err})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": out, "is_error": is_err})
            messages.append({"role": "user", "content": results})

        self.messages = messages
        return "[stopped: hit max_steps]"
