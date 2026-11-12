"""Pick the model backend. Lessons call `get_client()` and never care which one they got.

  AGENTLAB_BACKEND=anthropic -> Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
  AGENTLAB_BACKEND=local     -> any OpenAI-compatible server. Default: Ollama on your
                                own computer, free, no key, no internet after setup.
                                Also works with LM Studio, or free-tier cloud services
                                that expose an OpenAI-compatible endpoint.
If AGENTLAB_BACKEND is unset: "anthropic" when ANTHROPIC_API_KEY exists, else "local".

The LocalClient below is also a lesson in itself: it translates Anthropic-style
messages to the OpenAI "chat completions" format and back. Every agent framework
contains an adapter like this - the tool-calling idea is identical across vendors,
only the JSON field names differ.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from types import SimpleNamespace as NS

BACKEND = os.getenv("AGENTLAB_BACKEND") or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "local")
BASE_URL = os.getenv("AGENTLAB_BASE_URL", "http://localhost:11434/v1")  # Ollama default
MODEL = os.getenv("AGENTLAB_MODEL") or ("claude-sonnet-5-5" if BACKEND == "anthropic" else "qwen3:8b")


def get_client():
    if BACKEND == "anthropic":
        from anthropic import Anthropic
        return Anthropic()
    return LocalClient(BASE_URL, os.getenv("AGENTLAB_API_KEY", "not-needed"))


class LocalClient:
    """Mimics `anthropic.Anthropic().messages.create(...)` on top of an OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self.messages = NS(create=self._create)

    # ---------------------------------------------------------- Anthropic -> OpenAI
    @staticmethod
    def _to_openai(system, messages):
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                out.append({"role": m["role"], "content": c})
            elif m["role"] == "user":          # a list of tool_result dicts
                for r in c:
                    if r.get("type") == "tool_result":
                        prefix = "ERROR: " if r.get("is_error") else ""
                        out.append({"role": "tool", "tool_call_id": r["tool_use_id"],
                                    "content": prefix + str(r["content"])})
                    elif r.get("type") == "text":
                        out.append({"role": "user", "content": r["text"]})
            else:                              # assistant: text + tool_use blocks
                text = "".join(b.text for b in c if b.type == "text")
                calls = [{"id": b.id, "type": "function",
                          "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                         for b in c if b.type == "tool_use"]
                msg = {"role": "assistant", "content": text}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
        return out

    # ---------------------------------------------------------- OpenAI -> Anthropic
    def _create(self, model, messages, max_tokens=1024, system=None, tools=None, **_):
        body = {"model": model, "max_tokens": max_tokens,
                "messages": self._to_openai(system, messages)}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools]
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Can't reach a local model at {self.base_url} ({e}).\n"
                f"Is Ollama running? Try:  ollama pull {model}   then re-run.") from e

        msg = data["choices"][0]["message"]
        text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
        blocks = [NS(type="text", text=text)]
        for tc in msg.get("tool_calls") or []:
            args = tc["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": args}  # small models sometimes emit bad JSON; tool will error
            blocks.append(NS(type="tool_use", id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                             name=tc["function"]["name"], input=args))
        usage = data.get("usage") or {}
        return NS(content=blocks,
                  stop_reason="tool_use" if len(blocks) > 1 else "end_turn",
                  usage=NS(input_tokens=usage.get("prompt_tokens", 0),
                           output_tokens=usage.get("completion_tokens", 0)))
