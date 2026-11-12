"""LESSON 03 - Memory: short-term (context) vs long-term (stored notes).

The model is stateless. "Memory" is whatever you put back into the prompt.
Two kinds:

  SHORT-TERM = the messages list. Grows every turn; eventually too long/expensive.
               Fix: *compaction* - summarise old turns, keep recent ones verbatim.
  LONG-TERM  = an external store the agent reads/writes through tools
               (a JSON file here; a database or vector store in production).

Run:  python lessons/03_memory_and_context.py
It runs a short multi-turn chat, compacts it, and shows notes persisting to disk.
Run it twice: on the second run the agent already "knows" your preferences.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentlab.core import Agent, tool  # noqa: E402
from agentlab.data_tools import READ_ONLY  # noqa: E402

MEMORY_FILE = Path(__file__).resolve().parent.parent / "outputs" / "memory.json"


def _load() -> list[dict]:
    return json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else []


@tool("remember", "Save a durable fact or user preference for future sessions "
      "(e.g. 'user prefers basis points'). Don't save one-off details.",
      {"fact": {"type": "string"}})
def remember(fact):
    notes = _load()
    notes.append({"fact": fact})
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(notes, indent=2))
    return "saved"


@tool("recall", "Search saved notes by keyword. Empty keyword returns everything.",
      {"keyword": {"type": "string"}}, required=[])
def recall(keyword=""):
    hits = [n["fact"] for n in _load() if keyword.lower() in n["fact"].lower()]
    return hits or "no notes"


def _as_text(content) -> str:
    """Flatten a message's content (string or list of blocks) into plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, dict):                      # our tool_result dicts
            parts.append(f"[tool result] {b.get('content', '')}"[:500])
        elif b.type == "text":
            parts.append(b.text)
        elif b.type == "tool_use":
            parts.append(f"[called {b.name}({json.dumps(b.input)})]")
    return " ".join(parts)


def compact(agent: Agent, messages: list[dict], keep_last: int = 2) -> list[dict]:
    """Replace all but the last `keep_last` exchanges with a model-written summary.

    Must cut at a clean boundary: a user turn that is plain text (not tool_results),
    otherwise a tool_result would be orphaned from its tool_use and the API errors.
    """
    starts = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str)]
    if len(starts) <= keep_last:
        return messages
    cut = starts[-keep_last]
    old, recent = messages[:cut], messages[cut:]
    transcript = "\n".join(f"{m['role']}: {_as_text(m['content'])}" for m in old)
    summary = agent.client.messages.create(
        model=agent.model, max_tokens=600,
        messages=[{"role": "user", "content":
                   "Summarise this analyst session in <=150 words. Keep every number, "
                   "decision and open question.\n\n" + transcript}],
    ).content[0].text
    print(f"\n[compaction] {len(old)} messages -> 1 summary ({len(summary.split())} words)\n")
    return [{"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"},
            {"role": "assistant", "content": "Understood, continuing from that summary."}] + recent


if __name__ == "__main__":
    agent = Agent(
        system="You are a credit-risk analyst. At the start of a session call recall('') to load "
               "saved notes and follow any preferences in them. Use `remember` when the user states "
               "a lasting preference.",
        tools=READ_ONLY + [remember, recall], name="analyst",
    )
    msgs: list[dict] = []
    turns = [
        "Hi. FYI I always want rate effects in basis points and I report to the Ontario credit-union team.",
        "What was the average delinquency rate across all credit unions in 2020 vs 2019?",
        "Which credit union had the highest average delinquency in 2023?",
        "Remind me: what were the two numbers from my first data question?",
    ]
    for i, t in enumerate(turns):
        print("\n>>>", agent.run(t, msgs)[:800])
        if i == 2:
            msgs[:] = compact(agent, msgs)  # turn 4 must answer from the summary
    print("\nLong-term notes on disk:", _load())

# EXERCISES
# 1. Print len(str(msgs)) after each turn to watch the context grow; then after compaction.
# 2. Replace keyword recall with embedding search (e.g. sentence-transformers + cosine).
# 3. What should NOT go in long-term memory? (one-off numbers, secrets). Encode that in the tool description.
