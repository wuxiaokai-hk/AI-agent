"""LESSON 06b - Your own agent as an MCP client.

This agent has NO tools written in it. It launches the MCP server from 06, asks it
what tools exist, hands those schemas to Claude, and forwards tool calls over MCP.
Swap the server for any other MCP server (filesystem, GitHub, Postgres...) and the
same agent gains those abilities - that's the point of a protocol.

Run:  python lessons/06b_mcp_client_agent.py
      python lessons/06b_mcp_client_agent.py --list   # just list server tools, no LLM
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SERVER = Path(__file__).resolve().parent / "06_mcp_server.py"
sys.path.insert(0, str(SERVER.parent.parent))
from agentlab.llm import MODEL, get_client  # noqa: E402


def _get(obj, new, old):
    """mcp 2.x uses snake_case fields, 1.x used camelCase - support both."""
    return getattr(obj, new, None) if hasattr(obj, new) else getattr(obj, old)


def result_text(res) -> str:
    return "\n".join(getattr(c, "text", str(c)) for c in res.content)


async def main(question: str | None):
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
        tools = [{"name": t.name, "description": t.description or "", "input_schema": _get(t, "input_schema", "inputSchema")}
                 for t in listed.tools]
        print("Discovered over MCP:", [t["name"] for t in tools])

        if question is None:                      # --list mode: call one tool directly
            res = await session.call_tool("query_data", {"columns": ["delinquency_rate"],
                                                         "group_by": "cu_id", "agg": "mean"})
            print(result_text(res)[:500])
            return

        client = get_client()
        messages = [{"role": "user", "content": question}]
        for _ in range(10):  # the same loop as lesson 01 - only tool *execution* changed
            resp = client.messages.create(model=MODEL, max_tokens=2048, tools=tools, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    print(f"  MCP -> {b.name}({json.dumps(b.input)[:150]})")
                    res = await session.call_tool(b.name, b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": result_text(res), "is_error": bool(_get(res, "is_error", "isError"))})
            messages.append({"role": "user", "content": results})
        print("\n" + "".join(b.text for b in resp.content if b.type == "text"))


if __name__ == "__main__":
    q = None if "--list" in sys.argv else (
        "Which three credit unions had the highest average delinquency in 2024, and how does "
        "delinquency respond to unemployment once CU fixed effects are included?")
    asyncio.run(main(q))
