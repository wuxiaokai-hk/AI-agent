# agent-lab: learn AI agents by building them

Seven small, runnable lessons. You'll start with a hand-written agent loop and finish
with multi-agent systems, MCP and evals. Everything runs on a synthetic panel of 20 credit
unions (2015Q1 to 2025Q4) with a **known data-generating process**, so you can check
whether an agent's numbers are right, the same way you'd backtest a forecast.

There's no LangChain or other framework here. The whole runtime (`agentlab/core.py`) is
about 120 lines you can read in ten minutes. Once you know what frameworks do underneath,
picking one up later is easy.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/make_data.py                    # creates data/credit_unions.csv
python lessons/07_evals.py --offline        # sanity check, no model needed
```

Then pick **one** model backend.

### Option A: free, no API key (local model with Ollama)

1. Install Ollama from ollama.com (Mac app) and open it.
2. Download a model that supports tool calling:
   ```bash
   ollama pull qwen3:8b        # ~5 GB; fine on a 16 GB Mac
   # 8 GB Mac: qwen3:4b    |  32 GB+ Mac: qwen3:14b or larger (noticeably smarter)
   ```
3. Run any lesson. With no `ANTHROPIC_API_KEY` set, agent-lab uses the local model automatically.
   To use a different model: `export AGENTLAB_MODEL=qwen3:4b`

Everything runs on your Mac: free, private, and works offline once the model is downloaded.

**What to expect from a small local model:** it's slower, it makes more mistakes (wrong
column names, malformed JSON, missed steps), and lessons 4 and 5 are the hardest for it.
That's useful to see. The errors show why lesson 2's error handling and lesson 7's evals
matter, and why bigger models are worth paying for in production.

### Option B: Claude via the Anthropic API (paid, stronger)

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...        # from console.anthropic.com
```
A full pass through all lessons uses about 0.5M tokens. Set `AGENTLAB_MODEL` to change the model.

### Other options

- **Any OpenAI-compatible server** (LM Studio, or a cloud service's free tier):
  `export AGENTLAB_BACKEND=local AGENTLAB_BASE_URL=<their /v1 URL> AGENTLAB_API_KEY=<key> AGENTLAB_MODEL=<model>`
- **Lesson 6 with your normal Claude app:** add the MCP server to Claude Desktop (see the
  top of `06_mcp_server.py`). Claude can then call your tools with no API key.

`agentlab/llm.py` translates between the Anthropic and OpenAI message formats. Read it
after lesson 1: tool calling works the same way across vendors, and only the JSON field
names differ.

## Curriculum

| # | File | Concept | What to notice |
|---|------|---------|----------------|
| 01 | `01_agent_loop_from_scratch.py` (+ `agentlab/llm.py`) | **The agent loop.** Model asks for a tool, your code runs it, the result goes back, repeat. | The `messages` list is the agent's entire memory. The model never runs code. |
| 02 | `02_tools_and_analyst_agent.py` | **Tool design** and **human-in-the-loop** | Errors get returned to the model, which fixes itself. `save_report` asks for your approval. |
| 03 | `03_memory_and_context.py` | **Short-term vs long-term memory**, context **compaction** | Run it twice: preferences persist. Turn 4 answers from a summary. |
| 04 | `04_planning_and_reflection.py` | **Workflows**: plan, execute, critique, revise; **LLM-as-judge** | Structure in code vs freedom for the model. Bounded retries. |
| 05 | `05_multi_agent.py` | **Orchestrator-workers**, context isolation, least privilege | Workers can't see the boss's context, so the briefs must be self-contained. Compare token cost with lesson 02. |
| 06 | `06_mcp_server.py` + `06b_mcp_client_agent.py` | **MCP**: package tools once, use them from any client | Plug the server into Claude Desktop/Claude Code, then into your own agent. |
| 07 | `07_evals.py` | **Evals**: tool tests, outcome accuracy, trajectory, consistency (pass^k) | Change a prompt, rerun, compare scores. Treat it like a backtest. |

Each file ends with **exercises**. Doing them is where most of the learning happens.

## Mental model

```
           ┌─────────────── your code (the "harness") ───────────────┐
 user ───▶ │ messages ──▶ LLM ──▶ tool_use? ──yes──▶ run tool ──┐    │
           │    ▲                    │no                         │    │
           │    └──── tool_result ◀──┼───────────────────────────┘    │
           │                         ▼                                │
           │                    final answer                          │
           └──────────────────────────────────────────────────────────┘
 Everything else (memory, planning, sub-agents, MCP, guardrails) is a variation on
 what goes INTO messages, which TOOLS exist, and what CODE wraps the loop.
```

## Glossary

- **Agent**: an LLM in a loop, choosing tools, until a task is done. A **workflow** is when the steps are fixed in code instead.
- **Tool / function calling**: the model emits structured JSON naming a function and its arguments, and your code executes it.
- **Context window**: everything the model sees on one call. It's finite and costs money, so manage it.
- **Compaction**: summarising old conversation turns to free up context.
- **RAG**: retrieval-augmented generation, i.e. a tool that fetches relevant documents into context.
- **MCP**: Model Context Protocol, a standard interface between AI apps and tools/data.
- **Guardrails**: approval gates, input validation, least-privilege tool sets, `max_steps`.
- **Eval**: a fixed test set plus a metric. pass^k means the agent passes on every one of k tries, which measures reliability, not just ability.

## Next steps (ideas that connect to your work)

1. **Point it at real data.** Swap the CSV for a public series (e.g. a StatCan unemployment
   table) and give the agent a `fetch_series` tool.
2. **Wrap your own models as tools.** For example, an existing forecasting or credit-loss
   script becomes `run_forecast(scenario)`. The agent handles the questions, and your code
   handles the maths.
3. **Agentic coding.** Use Claude Code on this repo: "add a plot_series tool with a test
   and an eval case." Watch how a production coding agent uses the same loop.
4. **Try a framework** (Claude Agent SDK, LangGraph, OpenAI Agents SDK) and map each of
   its concepts back to these lessons.

## Further reading

- Anthropic, *Building effective agents* (workflows vs agents, the core patterns)
- Anthropic docs: Tool use; Model Context Protocol (modelcontextprotocol.io)
- Anthropic, *Writing effective tools for agents*
