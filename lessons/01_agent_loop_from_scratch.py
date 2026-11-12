r"""LESSON 01 - The agent loop, from scratch, no framework.

Key idea: an LLM can't *do* anything. It can only emit text that says
"please call tool X with args Y". YOUR code runs the tool and sends the result back.
Repeat until the model stops asking. That loop IS the agent.

    user -> model -> tool_use? --yes--> run tool -> tool_result -> model -> ...
                          \--no---> final answer

Run:  python lessons/01_agent_loop_from_scratch.py
Then read the printed `messages` list: it is the agent's entire "mind".
"""
import ast
import json
import operator as op
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# get_client() returns Claude (if you have an API key) or a free local model (Ollama).
# Both expose the same .messages.create(...) call, so the loop below is unchanged.
from agentlab.llm import MODEL, get_client  # noqa: E402

client = get_client()

# ---------------------------------------------------------------- 1. tools as plain functions
POLICY_RATES = {"canada": 2.25, "us": 3.75, "euro area": 2.00, "uk": 3.75}  # illustrative values


def get_policy_rate(economy: str) -> str:
    rate = POLICY_RATES.get(economy.lower())
    if rate is None:
        return f"No data for {economy}. Known: {list(POLICY_RATES)}"
    return f"{rate}%"


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg}


def calculator(expression: str) -> str:
    """Safe arithmetic. Never eval() model output - treat it like user input."""
    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only arithmetic allowed")
    return str(ev(ast.parse(expression, mode="eval").body))


# ---------------------------------------------------------------- 2. describe tools to the model
# The model never sees your Python. It only sees name + description + JSON schema.
TOOLS = [
    {"name": "get_policy_rate",
     "description": "Get the current central bank policy rate (percent) for an economy.",
     "input_schema": {"type": "object",
                      "properties": {"economy": {"type": "string", "description": "e.g. 'canada'"}},
                      "required": ["economy"]}},
    {"name": "calculator",
     "description": "Evaluate an arithmetic expression like '3.75 - 2.25'.",
     "input_schema": {"type": "object",
                      "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
]
FUNCS = {"get_policy_rate": get_policy_rate, "calculator": calculator}


# ---------------------------------------------------------------- 3. the loop
def run(question: str, max_steps: int = 8) -> str:
    messages = [{"role": "user", "content": question}]
    for step in range(max_steps):
        resp = client.messages.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages,
                                      system="You are a concise macro analyst. Use tools for facts and math.")
        messages.append({"role": "assistant", "content": resp.content})
        print(f"\n--- step {step + 1}: stop_reason={resp.stop_reason}")

        if resp.stop_reason != "tool_use":       # model is done
            break

        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = FUNCS[block.name](**block.input)
                print(f"  tool {block.name}({block.input}) -> {result}")
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        # tool results go back as a *user* message. The model has no other memory.
        messages.append({"role": "user", "content": tool_results})

    print("\n=== full transcript (this is all the model 'knows') ===")
    for m in messages:
        content = m["content"] if isinstance(m["content"], str) else [
            getattr(b, "model_dump", lambda: b)() for b in m["content"]]
        print(m["role"], json.dumps(content, default=str)[:400])
    return "".join(b.text for b in resp.content if b.type == "text")


if __name__ == "__main__":
    answer = run("What's the policy-rate spread between the US and Canada, in basis points? "
                 "And between the UK and the euro area?")
    print("\nFINAL:", answer)

# EXERCISES
# 1. Remove the calculator tool. Does the model still do the math? (It will - but can you trust it?)
# 2. Ask about Japan. Watch how the model handles the "No data" tool result.
# 3. Make get_policy_rate raise an exception. The loop crashes. Fix it by returning the error
#    as a tool_result with "is_error": True - then the model can recover. (core.py does this.)
# 4. Count model calls. Notice the model may call two tools in ONE turn (parallel tool use).
