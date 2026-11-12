"""LESSON 05 - Multi-agent: an orchestrator that delegates to specialists.

Pattern: orchestrator-workers. The orchestrator's only tool is `delegate`, which
spins up a *fresh* sub-agent with its own system prompt, tools and empty context.

Why bother?
  * Context isolation: each worker sees only its task, not 50 turns of noise.
  * Least privilege: the reviewer gets NO tools, the data agent can't write files.
  * Parallelism (not shown): independent workers can run concurrently.
Costs: more tokens, more latency, and information lost in hand-offs. The orchestrator
must write self-contained task briefs - workers can't see its context.

Run:  python lessons/05_multi_agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentlab.core import Agent, tool  # noqa: E402
from agentlab.data_tools import READ_ONLY  # noqa: E402

SPECIALISTS = {
    "data_analyst": dict(
        system="You pull descriptive statistics from the credit-union panel. Report exact numbers "
               "in a compact table. Do not run regressions.",
        tools=[t for t in READ_ONLY if t.name in ("describe_dataset", "query_data")]),
    "econometrician": dict(
        system="You estimate and interpret regressions on the credit-union panel and compute "
               "scenario implications. Report coefficients with standard errors.",
        tools=READ_ONLY),
    "reviewer": dict(
        system="You are a sceptical senior economist. You have no data access. Critique the "
               "draft you are given: unsupported claims, missing caveats, wrong interpretation. "
               "Return a numbered list of issues, most important first.",
        tools=[]),
}

worker_log: list[dict] = []


@tool("delegate",
      "Send a self-contained task to a specialist and get their answer. The specialist "
      "cannot see this conversation, so include all needed context (numbers, drafts) in `task`. "
      "Specialists: data_analyst (descriptive stats), econometrician (regressions, scenarios), "
      "reviewer (critiques a draft; no data access).",
      {"specialist": {"type": "string", "enum": list(SPECIALISTS)}, "task": {"type": "string"}})
def delegate(specialist, task):
    worker = Agent(name=specialist, **SPECIALISTS[specialist])
    answer = worker.run(task)
    worker_log.append({"specialist": specialist, "tokens": worker.usage})
    return answer


ORCHESTRATOR = """You lead a small credit-risk team. You do not touch data yourself.
Break the request into tasks, delegate them, combine the results, send your draft to the
reviewer, address the most important issues, then give the final answer."""

if __name__ == "__main__":
    boss = Agent(system=ORCHESTRATOR, tools=[delegate], name="orchestrator", max_steps=15)
    final = boss.run("Brief me on credit-union delinquency risk: how it evolved 2019-2025, its "
                     "macro drivers, and what an adverse scenario (unemployment 9.5%, policy rate "
                     "4.5%, HPI -10%) would imply.")
    print("\n================ FINAL ================\n", final)
    print("\nWorker calls:", worker_log)
    print("Orchestrator tokens:", boss.usage)

# EXERCISES
# 1. Compare quality and total tokens against the single agent in lesson 02. Was it worth it?
# 2. Run independent delegations in parallel (concurrent.futures) - the model can emit
#    several delegate calls in one turn.
# 3. Give the reviewer read-only data tools so it can verify numbers. Does it catch more?
