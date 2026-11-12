"""LESSON 02 - Tool design: an agent that does real analysis.

Same loop as lesson 01 (now in agentlab/core.py). What changes is the TOOLS.
Most agent quality comes from tool design, not prompt tricks:

  * Narrow, well-named tools beat one "run_python" god-tool (safer, easier to evaluate).
  * Descriptions are documentation for the model: say WHEN to use it and give examples.
  * Return compact, structured results. Huge outputs burn context and confuse the model.
  * Return errors as text; the model reads them and self-corrects.
  * Risky actions (writing files, sending email) get a human approval gate.

Run:  python lessons/02_tools_and_analyst_agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentlab.core import Agent  # noqa: E402
from agentlab.data_tools import ALL  # noqa: E402

SYSTEM = """You are a credit-risk analyst at a provincial regulator.
You have a quarterly panel of 20 credit unions (2015Q1-2025Q4).
Rules:
- Always inspect the data (describe_dataset) before modelling.
- Every number you report must come from a tool result. Never invent figures.
- State the specification you estimated and caveats (e.g. omitted variables, short sample).
- Be concise: bullet points, then a 2-sentence conclusion."""

if __name__ == "__main__":
    agent = Agent(system=SYSTEM, tools=ALL, name="analyst")
    answer = agent.run(
        "How sensitive is credit-union delinquency to unemployment and the policy rate? "
        "Use credit-union fixed effects. Then estimate the average delinquency rate under an "
        "adverse scenario: unemployment 9.5%, policy rate 4.5%, house prices -10% y/y. "
        "Save a short report as adverse_scenario.md."
    )
    print("\n" + answer)
    print("\nTools called:", [t["tool"] for t in agent.trajectory])

# EXERCISES
# 1. Make the description of run_regression vague ("runs a model"). Does the agent misuse it?
# 2. Note the fixed-effects regression's const is the base CU's intercept, not the average.
#    Does the agent notice when plugging into scenario_delinquency? Improve the tool or prompt so it does.
# 3. Add a tool `plot_series(column, cu_id)` that saves a PNG to outputs/ and returns the path.
# 4. Decline the save_report approval. How does the agent react?
