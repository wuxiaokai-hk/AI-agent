"""LESSON 04 - Workflows vs agents: plan -> execute -> critique -> revise.

A plain agent loop decides everything on the fly. For longer tasks it helps to add
*structure in code*:

  PLANNER   one model call, no tools, returns a JSON list of steps (structured output)
  EXECUTOR  the tool-using agent, run once per step, sees prior step results
  CRITIC    a separate call that grades the draft against a checklist (LLM-as-judge)
  REVISE    if the critic fails it, feed the critique back and try again (bounded!)

Rule of thumb (Anthropic's "Building effective agents"): use the simplest thing that
works. Fixed workflows are cheaper and more predictable; free-form agents are for
tasks where you can't predict the steps.

Run:  python lessons/04_planning_and_reflection.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentlab.core import Agent  # noqa: E402
from agentlab.data_tools import READ_ONLY  # noqa: E402

TASK = ("Write a one-page note for a supervisor: did the 2022-23 rate-hiking cycle raise "
        "credit-union delinquency, controlling for unemployment and house prices? "
        "Include magnitude, uncertainty and a robustness check.")

CHECKLIST = """1. Every number is traceable to an analysis step (no invented figures).
2. Reports a coefficient with a standard error or p-value.
3. Includes at least one robustness check (different sample, fixed effects, or added control).
4. Distinguishes correlation from causation.
5. Under 350 words."""


def ask(agent: Agent, prompt: str, max_tokens: int = 1500) -> str:
    r = agent.client.messages.create(model=agent.model, max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
    return r.content[0].text


def extract_json(text: str):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    return json.loads(m.group(0))


def main():
    executor = Agent(system="You are a careful credit-risk econometrician. Use tools; report "
                            "exact numbers from tool output.", tools=READ_ONLY, name="executor")

    # ---- PLAN
    plan = extract_json(ask(executor,
        f"Task: {TASK}\nAvailable tools: describe_dataset, query_data, run_regression.\n"
        "Return ONLY a JSON array of 3-5 short analysis steps (strings). No prose."))
    print("PLAN:", json.dumps(plan, indent=2))

    # ---- EXECUTE each step; results accumulate as a scratchpad
    scratch = []
    for i, step in enumerate(plan, 1):
        context = "\n".join(f"Step {j}: {s}\nResult: {r}" for j, (s, r) in enumerate(scratch, 1))
        out = executor.run(f"Overall task: {TASK}\nDone so far:\n{context or '(nothing)'}\n\n"
                           f"Now do step {i}: {step}\nReport the result briefly with exact numbers.")
        scratch.append((step, out))

    findings = "\n\n".join(f"Step: {s}\nResult: {r}" for s, r in scratch)

    # ---- DRAFT, CRITIQUE, REVISE (max 2 revisions)
    draft = ask(executor, f"Using ONLY these findings, write the note.\nTask: {TASK}\n\n{findings}")
    for attempt in range(3):
        verdict = extract_json(ask(executor,
            f"You are a strict reviewer. Grade the note against the checklist.\n"
            f"Checklist:\n{CHECKLIST}\n\nFindings (ground truth):\n{findings}\n\nNote:\n{draft}\n\n"
            'Return ONLY JSON: {"pass": bool, "failed_items": [int], "feedback": str}'))
        print(f"\nCRITIC (attempt {attempt + 1}):", verdict)
        if verdict["pass"]:
            break
        draft = ask(executor, f"Revise the note to fix this feedback: {verdict['feedback']}\n"
                              f"Use only these findings:\n{findings}\n\nNote:\n{draft}")

    print("\n================ FINAL NOTE ================\n", draft)
    print("\nTotal tokens (executor agent):", executor.usage)


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Make the critic and the writer different models (e.g. a cheaper Haiku writer, stronger critic).
# 2. The critic is checking itself (same model family). Add a *deterministic* check: regex every
#    number in the note and verify it appears in `findings`. Which catches more errors?
# 3. Replace the free-text JSON parsing with the API's structured outputs / a tool whose
#    input_schema IS the plan schema (a common trick: "tool use for structured output").
