"""LESSON 07 - Evals: how you know your agent works (and didn't get worse).

An agent is a stochastic program. "It looked right when I tried it" is not evidence.
Like out-of-sample testing for a forecast model, you need a fixed test set + a metric.

Three layers, cheapest first:
  1. TOOL TESTS   - ordinary unit tests on the tools (no LLM, deterministic).
  2. OUTCOME      - did the final answer contain the right number (within tolerance)?
  3. TRAJECTORY   - did it use the right tools, how many steps, any errors, how many tokens?
(4. LLM-as-judge for fuzzy qualities - seen in lesson 04 - calibrate it against human grades.)

Ground truth here is computed directly from the data, so it's exact.

Run:  python lessons/07_evals.py --offline   # layer 1 only, free
      python lessons/07_evals.py             # full eval, ~6 agent runs
      python lessons/07_evals.py --trials 3  # repeat each case to measure consistency
"""
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentlab import data_tools as dt  # noqa: E402

df = dt.load()


# ------------------------------------------------------------------ 1. tool tests
def test_tools():
    r = dt.run_regression.fn("delinquency_rate", ["unemployment", "policy_rate", "hpi_yoy"], entity_fe=True)
    from data.make_data import TRUE_BETA  # the known data-generating process
    for k in ("unemployment", "policy_rate", "hpi_yoy"):
        assert abs(r["coef"][k] - TRUE_BETA[k]) < 0.03, (k, r["coef"][k])
    assert dt.query_data.fn(["loans_bn"], where="cu_id == 'CU01'")["n_rows"] == 44
    try:
        dt.scenario_delinquency.fn({"const": 1, "unemployment": 0.2}, {})
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("tool tests: PASS (regression recovers true betas:", r["coef"], ")")


# ------------------------------------------------------------------ 2+3. agent eval set
def truth_mean(where):
    return round(df.query(where).delinquency_rate.mean(), 3)


CASES = [
    dict(q="What was the average delinquency rate across all credit unions in 2020? Give the number to 3 decimals.",
         expect=truth_mean("quarter.str.startswith('2020')"), tol=0.01, tools={"query_data"}),
    dict(q="Which credit union had the highest average delinquency rate over the whole sample? Answer with its id.",
         expect_text=df.groupby("cu_id").delinquency_rate.mean().idxmax(), tools={"query_data"}),
    dict(q="Regress delinquency_rate on unemployment, policy_rate and hpi_yoy with credit-union fixed "
           "effects. What is the unemployment coefficient (3 decimals)?",
         expect=dt.run_regression.fn("delinquency_rate", ["unemployment", "policy_rate", "hpi_yoy"],
                                     entity_fe=True)["coef"]["unemployment"], tol=0.002, tools={"run_regression"}),
    dict(q="How many credit unions are in the dataset?", expect=df.cu_id.nunique(), tol=0, tools=set()),
    dict(q="What was CU05's delinquency rate in 2025Q4?",
         expect=float(df.query("cu_id=='CU05' and quarter=='2025Q4'").delinquency_rate.iloc[0]), tol=0.001,
         tools={"query_data"}),
    # Adversarial: the data doesn't contain this. Correct behaviour = say so, not invent.
    dict(q="What was the delinquency rate for credit cards specifically in 2024?",
         expect_refusal=True, tools=set()),
]

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def grade(case, answer: str) -> bool:
    if "expect" in case:
        return any(abs(float(n) - case["expect"]) <= case["tol"] + 1e-9 for n in NUM.findall(answer.replace(",", "")))
    if "expect_text" in case:
        return case["expect_text"] in answer
    if case.get("expect_refusal"):
        return bool(re.search(r"not (available|included|contain|in the data)|doesn'?t (contain|include|have)|"
                              r"no (data|breakdown|information)|cannot|can't|unable", answer, re.I))
    return False


def run_evals(trials: int = 1):
    from agentlab.core import Agent
    system = ("You are a credit-risk analyst with a credit-union panel. Use tools; never invent "
              "numbers. If the data can't answer, say so plainly.")
    rows = []
    for i, case in enumerate(CASES):
        for t in range(trials):
            agent = Agent(system=system, tools=dt.READ_ONLY, verbose=False, name=f"eval{i}")
            ans = agent.run(case["q"])
            used = {s["tool"] for s in agent.trajectory}
            rows.append(dict(case=i, trial=t, correct=grade(case, ans),
                             right_tools=case["tools"] <= used, steps=len(agent.trajectory),
                             errors=sum(s["error"] for s in agent.trajectory),
                             tokens=agent.usage["input_tokens"] + agent.usage["output_tokens"]))
            mark = "PASS" if rows[-1]["correct"] else "FAIL"
            print(f"[{mark}] case {i} trial {t}: tools={sorted(used)} | {ans[:120]!r}")

    print("\n=========== SCORECARD ===========")
    print(f"accuracy        {mean(r['correct'] for r in rows):.0%}")
    print(f"right tools     {mean(r['right_tools'] for r in rows):.0%}")
    print(f"avg tool calls  {mean(r['steps'] for r in rows):.1f}")
    print(f"tool errors     {sum(r['errors'] for r in rows)}")
    print(f"avg tokens/case {mean(r['tokens'] for r in rows):,.0f}")
    if trials > 1:  # consistency: does each case pass on every trial?
        per_case = {c: [r["correct"] for r in rows if r["case"] == c] for c in range(len(CASES))}
        print("pass^k (all trials pass):", f"{mean(all(v) for v in per_case.values()):.0%}")


if __name__ == "__main__":
    test_tools()
    if "--offline" not in sys.argv:
        n = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 1
        run_evals(n)

# EXERCISES
# 1. Change the system prompt (e.g. drop "never invent numbers") and re-run. Did the score move?
#    This is how you make prompt changes safely - like a backtest.
# 2. Switch AGENTLAB_MODEL to a smaller model. Plot accuracy vs tokens: a cost/quality frontier.
# 3. Add 5 cases from your real work questions. The eval set is your most valuable asset.
