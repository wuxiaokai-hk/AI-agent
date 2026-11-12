"""Generate a synthetic quarterly panel of Ontario-style credit unions.

Columns: cu_id, quarter, loans_bn, delinquency_rate (%), unemployment (%),
policy_rate (%), hpi_yoy (%), mortgage_share.

The data-generating process is known, so the evals in lesson 07 can check
whether the agent recovers the "true" coefficients.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TRUE_BETA = {"const": 0.40, "unemployment": 0.18, "policy_rate": 0.09, "hpi_yoy": -0.02}


def make(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2015Q1", "2025Q4", freq="Q")
    n_q = len(quarters)

    # Macro block (common to all credit unions)
    unemp = 6.5 + np.cumsum(rng.normal(0, 0.25, n_q))
    unemp[21:25] += np.array([5.0, 3.5, 2.0, 1.0])  # 2020 shock
    policy = np.clip(1.0 + np.cumsum(rng.normal(0, 0.2, n_q)), 0.25, 5.0)
    policy[29:36] = np.linspace(0.5, 5.0, 7)  # 2022-23 hiking cycle
    hpi = 6 + np.cumsum(rng.normal(0, 1.2, n_q))

    rows = []
    for cu in range(1, 21):
        fe = rng.normal(0, 0.15)
        size = rng.lognormal(0.5, 0.8)
        mshare = rng.uniform(0.35, 0.75)
        for t, q in enumerate(quarters):
            dq = (TRUE_BETA["const"] + fe
                  + TRUE_BETA["unemployment"] * unemp[t]
                  + TRUE_BETA["policy_rate"] * policy[t]
                  + TRUE_BETA["hpi_yoy"] * hpi[t]
                  + rng.normal(0, 0.08))
            rows.append({
                "cu_id": f"CU{cu:02d}", "quarter": str(q),
                "loans_bn": round(size * (1.012 ** t), 3),
                "delinquency_rate": round(max(dq, 0.05), 3),
                "unemployment": round(unemp[t], 2),
                "policy_rate": round(policy[t], 2),
                "hpi_yoy": round(hpi[t], 2),
                "mortgage_share": round(mshare, 2),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "credit_unions.csv"
    df = make()
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
