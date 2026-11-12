"""Analyst tools over the credit-union panel. Pure Python: you can unit-test them
without any LLM (see tests in lesson 07). Good agents start with good tools."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from .core import tool

DATA = Path(__file__).resolve().parent.parent / "data" / "credit_unions.csv"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def load() -> pd.DataFrame:
    if not DATA.exists():
        from data.make_data import make  # type: ignore
        make().to_csv(DATA, index=False)
    return pd.read_csv(DATA)


@tool("describe_dataset",
      "Return the columns, dtypes, row count, date range and summary statistics of the "
      "credit-union panel. Call this first before any analysis.",
      {})
def describe_dataset():
    df = load()
    return {
        "rows": len(df), "columns": dict(df.dtypes.astype(str)),
        "quarters": [df.quarter.min(), df.quarter.max()],
        "n_credit_unions": df.cu_id.nunique(),
        "summary": df.describe().round(3).to_dict(),
    }


@tool("query_data",
      "Filter and aggregate the panel with a pandas query. `where` is a pandas .query() "
      "string (e.g. \"cu_id == 'CU03' and quarter >= '2020Q1'\") or empty. `group_by` is "
      "a column or empty. `agg` is mean|sum|median|max|min. Returns at most 60 rows.",
      {"where": {"type": "string"}, "columns": {"type": "array", "items": {"type": "string"}},
       "group_by": {"type": "string"}, "agg": {"type": "string", "enum": ["mean", "sum", "median", "max", "min"]}},
      required=["columns"])
def query_data(columns, where="", group_by="", agg="mean"):
    df = load()
    if where:
        df = df.query(where)
    if group_by:
        df = df.groupby(group_by)[columns].agg(agg).reset_index()
    else:
        df = df[columns]
    return {"n_rows": len(df), "rows": df.head(60).round(4).to_dict(orient="records")}


@tool("run_regression",
      "OLS of `y` on `x` columns over the panel (optionally filtered by `where`). "
      "Set entity_fe=true for credit-union fixed effects. Returns coefficients, std errors "
      "(clustered by cu_id), R-squared and N.",
      {"y": {"type": "string"}, "x": {"type": "array", "items": {"type": "string"}},
       "where": {"type": "string"}, "entity_fe": {"type": "boolean"}},
      required=["y", "x"])
def run_regression(y, x, where="", entity_fe=False):
    df = load()
    if where:
        df = df.query(where)
    X = df[x].copy()
    if entity_fe:
        X = pd.concat([X, pd.get_dummies(df.cu_id, prefix="fe", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    res = sm.OLS(df[y], X).fit(cov_type="cluster", cov_kwds={"groups": df.cu_id.astype("category").cat.codes})
    keep = ["const"] + list(x)
    return {
        "coef": res.params[keep].round(4).to_dict(),
        "std_err": res.bse[keep].round(4).to_dict(),
        "p_value": res.pvalues[keep].round(4).to_dict(),
        "r2": round(res.rsquared, 4), "n": int(res.nobs),
    }


@tool("scenario_delinquency",
      "Given regression coefficients and a macro scenario, compute the implied average "
      "delinquency rate. `coef` maps variable->coefficient (must include const); "
      "`scenario` maps variable->value.",
      {"coef": {"type": "object"}, "scenario": {"type": "object"}})
def scenario_delinquency(coef, scenario):
    missing = [k for k in coef if k != "const" and k not in scenario]
    if missing:
        raise ValueError(f"scenario is missing values for {missing}")
    val = coef.get("const", 0.0) + sum(coef[k] * scenario[k] for k in coef if k != "const")
    return {"implied_delinquency_rate": round(float(val), 4)}


@tool("save_report",
      "Write a markdown report to outputs/<filename>. Requires human approval.",
      {"filename": {"type": "string"}, "markdown": {"type": "string"}},
      needs_approval=True)
def save_report(filename, markdown):
    OUT.mkdir(exist_ok=True)
    path = OUT / Path(filename).name  # .name blocks path traversal like ../../etc
    path.write_text(markdown)
    return f"saved {path}"


ALL = [describe_dataset, query_data, run_regression, scenario_delinquency, save_report]
READ_ONLY = [describe_dataset, query_data, run_regression, scenario_delinquency]
