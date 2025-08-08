#!/usr/bin/env python3
"""
Universal, token-efficient column cleaner using LangChain + OpenAI.

Usage:
    python module.py --table SCHEMA.TABLE --column COLUMN_NAME --out preview.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import snowflake.connector
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

_LLM: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    """Instantiate (cached) OpenAI Chat model."""
    global _LLM
    if _LLM is None:
        _LLM = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )
    return _LLM


def get_snowflake_conn() -> snowflake.connector.SnowflakeConnection:
    """Create a Snowflake connection using environment variables."""
    params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
    }
    return snowflake.connector.connect(**{k: v for k, v in params.items() if v})


def fetch_sample(table: str, column: str, n: int = 300) -> pd.Series:
    """Fetch ~n random rows from the target column."""
    query = f"SELECT {column} FROM {table} ORDER BY RANDOM() LIMIT {n}"
    with get_snowflake_conn() as conn:
        df = pd.read_sql(query, conn)
    return df[column]


def fetch_full(table: str, column: str, limit: Optional[int]) -> pd.Series:
    """Fetch the full column (optionally limited)."""
    query = f"SELECT {column} FROM {table}"
    if limit:
        query += f" LIMIT {limit}"
    with get_snowflake_conn() as conn:
        df = pd.read_sql(query, conn)
    return df[column]


def profile_series(series: pd.Series) -> Dict[str, Any]:
    """Compute simple profile statistics for a Series."""
    non_null = series.dropna().astype(str)
    lengths = non_null.map(len) if not non_null.empty else pd.Series(dtype=int)
    return {
        "count": int(series.size),
        "nulls": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
        "min_length": int(lengths.min()) if not lengths.empty else 0,
        "max_length": int(lengths.max()) if not lengths.empty else 0,
        "dtype": str(series.dtype),
        "sample_uniques": non_null.unique()[:10].tolist(),
    }


def infer_intent(samples: List[str], profile: Dict[str, Any]) -> Dict[str, Any]:
    """Infer semantic intent for a column via LLM."""
    llm = get_llm()
    system = (
        "You are a senior data quality engineer. Infer semantic_type and target_format "
        "from samples + profile. Return STRICT JSON with keys semantic_type, "
        "target_format, rules[], constraints[]. No prose."
    )
    column = profile.get("column", "column")
    human = (
        f"column: {column}\n"
        f"samples: {json.dumps(samples[:25])}\n"
        f"profile: {json.dumps({k: v for k, v in profile.items() if k != 'column'})}"
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return json.loads(resp.content)


def generate_cleaner_code(intent: Dict[str, Any], examples: List[str]) -> str:
    """Ask LLM to generate a cleaner function."""
    llm = get_llm()
    rules = json.dumps(intent.get("rules", []))
    target_format = intent.get("target_format", "")
    system = (
        "Generate ONLY Python code for a pure function: "
        "def clean(series: pandas.Series) -> pandas.Series using only 're' and 'pandas'. "
        f"Enforce rules: {rules}. Target format: {target_format}. "
        "If impossible/ambiguous, return pd.NA. Output ONE fenced code block."
    )
    human = f"examples: {json.dumps(examples[:25])}"
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    match = re.search(r"```python\n(.*?)```", resp.content, re.DOTALL)
    if not match:
        match = re.search(r"```(.*?)```", resp.content, re.DOTALL)
    if not match:
        raise ValueError("No code block found in LLM response.")
    return match.group(1).strip()


def run_cleaner_locally(code_str: str, series: pd.Series) -> pd.Series:
    """Execute generated cleaner code and apply to series."""
    global_ns: Dict[str, Any] = {"re": re, "pandas": pd, "__builtins__": {}}
    local_ns: Dict[str, Any] = {}
    exec(code_str, global_ns, local_ns)
    if "clean" not in local_ns:
        raise ValueError("Generated code missing 'clean' function.")
    func = local_ns["clean"]
    return func(series.copy())


def normalize_with_llm(intent: Dict[str, Any], values: List[str]) -> List[Optional[str]]:
    """Normalize residual values via LLM."""
    llm = get_llm()
    rules = json.dumps(intent.get("rules", []))
    target_format = intent.get("target_format", "")
    system = (
        f"Normalize each value to {target_format}. Apply rules: {rules}. "
        "If impossible/ambiguous, output null. Return a JSON array aligned to inputs. No prose."
    )
    human = json.dumps(values)
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    try:
        data = json.loads(resp.content)
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.error("Normalization parse error: %s", exc)
    return [None] * len(values)


def validate(series: pd.Series, intent: Dict[str, Any]) -> Dict[str, Any]:
    """Validate cleaned series based on semantic type."""
    semantic_type = intent.get("semantic_type", "")
    ok = True
    errors: List[str] = []

    if semantic_type in {"number", "integer", "float"}:
        conv = pd.to_numeric(series, errors="coerce")
        invalid = conv.isna() & series.notna()
        if invalid.any():
            ok = False
            errors.append(f"{int(invalid.sum())} non-numeric values")
    elif semantic_type in {"date", "datetime"}:
        conv = pd.to_datetime(series, errors="coerce")
        invalid = conv.isna() & series.notna()
        if invalid.any():
            ok = False
            errors.append(f"{int(invalid.sum())} non-date values")
    else:
        if series.isna().any():
            ok = False
            errors.append(f"{int(series.isna().sum())} null values")

    stats = {
        "total": int(series.size),
        "nulls": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
    }
    return {"ok": ok, "stats": stats, "errors": errors}


def save_preview(df: pd.DataFrame, path: str, n: int = 500) -> None:
    """Save a CSV preview (raw vs cleaned)."""
    df.head(n).to_csv(path, index=False)


def chunk_list(data: List[Any], size: int) -> Iterable[List[Any]]:
    """Yield successive chunks of size from data."""
    for i in range(0, len(data), size):
        yield data[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal column cleaner")
    parser.add_argument("--table", required=True, help="Table name, e.g., SCHEMA.TABLE")
    parser.add_argument("--column", required=True, help="Column name")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    parser.add_argument("--out", required=True, help="Output CSV preview path")
    parser.add_argument("--no-codegen", action="store_true", help="Skip code generation")
    args = parser.parse_args()

    sample = fetch_sample(args.table, args.column)
    profile = profile_series(sample)
    profile["column"] = args.column

    samples_for_intent = (
        sample.dropna().astype(str).sample(min(25, len(sample.dropna())), random_state=0).tolist()
        if not sample.dropna().empty
        else []
    )
    intent = infer_intent(samples_for_intent, profile)

    full_series = fetch_full(args.table, args.column, args.limit)
    cleaned = pd.Series([pd.NA] * len(full_series), index=full_series.index)

    if args.no_codegen:
        values = full_series.astype(str).tolist()
        out_values: List[Optional[str]] = []
        for chunk in chunk_list(values, 150):
            out_values.extend(normalize_with_llm(intent, chunk))
        cleaned = pd.Series(out_values, index=full_series.index)
    else:
        examples = (
            sample.dropna().astype(str).sample(min(25, len(sample.dropna())), random_state=1).tolist()
            if not sample.dropna().empty
            else []
        )
        code = generate_cleaner_code(intent, examples)
        print(code)  # Print cleaner code to stdout
        cleaned = run_cleaner_locally(code, full_series)
        residual_mask = cleaned.isna() | (cleaned.astype(str) == full_series.astype(str))
        residual_values = full_series[residual_mask].astype(str).tolist()
        if residual_values:
            normalized: List[Optional[str]] = []
            for chunk in chunk_list(residual_values, 150):
                normalized.extend(normalize_with_llm(intent, chunk))
            cleaned.loc[residual_mask] = normalized

    report = validate(cleaned, intent)
    residual_count = int((cleaned.isna() | (cleaned.astype(str) == full_series.astype(str))).sum())
    coverage = 100 * (1 - residual_count / len(full_series)) if len(full_series) else 100
    print(
        f"Coverage: {coverage:.2f}% | Residuals: {residual_count} | "
        f"Validation: {'OK' if report['ok'] else 'FAIL'}"
    )
    if report["errors"]:
        print("Errors:", "; ".join(report["errors"]))

    df_preview = pd.DataFrame({"raw": full_series, "clean": cleaned})
    diffs = df_preview[df_preview["raw"].astype(str) != df_preview["clean"].astype(str)]
    if not diffs.empty:
        print("Example diffs:")
        print(diffs.head(5).to_string(index=False))

    save_preview(df_preview, args.out)
    print(f"Preview saved to {args.out}")


if __name__ == "__main__":
    main()
