import truststore
truststore.inject_into_ssl()
import pandas as pd
import sys
import logging
import structlog
from security import get_SFL_connection
from sf1_data_query import get_zidc_bit_data, get_zidc_rpm_data
from regex_func import parse_bitsize, parse_rpm, parse_od_number
from LLM import get_llm_connection, llm_chat

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

def batch_llm_clean(unprocessed_df, col, id_cols, semantic_type, target_format, rules, batch_size=120):
    """
    Cleans values in batches using the LLM and returns a combined DataFrame.
    """
    results = []
    llm = get_llm_connection()
    for start in range(0, len(unprocessed_df), batch_size):
        batch = unprocessed_df.iloc[start:start+batch_size].copy()
        cleaned = llm_chat(llm, batch, col, id_cols, semantic_type, target_format, rules)
        results.append(cleaned)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=id_cols + [f"{col}_CLEAN"])

def clean_column_with_llm(
    df: pd.DataFrame,
    col: str,
    parse_func,
    id_cols: list[str],
    semantic_type: str,
    target_format: str,
    rules: list[str],
    batch_size: int = 120,
) -> pd.DataFrame:
    """
    Column-agnostic cleaning pipeline using regex/LLM.
    """
    clean_col = f"{col}_CLEAN"
    df[clean_col] = df[col].apply(parse_func)
    residual_mask = (
        df[clean_col].isna()
        | (df[clean_col].astype(str).str.strip() == "")
        | (df[clean_col].astype(str) == df[col].astype(str))
    )
    unprocessed_df = df.loc[residual_mask, id_cols + [col]].copy()
    logger.info(f"Unprocessed {col.lower()} count", count=len(unprocessed_df))

    if not unprocessed_df.empty:
        llm_processed = batch_llm_clean(unprocessed_df, col, id_cols, semantic_type, target_format, rules, batch_size)
        df = df.merge(llm_processed, on=id_cols, how="left", suffixes=("", "_LLM"))
        df[clean_col] = df[f"{clean_col}_LLM"].combine_first(df[clean_col])
        df.drop(columns=[f"{clean_col}_LLM"], inplace=True, errors="ignore")

    if semantic_type == "numeric" and target_format == "2_decimals":
        df[clean_col] = pd.to_numeric(df[clean_col], errors="coerce").round(2)

    return df

if __name__ == "__main__":
    # Example usage with an in-memory DataFrame
    demo_df = pd.DataFrame({
        "GUID": ["a", "b"],
        "TOURID": ["1", "2"],
        "TOURDATE": ["2024-01-01", "2024-01-02"],
        "VALUE": ["123.456", "bad data"],
    })

    def parse_numeric(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    cleaned_df = clean_column_with_llm(
        demo_df,
        col="VALUE",
        parse_func=parse_numeric,
        id_cols=["GUID", "TOURID", "TOURDATE"],
        semantic_type="numeric",
        target_format="2_decimals",
        rules=["Extract numbers and format to two decimals"],
    )
    print(cleaned_df)
