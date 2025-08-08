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

def batch_llm_clean(unprocessed_df, column_name, template_name, batch_size=100):
    """
    Cleans values in batches using the LLM and returns a combined DataFrame.
    Accepts unprocessed_df as input, column_name to clean, and template_name for the LLM prompt.
    """
    results = []
    llm = get_llm_connection()   # create once
    for start in range(0, len(unprocessed_df), batch_size):
        batch = unprocessed_df.iloc[start:start+batch_size].copy()
        cleaned = llm_chat(llm, batch, template_name)
        results.append(cleaned)
    return pd.concat(results, ignore_index=True)

def clean_column_with_llm(
    df,
    col,
    parse_func,
    template_name,
    id_cols=None,
    batch_size=100,
):
    """
    Generalized function to clean a column in a DataFrame using regex and LLM.

    df: DataFrame to clean
    col: column name to clean (e.g., "BITSIZE", "RPM")
    parse_func: function to parse/clean the column (e.g., parse_bitsize)
    template_name: LLM template name (e.g., "BITSIZE", "RPM")
    id_cols: columns to keep as identifiers (default: ["GUID", "TOURID", "TOURDATE"])
    batch_size: LLM batch size

    Returns: DataFrame with a new column {col}_CLEAN
    """
    if id_cols is None:
        id_cols = ["GUID", "TOURID", "TOURDATE"]

    clean_col = f"{col}_CLEAN"
    df[clean_col] = df[col].apply(parse_func)
    logger.info(f"Parsed {col.lower()} values")

    # Find all values that were not processed (i.e., CLEAN is not a float or is NaN)
    def is_unprocessed(row):
        val = row[clean_col]
        return not isinstance(val, float) and not pd.isna(val)

    unprocessed_df = df[df.apply(is_unprocessed, axis=1)][id_cols + [col]].copy()
    logger.info(f"Unprocessed {col.lower()} count", count=len(unprocessed_df))

    if not unprocessed_df.empty:
        llm_processed = batch_llm_clean(unprocessed_df, col, template_name, batch_size)
        # Optionally: merge back into df if needed
        df = df.merge(llm_processed, on=id_cols, how="left", suffixes=("", "_LLM"))
        df[clean_col] = df[f"{col}_CLEAN_LLM"].combine_first(df[clean_col])
        df.drop(columns=[f"{col}_CLEAN_LLM"], inplace=True, errors="ignore")

    # Only round the successfully processed (float) values
    float_mask = df[clean_col].apply(lambda x: isinstance(x, float))
    df.loc[float_mask, clean_col] = df.loc[float_mask, clean_col].round(4)
    logger.info(f"Rounded {col.lower()} values", sample=df[clean_col].head(5).tolist())
    return df

if __name__ == "__main__":
    sfl_read, sfl_write = get_SFL_connection()
    logger.info("Starting bit size cleaning")

    df_bitsize = get_zidc_bit_data(sfl_read)
    logger.info("Loaded data", row_count=len(df_bitsize), columns=list(df_bitsize.columns))

    df_bitsize = clean_column_with_llm(
        df_bitsize,
        col="BITSIZE",
        parse_func=parse_bitsize,
        template_name="BITSIZE"
    )
