from langchain_openai import AzureChatOpenAI
from typing import Optional, Union
from regex import template
from pydantic import create_model, ValidationError
from security import Settings, get_settings
import pandas as pd
import io
import json

def get_llm_connection():
    """
    Return an AzureChatOpenAI client configured from get_settings().
    """
    settings: Settings = get_settings()
    llm = AzureChatOpenAI(
        azure_endpoint=settings.API_ENDPOINT,
        api_version=settings.API_VERSION,
        api_key=settings.API_KEY,
        azure_deployment="alpha-altaml-c-ai-wu-oai-gpt40",
        model="gpt-4o",
        temperature=0,
    )
    return llm

def get_prompt_template(template_name, df):
    """
    Returns a specific prompt template by name with multi-row support.
    For template_name == "BITSIZE", build a CSV-based prompt that:
    - Says: "You are a data cleaning expert."
    - Task: Clean and standardize ONLY the BITSIZE values in the following CSV data.
    - Do NOT modify GUID, TOURID, or TOURDATE.
    - Treat the following as missing/null values for BITSIZE:
      NOV, NOVR, RERUN, N/A, Unknown, UNKN, --, Empty strings
    - Instructions:
      1. For each row, clean ONLY the BITSIZE value.
      2. Extract any valid number from BITSIZE, convert it to float.
      3. If no number is found in BITSIZE, return null.
      4. Do not change GUID, TOURID, or TOURDATE.
      5. Return a new column called BITSIZE_CLEAN.
    - Output the cleaned data as a valid CSV with columns:
      GUID,TOURID,TOURDATE,BITSIZE_CLEAN
    - Only return the cleaned CSV; no code fences or extra text.
    - Provide a mini example Input/Output in the prompt (like the screenshot), then append the real CSV.
    """
    clean_col_name = template_name + "_CLEAN"
    if template_name == "BITSIZE":
        csv_input = df.to_csv(index=False)
        prompt = (
            "You are a data cleaning expert.\n\n"
            "Your task is to clean and standardize ONLY the BITSIZE values in the following CSV data.\n\n"
            "Do NOT modify GUID, TOURID, or TOURDATE. Copy them as-is to the output.\n\n"
            "Treat the following as missing/null values for BITSIZE:\n"
            " NOV, NOVR, RERUN, N/A, Unknown, UNKN, --, Empty strings\n\n"
            "Instructions:\n"
            "1. For each row, clean ONLY the BITSIZE value.\n"
            "2. Extract any valid number from BITSIZE, convert it to float.\n"
            "3. If no number is found in BITSIZE, return null.\n"
            "4. Do NOT change GUID, TOURID, or TOURDATE.\n"
            f"5. Return a new column called {clean_col_name}.\n\n"
            "Output the cleaned data as a valid CSV with the following columns:\n"
            "GUID,TOURID,TOURDATE,BITSIZE_CLEAN\n\n"
            "Only return the cleaned CSV. Do not include any explanations or formatting like code blocks.\n\n"
            "### Example:\n"
            "Input:\n"
            "GUID,TOURID,TOURDATE,BITSIZE\n"
            "abc,1,20280101,NOV\n"
            "def,2,20280102,12 1/4\\n\n"
            "Output:\n"
            "GUID,TOURID,TOURDATE,BITSIZE_CLEAN\n"
            "abc,1,20280101,\n"
            "def,2,20280102,12.25\n\n"
            "### Input CSV:\n"
            f"{csv_input.strip()}\n"
        )
        return prompt.strip()
    # Default: echo minimal prompt if unknown
    return "Return the input as CSV."

def llm_chat(llm, df, template_name):
    """
    Sends a message to the LLM using the specified prompt template
    and returns the df with the cleaned data.
    """
    prompt_template = get_prompt_template(template_name, df)
    try:
        response = llm.invoke([prompt_template])
    except Exception as e:
        print("LLM invoke failed:", e)
        raise

    # Extract content and parse CSV
    try:
        if hasattr(response, "content"):
            raw = response.content.strip()
        else:
            raw = str(response).strip()
            print("Raw LLM response content:", raw[:200])  # preview
        df_out = parse_llm_response(raw)
        return df_out
    except Exception as e:
        print("Failed to parse LLM output as CSV:", e)
        raise

def parse_llm_response(response):
    """
    Parses the CSV response from the LLM and returns a DataFrame with correct data types.
    Validates presence of GUID, TOURID, TOURDATE, and that BITSIZE_CLEAN is numeric (float-like).
    """
    print("DEBUG: Raw LLM response:", response)
    try:
        df = pd.read_csv(io.StringIO(response))
        print("DEBUG: Parsed DataFrame shape:", df.shape)

        # Validation step before converting
        required_columns = ["GUID", "TOURID", "TOURDATE"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            if not pd.api.types.is_object_dtype(df[col]):
                print(f"Warning: Column {col} is not of type object (string-like) before conversion.")

        if "BITSIZE_CLEAN" in df.columns:
            if not (pd.api.types.is_float_dtype(df["BITSIZE_CLEAN"]) or pd.api.types.is_object_dtype(df["BITSIZE_CLEAN"])):
                print("Warning: BITSIZE_CLEAN not float-like before conversion.")

        # Enforce correct data types
        df["GUID"] = df["GUID"].astype(str)
        df["TOURID"] = df["TOURID"].astype(str)
        df["TOURDATE"] = df["TOURDATE"].astype(str)
        if "BITSIZE_CLEAN" in df.columns:
            df["BITSIZE_CLEAN"] = pd.to_numeric(df["BITSIZE_CLEAN"], errors="coerce")

        return df
    except Exception as e:
        print("Failed to parse LLM output as CSV:", e)
        raise ValueError(f"Failed to parse LLM output as CSV: {e}")
