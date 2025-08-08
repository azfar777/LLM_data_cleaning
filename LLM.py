from langchain_openai import AzureChatOpenAI, ChatOpenAI
from typing import List
from security import Settings, get_settings
import pandas as pd
import json

def get_llm_connection():
    """
    Return a ChatOpenAI or AzureChatOpenAI client configured from get_settings().
    """
    settings: Settings = get_settings()
    model = getattr(settings, "MODEL", "gpt-4o")
    deployment = getattr(settings, "AZURE_DEPLOYMENT", model)
    if getattr(settings, "API_ENDPOINT", None):
        return AzureChatOpenAI(
            azure_endpoint=settings.API_ENDPOINT,
            api_version=settings.API_VERSION,
            api_key=settings.API_KEY,
            azure_deployment=deployment,
            model=model,
            temperature=0,
        )
    return ChatOpenAI(api_key=settings.API_KEY, model=model, temperature=0)

def build_prompt(col: str, id_cols: List[str], semantic_type: str, target_format: str, rules: List[str], df: pd.DataFrame, max_rows: int = 200) -> str:
    """
    Return a single string prompt. The model must output pure JSON Lines (one JSON object per line),
    each object containing the id columns and a field f"{col}_CLEAN".
    System instructions to include:

    - "You are a data cleaning expert."
    - "Normalize ONLY the column `{col}`."
    - f"semantic_type: {semantic_type}, target_format: {target_format}"
    - "Apply rules:" + bullet list from `rules`.
    - "If a value cannot be cleaned, set `{col}_CLEAN` to null."
    - "Respond as JSON Lines (no code fences, no explanations). Each line must be a valid JSON object 
       containing ONLY the identifier columns {id_cols} and `{col}_CLEAN`."
    - Provide the input as a JSON array with only id columns + the dirty column.
    """
    subset = df[id_cols + [col]].head(max_rows).to_dict(orient="records")
    rule_lines = "\n".join(f"- {r}" for r in rules) if rules else "- none"
    prompt = (
        "You are a data cleaning expert.\n"
        f"Normalize ONLY the column `{col}`.\n"
        f"semantic_type: {semantic_type}, target_format: {target_format}\n"
        "Apply rules:\n"
        f"{rule_lines}\n"
        f"If a value cannot be cleaned, set `{col}_CLEAN` to null.\n"
        f"Respond as JSON Lines (no code fences, no explanations). Each line must be a valid JSON object containing ONLY the identifier columns {id_cols} and `{col}_CLEAN`.\n"
        "Input:\n"
        f"{json.dumps(subset, ensure_ascii=False)}"
    )
    return prompt

def parse_llm_response(response: str, col: str, id_cols: List[str]) -> pd.DataFrame:
    """
    Parses a JSONL response from the LLM and returns a DataFrame.
    """
    lines = [line for line in response.strip().splitlines() if line.strip()]
    records = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON line: {line}") from e
        record = {key: obj.get(key) for key in id_cols}
        record[f"{col}_CLEAN"] = obj.get(f"{col}_CLEAN")
        records.append(record)
    return pd.DataFrame(records)

def llm_chat(llm, df: pd.DataFrame, col: str, id_cols: List[str], semantic_type: str, target_format: str, rules: List[str]) -> pd.DataFrame:
    """
    Sends a message to the LLM using the JSONL prompt and returns the cleaned data.
    """
    prompt = build_prompt(col, id_cols, semantic_type, target_format, rules, df)
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    return parse_llm_response(raw, col, id_cols)
