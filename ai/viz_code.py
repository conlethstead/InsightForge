"""
Generate Python visualization code from user request via LLM (Step 10).

Uses prompts from ai.prompts and OpenAI. Returns code that uses only
pandas, plotly, numpy and assigns a Plotly figure to `fig`.
Run from project root: python ai/viz_code.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ai.prompts import (
    ANALYTICAL_CODE_SYSTEM,
    get_analytical_code_prompt,
    get_viz_code_prompt,
    VIZ_CODE_SYSTEM,
)

# Schema for sales_data.csv (used in prompt)
DEFAULT_SCHEMA = "Date, Product, Region, Sales, Customer_Age, Customer_Gender, Customer_Satisfaction"


def generate_viz_code(
    user_request: str,
    csv_path: str | Path,
    schema: str = DEFAULT_SCHEMA,
) -> str:
    """
    Generate Python code that loads the CSV and creates a Plotly figure in `fig`.

    Args:
        user_request: Natural language request (e.g. "bar chart of sales by region").
        csv_path: Path to the project CSV.
        schema: Comma-separated column names for the prompt.

    Returns:
        Python code string. Raises ValueError if API key is missing.
    """
    if not config.has_api_key():
        raise ValueError("PAID_OPENAI_API_KEY is not set. Set it in .zshrc or .env.")
    csv_path = str(Path(csv_path).resolve())
    user_prompt = get_viz_code_prompt(user_request, csv_path, schema)
    llm = ChatOpenAI(
        api_key=config.PAID_OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0,
    )
    messages = [
        SystemMessage(content=VIZ_CODE_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    raw = (response.content or "").strip()
    # Strip markdown code fence if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    return raw


def generate_analytical_code(
    user_request: str,
    csv_path: str | Path,
    schema: str = DEFAULT_SCHEMA,
) -> str:
    """
    Generate Python code that loads the CSV and computes the answer, assigning it to `result`.

    Args:
        user_request: Natural language question (e.g. "What were sales by region?").
        csv_path: Path to the project CSV.
        schema: Comma-separated column names for the prompt.

    Returns:
        Python code string. Raises ValueError if API key is missing.
    """
    if not config.has_api_key():
        raise ValueError("PAID_OPENAI_API_KEY is not set. Set it in .zshrc or .env.")
    csv_path = str(Path(csv_path).resolve())
    user_prompt = get_analytical_code_prompt(user_request, csv_path, schema)
    llm = ChatOpenAI(
        api_key=config.PAID_OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0,
    )
    messages = [
        SystemMessage(content=ANALYTICAL_CODE_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    raw = (response.content or "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    return raw


if __name__ == "__main__":
    # Quick test: generate code for a simple request (requires API key)
    req = "Bar chart of total sales by region"
    path = config.DATA_PATH
    print("Request:", req)
    print("CSV path:", path)
    try:
        code = generate_viz_code(req, path)
        print("Generated code (first 400 chars):")
        print(code[:400])
        if len(code) > 400:
            print("...")
    except ValueError as e:
        print("Error:", e)
