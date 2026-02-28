"""Prompt templates for the BI assistant (RAG and direct data)."""

# ---------------------------------------------------------------------------
# System prompt — BI assistant role and instructions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Business Intelligence assistant. You help users understand their sales data by answering questions clearly and accurately.

Rules:
- Base your answers only on the context provided. If the context does not contain enough information, say so.
- Prefer concise answers; use bullet points or short paragraphs when helpful.
- When citing numbers or metrics, use the values from the context.
- Do not make up data or sources."""


def get_system_prompt(extra_instructions: str | None = None) -> str:
    """Return the system prompt, optionally with extra instructions appended."""
    if extra_instructions:
        return f"{SYSTEM_PROMPT}\n\n{extra_instructions}"
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# User prompt — RAG: context + question
# ---------------------------------------------------------------------------

RAG_USER_TEMPLATE = """Use the following context to answer the question.
{conversation_block}
Context:
{context}

Question: {question}

Answer:"""


def get_rag_user_prompt(
    context: str,
    question: str,
    conversation_context: str | None = None,
) -> str:
    """Build the user prompt for RAG: optional prior conversation, context, and question."""
    block = ""
    if conversation_context and conversation_context.strip():
        block = "Previous conversation (for follow-up questions):\n" + conversation_context.strip() + "\n\n"
    return RAG_USER_TEMPLATE.format(
        conversation_block=block,
        context=context.strip(),
        question=question.strip(),
    )


# ---------------------------------------------------------------------------
# User prompt — direct data snippet (e.g. for Step 7 LLM-only summary)
# ---------------------------------------------------------------------------

SUMMARY_USER_TEMPLATE = """Summarize or analyze the following data. Be concise and highlight the main insights or metrics.

Data:
{data_snippet}

Summary or analysis:"""


def get_summary_prompt(data_snippet: str) -> str:
    """Build the user prompt for summarizing a data snippet (no retrieval)."""
    return SUMMARY_USER_TEMPLATE.format(data_snippet=data_snippet.strip())


# ---------------------------------------------------------------------------
# Visualization code generation (Step 10)
# ---------------------------------------------------------------------------

VIZ_CODE_ALLOWED_IMPORTS = "pandas, plotly, numpy"

VIZ_CODE_SYSTEM = f"""You generate Python code that loads a CSV and creates a single Plotly figure.

Rules:
- Use ONLY these imports: {VIZ_CODE_ALLOWED_IMPORTS}. No other modules.
- Load the CSV from the path given in the prompt into a DataFrame (e.g. df = pd.read_csv(csv_path)).
- Before any arithmetic (division, multiplication, etc.) on columns, ensure they are numeric. CSV columns are often read as strings (e.g. Cost with currency, numbers with commas). Use pd.to_numeric(df['col'], errors='coerce') for columns used in calculations, then drop or handle rows with NaN if needed.
- Create a Plotly figure and assign it to the variable named exactly: fig
- Do not use subplots unless the user asks for multiple charts; one fig is enough.
- Output only runnable Python code, no markdown or explanation."""

VIZ_CODE_USER_TEMPLATE = """CSV path: {csv_path}

CSV columns: {schema}

User request: {user_request}

Generate Python code that assigns the Plotly figure to `fig`. Code only."""


def get_viz_code_prompt(user_request: str, csv_path: str, schema: str) -> str:
    """Build the user prompt for visualization code generation."""
    return VIZ_CODE_USER_TEMPLATE.format(
        csv_path=csv_path,
        schema=schema,
        user_request=user_request.strip(),
    )


# ---------------------------------------------------------------------------
# Analytical code generation (result not fig)
# ---------------------------------------------------------------------------

ANALYTICAL_CODE_ALLOWED_IMPORTS = "pandas, numpy"

ANALYTICAL_CODE_SYSTEM = f"""You generate Python code that loads a CSV and computes the answer to the user's question.

Rules:
- Use ONLY these imports: {ANALYTICAL_CODE_ALLOWED_IMPORTS}. No other modules (no plotly).
- Load the CSV from the path given in the prompt into a DataFrame (e.g. df = pd.read_csv(csv_path)).
- Before any arithmetic (division, multiplication, etc.) on columns, ensure they are numeric. CSV columns are often read as strings (e.g. Cost with currency, numbers with commas). Use pd.to_numeric(df['col'], errors='coerce') for columns used in calculations, then drop or handle rows with NaN if needed.
- Compute the answer (totals, breakdowns, top-N, counts, etc.) and assign it to the variable named exactly: result
- result must be one of: a pandas DataFrame, a dict, a list, or a single number/string.
- Output only runnable Python code, no markdown or explanation."""

ANALYTICAL_CODE_USER_TEMPLATE = """CSV path: {csv_path}

CSV columns: {schema}

User question: {user_request}

Generate Python code that computes the answer and assigns it to `result`. Code only."""


def get_analytical_code_prompt(user_request: str, csv_path: str, schema: str) -> str:
    """Build the user prompt for analytical (data) code generation."""
    return ANALYTICAL_CODE_USER_TEMPLATE.format(
        csv_path=csv_path,
        schema=schema,
        user_request=user_request.strip(),
    )


# ---------------------------------------------------------------------------
# Verification (Step 6)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # (1) Format system prompt (no or dummy placeholders), print first 200 chars.
    system = get_system_prompt()
    print("System prompt (first 200 chars):")
    print(system[:200])
    print("...")
    print()

    # (2) Format user prompt with dummy context/question, print first 200 chars.
    dummy_context = "Sales by region: North 1000, South 800, East 1200."
    dummy_question = "What were sales by region?"
    user_rag = get_rag_user_prompt(dummy_context, dummy_question)
    print("RAG user prompt (first 200 chars):")
    print(user_rag[:200])
    print("...")
    print()

    # Bonus: summary prompt
    dummy_snippet = "Product A: 50 units. Product B: 30 units. Product C: 45 units."
    user_summary = get_summary_prompt(dummy_snippet)
    print("Summary user prompt (first 200 chars):")
    print(user_summary[:200])
    print("...")
