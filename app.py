"""
InsightForge — Streamlit chat (Step 11).

On new session: prompt for CSV (upload or path); discover schema and build RAG chunks.
Then: user message → intent classifier. If general → RAG over session chunks + memory.
If visualization → LLM generates code using session CSV path and schema → run sandbox → show figure or error.
"""

import logging
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Logging: console + in-app pipeline log viewer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("insightforge").setLevel(logging.INFO)


class StreamlitPipelineLogHandler(logging.Handler):
    """Append log records to st.session_state.pipeline_logs for in-app display."""

    MAX_LINES = 200

    def emit(self, record: logging.LogRecord) -> None:
        try:
            import streamlit as _st
            msg = self.format(record)
            if "pipeline_logs" not in _st.session_state:
                _st.session_state.pipeline_logs = []
            _st.session_state.pipeline_logs.append(msg)
            # Keep last N lines
            if len(_st.session_state.pipeline_logs) > self.MAX_LINES:
                _st.session_state.pipeline_logs = _st.session_state.pipeline_logs[-self.MAX_LINES:]
        except Exception:
            self.handleError(record)


def _ensure_pipeline_log_handler() -> None:
    """Attach a single handler to the parent logger so each message is only handled once."""
    log = logging.getLogger("insightforge")
    # Remove any existing pipeline handlers to avoid duplicates across Streamlit reruns
    for h in log.handlers[:]:
        if isinstance(h, StreamlitPipelineLogHandler):
            log.removeHandler(h)
    handler = StreamlitPipelineLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)


import pandas as pd
import streamlit as st

import config
from ai.classifier import classify_intent
from ai.memory import ConversationMemory
from ai.rag import rag_answer
from ai.viz_code import generate_analytical_code, generate_viz_code
from data.knowledge_base import get_chunks
from data.schema import discover_schema
from visualization.run_viz_code import run_analytical_code, run_visualization_code

# Page config
st.set_page_config(page_title="InsightForge", page_icon="📊", layout="centered")

# Session state: CSV/schema/chunks for this session; messages; memory; debug
if "current_csv_path" not in st.session_state:
    st.session_state.current_csv_path = None
if "schema" not in st.session_state:
    st.session_state.schema = None  # comma-separated column names string
if "schema_obj" not in st.session_state:
    st.session_state.schema_obj = None  # DataSchema for viz/analytical
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "last_rag_context" not in st.session_state:
    st.session_state.last_rag_context = None
if "last_rag_query" not in st.session_state:
    st.session_state.last_rag_query = None
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []

_ensure_pipeline_log_handler()


def render_message(role: str, content: str, figure_json: str | None = None) -> None:
    with st.chat_message(role):
        st.markdown(content)
        if figure_json:
            import plotly.io as pio
            fig = pio.from_json(figure_json)
            st.plotly_chart(fig, width=1000)


def handle_general(user_message: str) -> tuple[str, str | None]:
    """Run RAG with session chunks and optional conversation context; return (answer_text, None)."""
    conv_ctx = st.session_state.memory.get_messages_for_context(max_turns=5)
    try:
        answer, context_used = rag_answer(
            user_message,
            conversation_context=conv_ctx or None,
            return_context=True,
            chunks=st.session_state.chunks,
        )
        st.session_state.last_rag_query = user_message
        st.session_state.last_rag_context = context_used
        return answer, None
    except ValueError as e:
        st.session_state.last_rag_context = None
        return f"Configuration error: {e}", None
    except Exception as e:
        st.session_state.last_rag_context = None
        return f"Error: {e}", None


def handle_visualization(user_message: str) -> tuple[str, str | None]:
    """Generate code using session CSV path and schema; run in sandbox; return (code_block + status, figure_json or None)."""
    csv_path = st.session_state.current_csv_path
    schema = st.session_state.schema or ""
    try:
        code = generate_viz_code(user_message, csv_path, schema=schema)
    except ValueError as e:
        return f"Configuration error: {e}", None
    except Exception as e:
        return f"Code generation failed: {e}", None

    code_block = f"```python\n{code}\n```"
    result = run_visualization_code(code, csv_path)

    if result.get("success") and result.get("figure_json"):
        return code_block + "\n\n**Chart:**", result["figure_json"]
    err = result.get("error", "Unknown error")
    return code_block + f"\n\n**Execution error:** {err}", None


def handle_analytical(user_message: str) -> tuple[str, str | None, dict | None]:
    """
    Generate analytical code (result not fig), run in sandbox, return (answer_text, None, result_payload).
    result_payload: {"result_type": "dataframe"|"dict"|"list"|"scalar", "result": ...} for table/text display.
    """
    csv_path = st.session_state.current_csv_path
    schema = st.session_state.schema or ""
    try:
        code = generate_analytical_code(user_message, csv_path, schema=schema)
    except ValueError as e:
        return f"Configuration error: {e}", None, None
    except Exception as e:
        return f"Code generation failed: {e}", None, None

    code_block = f"```python\n{code}\n```"
    run_result = run_analytical_code(code, csv_path)

    if not run_result.get("success"):
        err = run_result.get("error", "Unknown error")
        return code_block + f"\n\n**Execution error:** {err}", None, None

    result_type = run_result.get("result_type", "scalar")
    result_data = run_result.get("result")
    if result_type == "dataframe" and isinstance(result_data, list):
        text = "**Result:**\n\n" + _format_result_as_text(result_type, result_data)
        return code_block + "\n\n" + text, None, {"result_type": result_type, "result": result_data}
    text = "**Result:**\n\n" + _format_result_as_text(result_type, result_data)
    return code_block + "\n\n" + text, None, {"result_type": result_type, "result": result_data}


def _format_result_as_text(result_type: str, result_data) -> str:
    """Format analytical result for markdown display."""
    if result_type == "dataframe" and isinstance(result_data, list) and result_data:
        import pandas as pd
        df = pd.DataFrame(result_data)
        if hasattr(df, "to_markdown"):
            try:
                return df.to_markdown(index=False)
            except ImportError:
                # pandas.to_markdown requires the optional 'tabulate' package
                return df.to_string(index=False)
        return df.to_string(index=False)
    if result_type in ("dict", "list"):
        import json
        return f"```json\n{json.dumps(result_data, default=str, indent=2)}\n```"
    return str(result_data)


# Title and API key check
st.title("InsightForge")
if not config.has_api_key():
    st.error("PAID_OPENAI_API_KEY is not set. Set it in your environment (e.g. .zshrc or .env).")
    st.stop()

# When no CSV is set: show CSV selection form
if st.session_state.current_csv_path is None:
    st.subheader("Choose a dataset")
    st.caption("Upload a CSV or enter a path to get started. This dataset will be used for chat and visualizations for this session.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
    path_input = st.text_input("Or enter path to a CSV file", placeholder="e.g. Dataset/sales_data.csv")
    if st.button("Load dataset"):
        path_to_use = None
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                path_to_use = tmp.name
        elif path_input and path_input.strip():
            p = Path(path_input.strip())
            if not p.is_absolute():
                p = _PROJECT_ROOT / p
            path_to_use = str(p)
        if path_to_use:
            try:
                df = pd.read_csv(path_to_use)
                schema_info = discover_schema(df=df)
                st.session_state.schema = schema_info.schema_str
                st.session_state.schema_obj = schema_info
                st.session_state.chunks = get_chunks(df=df)
                st.session_state.current_csv_path = path_to_use
                st.success(f"Loaded {len(df)} rows, {len(st.session_state.chunks)} chunks. Columns: {st.session_state.schema}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")
        else:
            st.warning("Please upload a CSV file or enter a path.")
    st.stop()

# We have a session CSV: show New session button, then chat UI
if st.button("New session"):
    st.session_state.current_csv_path = None
    st.session_state.schema = None
    st.session_state.schema_obj = None
    st.session_state.chunks = None
    st.session_state.messages = []
    st.session_state.memory = ConversationMemory()
    st.session_state.last_rag_context = None
    st.session_state.last_rag_query = None
    st.rerun()

st.caption(f"Dataset: {st.session_state.current_csv_path}")

# Pipeline logs (RAG + retriever steps) — visible in app
with st.expander("Pipeline logs (RAG & retriever steps)", expanded=False):
    if st.session_state.pipeline_logs:
        log_text = "\n".join(st.session_state.pipeline_logs)
        st.text_area("Recent logs", value=log_text, height=300, disabled=True, label_visibility="collapsed")
        if st.button("Clear pipeline logs", key="clear_logs"):
            st.session_state.pipeline_logs = []
            st.rerun()
    else:
        st.caption("Ask a general question to see retrieval and RAG step logs here.")

# Debug expander
with st.expander("Debug: last RAG context (query + context sent to LLM)"):
    if st.session_state.last_rag_query:
        st.text("Query: " + st.session_state.last_rag_query)
        if st.session_state.last_rag_context:
            st.text_area("Context sent to LLM", value=st.session_state.last_rag_context, height=200, disabled=True)
        else:
            st.caption("No context stored (e.g. last turn was not a general question).")
    else:
        st.caption("Ask a general question to see the retrieved context here.")

# Render past messages
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("figure_json"))

# Chat input
if prompt := st.chat_input("Ask about your data or request a chart..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "figure_json": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = classify_intent(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if intent == "semantic":
                answer, fig_json = handle_general(prompt)
                st.markdown(answer)
                if fig_json:
                    import plotly.io as pio
                    st.plotly_chart(pio.from_json(fig_json), width=1000)
                st.session_state.memory.add_turn(prompt, answer)
                result_payload = None
            elif intent == "analytical":
                answer, fig_json, result_payload = handle_analytical(prompt)
                st.markdown(answer)
                if result_payload and result_payload.get("result_type") == "dataframe":
                    r = result_payload.get("result")
                    if isinstance(r, list) and r:
                        st.dataframe(pd.DataFrame(r), use_container_width=True)
                st.session_state.memory.add_turn(prompt, answer)
            else:
                answer, fig_json = handle_visualization(prompt)
                st.markdown(answer)
                if fig_json:
                    import plotly.io as pio
                    st.plotly_chart(pio.from_json(fig_json), width=1000)
                st.session_state.memory.add_turn(prompt, answer)
                result_payload = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "figure_json": fig_json,
    })


if __name__ == "__main__":
    import config as _config
    from ai.classifier import classify_intent as _ci
    assert _config.DATA_PATH is not None
    assert _ci("chart") == "visualization"
    assert _ci("what is sales?") == "semantic"
    assert _ci("What were sales by region?") == "analytical"
    print("Config and classifier OK. Run with: streamlit run app.py")
