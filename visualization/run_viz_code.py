"""
Run visualization code in a sandbox (Step 10).

Accepts code string and CSV path; enforces import allowlist (pandas, plotly, numpy);
returns figure as JSON (for Streamlit plotly_chart) or error. Timeout and clear errors.
Run as module for subprocess entry: python -m visualization.run_viz_code <code_file> <csv_path>
"""

import ast
import json
import sys
import tempfile
from pathlib import Path
from subprocess import run as subprocess_run, TimeoutExpired

ALLOWED_IMPORTS = frozenset({"pandas", "plotly", "numpy", "pd", "np"})


def _check_imports(code: str) -> str | None:
    """
    Return None if code only uses allowed imports; else return error message.
    Allowed: pandas, plotly, numpy (and aliases pd, np).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base not in ALLOWED_IMPORTS:
                    return f"Disallowed import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base = node.module.split(".")[0]
            if base not in ALLOWED_IMPORTS:
                return f"Disallowed import: {node.module}"
    return None


def _run_in_subprocess(code: str, csv_path: str, timeout_seconds: int = 30) -> dict:
    """
    Execute code in a subprocess; return dict with success, figure_base64, or error.
    Child process: exec code with csv_path and allowed modules, then output fig as base64.
    """
    csv_path = str(Path(csv_path).resolve())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        code_file = f.name
    try:
        # Child script: set up env, exec code, get fig, print base64
        child_script = f'''
import sys
import base64
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

code_file = {repr(code_file)}
csv_path = {repr(csv_path)}

with open(code_file, "r") as f:
    code = f.read()

globs = {{"pd": pd, "px": px, "go": go, "plotly": plotly, "numpy": np, "np": np, "csv_path": csv_path}}
exec(code, globs)
fig = globs.get("fig")
if fig is None:
    print("ERROR: No variable named fig", file=sys.stderr)
    sys.exit(1)
try:
    out = fig.to_json()
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(2)
print(out)
'''
        result = subprocess_run(
            [sys.executable, "-c", child_script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if result.returncode == 0 and result.stdout:
            return {"success": True, "figure_json": result.stdout.strip()}
        err = (result.stderr or result.stdout or "Unknown error").strip()
        return {"success": False, "error": err}
    except TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout_seconds}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        Path(code_file).unlink(missing_ok=True)


def _run_analytical_subprocess(code: str, csv_path: str, timeout_seconds: int = 30) -> dict:
    """
    Execute code in subprocess; expect variable `result` (DataFrame/dict/list/scalar).
    Return {"success": True, "result_type": ..., "result": ...} or {"success": False, "error": ...}.
    """
    csv_path = str(Path(csv_path).resolve())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        code_file = f.name
    try:
        child_script = f'''
import sys
import json
import pandas as pd
import numpy as np

code_file = {repr(code_file)}
csv_path = {repr(csv_path)}

with open(code_file, "r") as f:
    code = f.read()

globs = {{"pd": pd, "pandas": __import__("pandas"), "numpy": np, "np": np, "csv_path": csv_path}}
exec(code, globs)
result = globs.get("result")
if result is None:
    print("ERROR: No variable named result", file=sys.stderr)
    sys.exit(1)
# Serialize for stdout: one line result_type, rest is payload
if isinstance(result, pd.DataFrame):
    out_type = "dataframe"
    payload = result.to_dict(orient="records")
elif isinstance(result, dict):
    out_type = "dict"
    payload = result
elif isinstance(result, list):
    out_type = "list"
    payload = result
else:
    out_type = "scalar"
    payload = str(result)
print(out_type)
print(json.dumps(payload, default=str))
'''
        result = subprocess_run(
            [sys.executable, "-c", child_script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "Unknown error").strip()
            return {"success": False, "error": err}
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return {"success": False, "error": "Malformed output from analytical code"}
        out_type = lines[0].strip()
        payload_str = "\n".join(lines[1:])
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            payload = payload_str
        return {"success": True, "result_type": out_type, "result": payload}
    except TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout_seconds}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        Path(code_file).unlink(missing_ok=True)


def run_analytical_code(
    code: str,
    csv_path: str | Path,
    timeout_seconds: int = 30,
) -> dict:
    """
    Run Python code that computes an answer and assigns it to `result`.
    Same sandbox as viz (pandas, numpy allowed; no plotly required).

    Args:
        code: Python code that must assign the answer to `result` (DataFrame, dict, list, or scalar).
        csv_path: Path to the CSV (injected as csv_path in the sandbox).
        timeout_seconds: Max execution time.

    Returns:
        {"success": True, "result_type": "dataframe"|"dict"|"list"|"scalar", "result": ...}
        or {"success": False, "error": "<message>"}.
    """
    err = _check_imports(code)
    if err is not None:
        return {"success": False, "error": err}
    return _run_analytical_subprocess(code, str(csv_path), timeout_seconds=timeout_seconds)


def run_visualization_code(
    code: str,
    csv_path: str | Path,
    timeout_seconds: int = 30,
) -> dict:
    """
    Run user-provided Python code in a sandbox and return the Plotly figure or error.

    Args:
        code: Python code that must assign a Plotly figure to `fig`.
        csv_path: Path to the project CSV (injected as csv_path in the sandbox).
        timeout_seconds: Max execution time.

    Returns:
        {"success": True, "figure_json": "<plotly figure json>"} or {"success": False, "error": "<message>"}.
        Use plotly.io.from_json() or go.Figure() with the JSON for display. Import allowlist: pandas, plotly, numpy.
    """
    err = _check_imports(code)
    if err is not None:
        return {"success": False, "error": err}
    return _run_in_subprocess(code, str(csv_path), timeout_seconds=timeout_seconds)


if __name__ == "__main__":
    # Entry for subprocess: python -m visualization.run_viz_code <code_file> <csv_path>
    # (Used when we want to run in a separate process; here we use -c inline in _run_in_subprocess.)
    # Standalone test:
    if len(sys.argv) >= 3:
        code_path, csv_path = sys.argv[1], sys.argv[2]
        code = Path(code_path).read_text()
        out = run_visualization_code(code, csv_path)
        print("success:", out.get("success"))
        if out.get("figure_json"):
            print("figure_json length:", len(out["figure_json"]))
        if out.get("error"):
            print("error:", out["error"])
    else:
        # (1) Allowed snippet — should return figure JSON
        good = """
import pandas as pd
import plotly.express as px
df = pd.read_csv(csv_path)
agg = df.groupby("Region")["Sales"].sum().reset_index()
fig = px.bar(agg, x="Region", y="Sales", title="Sales by Region")
"""
        data_path = Path(__file__).resolve().parent.parent / "Dataset" / "sales_data.csv"
        r = run_visualization_code(good, data_path)
        print("Allowed snippet: success =", r.get("success"), "error =" if not r.get("success") else "figure_json length =", r.get("error") or len(r.get("figure_json", "")))

        # (2) Disallowed import — should return error
        bad = "import os\nfig = None"
        r2 = run_visualization_code(bad, data_path)
        print("Disallowed import: success =", r2.get("success"), "error =", r2.get("error"))
