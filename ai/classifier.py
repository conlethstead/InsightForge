"""
Intent classifier: route user message to "visualization" | "analytical" | "semantic" (RAG).

Three-way: viz (chart) -> code-gen + chart; analytical (numbers/table) -> code-gen + result;
semantic (meaning, metadata, explanation) -> RAG. Simple rule/keyword based.
Run from project root: python ai/classifier.py
"""

# Keywords that indicate the user wants a chart or graph
VIZ_KEYWORDS = (
    "chart", "charts", "graph", "graphs", "plot", "plots", "visualize", "visualization",
    "show me a", "draw", "bar chart", "line chart", "pie chart", "histogram", "scatter",
    "heatmap", "dashboard", "figure", "fig",
)

# Keywords that indicate the user wants a computed number or table (analytical path)
ANALYTICAL_KEYWORDS = (
    "what were", "what was", "how many", "how much", "total", "sum of", "average", "avg",
    "by region", "by product", "by month", "by year", "breakdown", "top ", "bottom ",
    "sales by", "revenue by", "count of", "number of", "list the", "show the",
    "which product", "which region", "which customer", "per region", "per product",
)


def classify_intent(message: str) -> str:
    """
    Classify user message as "visualization" | "analytical" | "semantic".

    - visualization: chart/plot intent -> code-gen + chart.
    - analytical: wants a number or table (totals, breakdowns, top-N) -> code-gen + result.
    - semantic: meaning, metadata, explanation -> RAG.

    Args:
        message: Raw user input.

    Returns:
        "visualization" | "analytical" | "semantic"
    """
    if not message or not message.strip():
        return "semantic"
    lower = message.lower().strip()
    if any(kw in lower for kw in VIZ_KEYWORDS):
        return "visualization"
    if any(kw in lower for kw in ANALYTICAL_KEYWORDS):
        return "analytical"
    return "semantic"


if __name__ == "__main__":
    # (1) Semantic (RAG) — expect "semantic"
    for q in ["What does the Region column represent?", "What time period does this dataset cover?"]:
        print(f"  {q!r} -> {classify_intent(q)}")
    # (2) Analytical — expect "analytical"
    for q in ["What were sales by region?", "Which product sold the most?", "Total revenue in 2023"]:
        print(f"  {q!r} -> {classify_intent(q)}")
    # (3) Visualization — expect "visualization"
    for q in ["Show me a bar chart of sales by region", "Plot sales over time", "Draw a pie chart"]:
        print(f"  {q!r} -> {classify_intent(q)}")
