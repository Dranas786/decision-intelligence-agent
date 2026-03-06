from app.agent.tools import (
    load_dataset,
    profile_table,
    detect_anomalies,
    segment_drivers,
)
from app.llm.factory import get_llm_client


def run_agent(dataset_path: str, question: str, table_name: str = "data") -> dict:
    """
    Main agent workflow:
    1. Load dataset
    2. Ask LLM for a plan
    3. Execute analytics tools
    4. Return structured insights
    """

    df = load_dataset(dataset_path)

    llm = get_llm_client()

    tool_context = {
        "available_tools": [
            "profile_table",
            "detect_anomalies",
            "segment_drivers",
        ],
        "question": question,
        "columns": df.columns.tolist(),
        "row_count": len(df),
    }

    plan_response = llm.plan(tool_context)

    plan = plan_response.get("plan", [])
    insights = []

    for step in plan:
        tool_name = step.get("tool")
        args = step.get("args", {})

        if tool_name == "profile_table":
            result = profile_table(df)
            insights.append(result["summary"])

        elif tool_name == "detect_anomalies":
            metric_col = args.get("metric_col")
            date_col = args.get("date_col")
            result = detect_anomalies(df, metric_col=metric_col, date_col=date_col)
            insights.extend(result["insights"])

        elif tool_name == "segment_drivers":
            metric_col = args.get("metric_col")
            segment_col = args.get("segment_col")
            result = segment_drivers(df, metric_col=metric_col, segment_col=segment_col)
            insights.extend(result["insights"])

        else:
            insights.append(f"Skipped unknown tool: {tool_name}")

    return {
        "plan": [step["tool"] for step in plan],
        "insights": insights,
        "charts": [],
    }