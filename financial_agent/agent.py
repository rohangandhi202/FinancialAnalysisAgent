"""
financial_agent/agent.py

The Claude-powered reasoning layer. Takes financial metrics from the analysis
engine and produces structured investment analysis via tool use.

The agent follows a multi-step chain:
  1. Fetch quantitative metrics (via tools)
  2. Reason over strengths, risks, and anomalies
  3. Return a structured JSON report

Usage:
    python -m financial_agent.agent --ticker AAPL
    python -m financial_agent.agent --ticker AAPL --vs MSFT
"""

import json
import os
import argparse
import logging
from pathlib import Path
from dataclasses import asdict

import anthropic

from financial_agent.financial_analysis import (
    analyze_ticker,
    analyze_tickers,
    FinancialMetrics,
)
from financial_agent.data_ingestion import init_db, get_recent_prices

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = Path("financial_data.db")

# Load API key from environment (set in .env or shell)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# ── Tool definitions (what Claude can call) ────────────────────────────────────

TOOLS = [
    {
        "name": "get_financial_metrics",
        "description": (
            "Retrieves computed financial ratios and metrics for a company. "
            "Returns profitability ratios (net margin, ROE, ROA), solvency ratios "
            "(D/E, interest coverage), liquidity (current ratio), cash flow quality "
            "(FCF/NI ratio), YoY growth rates, and anomaly flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_price_history",
        "description": (
            "Retrieves recent stock price history for a company. "
            "Returns closing prices, volume, and computed 52-week high/low. "
            "Use to assess current valuation context and recent price momentum."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of trading days to retrieve (default: 90)",
                    "default": 90
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "compare_companies",
        "description": (
            "Compares financial metrics across two or more companies side by side. "
            "Returns a structured comparison of key ratios, growth rates, and flags "
            "to identify relative strengths and weaknesses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to compare, e.g. ['AAPL', 'MSFT']"
                }
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "get_anomaly_flags",
        "description": (
            "Returns a focused summary of any anomaly flags detected for a company: "
            "FCF divergence from net income (earnings quality concern), "
            "earnings miss (revenue up but income down), "
            "margin compression, or debt spikes. "
            "Use this to highlight specific risk factors in your analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["ticker"]
        }
    }
]


# ── Tool execution ─────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool call from the agent and return the result as a JSON string.
    This is the bridge between Claude's reasoning and your analysis engine.
    """
    try:
        if tool_name == "get_financial_metrics":
            ticker = tool_input["ticker"].upper()
            metrics = analyze_ticker(ticker, DB_PATH)
            
            # Build a clean, well-structured dict for Claude to reason over
            result = {
                "ticker": metrics.ticker,
                "latest_financials_billions": {
                    "revenue": round(metrics.latest_revenue / 1e9, 2) if metrics.latest_revenue else None,
                    "net_income": round(metrics.latest_net_income / 1e9, 2) if metrics.latest_net_income else None,
                    "free_cash_flow": round(metrics.latest_fcf / 1e9, 2) if metrics.latest_fcf else None,
                    "total_debt": round(metrics.latest_total_debt / 1e9, 2) if metrics.latest_total_debt else None,
                    "cash": round(metrics.latest_cash / 1e9, 2) if metrics.latest_cash else None,
                    "total_equity": round(metrics.latest_total_equity / 1e9, 2) if metrics.latest_total_equity else None,
                    "eps": metrics.latest_eps,
                },
                "profitability": {
                    "net_margin_pct": round(metrics.net_margin * 100, 2) if metrics.net_margin else None,
                    "gross_margin_pct": round(metrics.gross_margin * 100, 2) if metrics.gross_margin else None,
                    "operating_margin_pct": round(metrics.operating_margin * 100, 2) if metrics.operating_margin else None,
                    "ebitda_margin_pct": round(metrics.ebitda_margin * 100, 2) if metrics.ebitda_margin else None,
                    "roe_pct": round(metrics.roe * 100, 2) if metrics.roe else None,
                    "roa_pct": round(metrics.roa * 100, 2) if metrics.roa else None,
                },
                "solvency": {
                    "debt_to_equity": round(metrics.debt_to_equity, 2) if metrics.debt_to_equity else None,
                    "debt_to_assets": round(metrics.debt_to_assets, 2) if metrics.debt_to_assets else None,
                    "interest_coverage_x": round(metrics.interest_coverage, 1) if metrics.interest_coverage else None,
                    "current_ratio": round(metrics.current_ratio, 2) if metrics.current_ratio else None,
                },
                "cash_flow_quality": {
                    "fcf_to_net_income": round(metrics.fcf_to_net_income, 2) if metrics.fcf_to_net_income else None,
                    "note": (
                        "FCF/NI > 1.0 indicates high earnings quality. "
                        "FCF/NI < 0.7 may indicate aggressive accounting."
                    )
                },
                "growth_yoy": {
                    "revenue_pct": round(metrics.revenue_growth_yoy * 100, 1) if metrics.revenue_growth_yoy else None,
                    "net_income_pct": round(metrics.net_income_growth_yoy * 100, 1) if metrics.net_income_growth_yoy else None,
                    "fcf_pct": round(metrics.fcf_growth_yoy * 100, 1) if metrics.fcf_growth_yoy else None,
                },
                "trends": {
                    "margin_expansion_pp": round(metrics.margin_expansion, 2) if metrics.margin_expansion else None,
                    "debt_change_yoy_pct": round(metrics.debt_change_pct * 100, 1) if metrics.debt_change_pct else None,
                },
            }
            return json.dumps(result, indent=2)

        elif tool_name == "get_price_history":
            ticker = tool_input["ticker"].upper()
            days = tool_input.get("days", 90)

            conn = init_db(DB_PATH)
            prices = get_recent_prices(conn, ticker, days=days)
            conn.close()

            if not prices:
                return json.dumps({"error": f"No price data found for {ticker}"})

            closes = [p["close"] for p in prices if p["close"]]
            result = {
                "ticker": ticker,
                "trading_days_returned": len(prices),
                "latest_close": closes[0] if closes else None,
                "52_week_high": max(closes) if closes else None,
                "52_week_low": min(closes) if closes else None,
                "pct_from_52w_high": (
                    round((closes[0] - max(closes)) / max(closes) * 100, 1)
                    if closes else None
                ),
                "avg_volume_recent": (
                    int(sum(p["volume"] for p in prices[:20] if p["volume"]) / 20)
                    if len(prices) >= 20 else None
                ),
                "recent_prices_sample": [
                    {"date": p["date"], "close": p["close"]}
                    for p in prices[:5]
                ],
            }
            return json.dumps(result, indent=2)

        elif tool_name == "compare_companies":
            tickers = [t.upper() for t in tool_input["tickers"]]
            df = analyze_tickers(tickers, DB_PATH)

            comparison = []
            for _, row in df.iterrows():
                comparison.append({
                    "ticker": row["ticker"],
                    "revenue_B": row.get("revenue_B"),
                    "net_margin_%": row.get("net_margin_%"),
                    "gross_margin_%": row.get("gross_margin_%"),
                    "roe_%": row.get("roe_%"),
                    "debt_to_equity": row.get("debt_to_equity"),
                    "current_ratio": row.get("current_ratio"),
                    "revenue_growth_%": row.get("revenue_growth_yoy_%"),
                    "fcf_to_ni": row.get("fcf_to_ni"),
                    "flags": {
                        "fcf_divergence": bool(row.get("flag_fcf_divergence")),
                        "earnings_miss": bool(row.get("flag_earnings_miss")),
                        "margin_compression": bool(row.get("flag_margin_compression")),
                        "debt_spike": bool(row.get("flag_debt_spike")),
                    }
                })
            return json.dumps(comparison, indent=2)

        elif tool_name == "get_anomaly_flags":
            ticker = tool_input["ticker"].upper()
            metrics = analyze_ticker(ticker, DB_PATH)

            flags = []
            if metrics.fcf_divergence:
                flags.append({
                    "flag": "FCF_DIVERGENCE",
                    "severity": "MEDIUM",
                    "detail": (
                        f"Free cash flow (${metrics.latest_fcf/1e9:.1f}B) diverges >25% from "
                        f"net income (${metrics.latest_net_income/1e9:.1f}B). "
                        "This can indicate aggressive revenue recognition or non-cash earnings."
                    )
                })
            if metrics.earnings_miss:
                flags.append({
                    "flag": "EARNINGS_MISS",
                    "severity": "HIGH",
                    "detail": (
                        f"Net income fell significantly despite revenue growth. "
                        f"Revenue growth: {metrics.revenue_growth_yoy*100:.1f}%, "
                        f"Net income growth: {metrics.net_income_growth_yoy*100:.1f}%."
                    )
                })
            if metrics.margin_compression:
                flags.append({
                    "flag": "MARGIN_COMPRESSION",
                    "severity": "MEDIUM",
                    "detail": (
                        f"Net margin contracted {abs(metrics.margin_expansion):.1f} percentage "
                        "points year-over-year. May indicate rising costs or pricing pressure."
                    )
                })
            if metrics.debt_spike:
                flags.append({
                    "flag": "DEBT_SPIKE",
                    "severity": "MEDIUM",
                    "detail": (
                        f"Total debt increased {metrics.debt_change_pct*100:.1f}% YoY "
                        f"to ${metrics.latest_total_debt/1e9:.1f}B. "
                        "Elevated leverage warrants monitoring of interest coverage."
                    )
                })
            if not flags:
                flags.append({
                    "flag": "NONE",
                    "severity": "LOW",
                    "detail": "No major anomalies detected in recent financial data."
                })

            return json.dumps({"ticker": ticker, "anomalies": flags}, indent=2)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        log.error(f"Tool {tool_name} failed: {e}")
        return json.dumps({"error": str(e)})


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior financial analyst with expertise in equity research and fundamental analysis. Your role is to analyze companies using quantitative financial data and produce structured, actionable investment analysis.

When analyzing a company, you MUST:
1. Call get_financial_metrics to retrieve all key ratios and trends
2. Call get_anomaly_flags to check for any red flags
3. Call get_price_history to understand recent price context
4. If comparing companies, call compare_companies and get_financial_metrics for each

Your analysis must be grounded entirely in the data returned by your tools. Do not invent numbers.

After gathering data, produce a structured JSON report with this exact schema:
{
  "ticker": "AAPL",
  "analysis_date": "2025-01-15",
  "executive_summary": "2-3 sentence overview of the company's financial health",
  "strengths": [
    {"point": "specific strength", "supporting_data": "metric that proves it"}
  ],
  "risks": [
    {"point": "specific risk", "supporting_data": "metric that proves it"}
  ],
  "anomalies": [
    {"flag": "FLAG_NAME", "severity": "HIGH|MEDIUM|LOW", "interpretation": "what it means"}
  ],
  "recommendation": {
    "stance": "BULLISH | CAUTIOUS | BEARISH",
    "confidence": "HIGH | MEDIUM | LOW",
    "rationale": "2-3 sentences tying stance to specific data points",
    "key_metrics_to_watch": ["metric 1", "metric 2"]
  },
  "disclaimer": "This analysis is generated by an AI system for informational purposes only and does not constitute financial advice. Always conduct your own due diligence."
}

Confidence levels:
- HIGH: Multiple strong signals pointing the same direction, no major anomalies
- MEDIUM: Mixed signals or one significant anomaly present  
- LOW: Contradictory signals or multiple anomalies — more research needed

Be specific. Reference actual numbers from the tool results. Avoid vague language like "the company is doing well." Instead: "Net margin of 25.3% is significantly above the S&P 500 median of ~10%."

Return ONLY the JSON object. No preamble, no markdown, no explanation outside the JSON."""


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(ticker: str, compare_with: list[str] = None) -> dict:
    """
    Run the financial analysis agent for a ticker.
    
    Args:
        ticker: Primary ticker to analyze (e.g. "AAPL")
        compare_with: Optional list of tickers to compare against (e.g. ["MSFT"])
    
    Returns:
        Structured analysis dict
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build the initial user message
    if compare_with:
        all_tickers = [ticker] + compare_with
        user_msg = (
            f"Analyze {ticker} and compare it against {', '.join(compare_with)}. "
            f"Identify which company has stronger fundamentals and explain why. "
            f"Focus on {ticker} as the primary subject of the recommendation."
        )
    else:
        user_msg = (
            f"Analyze {ticker}. Use your tools to gather all relevant financial data, "
            f"identify strengths and risks, check for anomalies, "
            f"and produce a structured investment analysis."
        )

    messages = [{"role": "user", "content": user_msg}]
    log.info(f"Starting agent for {ticker}" + (f" vs {compare_with}" if compare_with else ""))

    # ── Agentic loop ───────────────────────────────────────────────────────────
    # Claude will call tools iteratively until it has enough data to write the report
    max_iterations = 10  # safety cap
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        log.info(f"Agent iteration {iteration}...")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        log.info(f"Stop reason: {response.stop_reason}")

        # Append assistant's response to conversation
        messages.append({"role": "assistant", "content": response.content})

        # If Claude is done (no more tool calls), extract the final answer
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text") and block.text.strip():
                    try:
                        text = block.text.strip()

                        # Strategy 1: find the first { and last } and parse between them.
                        # This handles preamble text, markdown fences, and trailing content.
                        start = text.find("{")
                        end   = text.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            return json.loads(text[start:end + 1])

                        # Strategy 2: strip markdown fences if strategy 1 somehow failed
                        if "```" in text:
                            inner = text.split("```")[1]
                            if inner.startswith("json"):
                                inner = inner[4:]
                            return json.loads(inner.strip())

                        # Strategy 3: try the raw text as-is
                        return json.loads(text)

                    except json.JSONDecodeError as e:
                        log.error(f"Failed to parse agent JSON output: {e}")
                        log.error(f"Raw output: {block.text}")
                        return {"error": "Failed to parse agent output", "raw": block.text}
            break

        # If Claude wants to use tools, execute them all and feed results back
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    log.info(f"Tool call: {block.name}({json.dumps(block.input)})")
                    result = execute_tool(block.name, block.input)
                    log.info(f"Tool result preview: {result[:120]}...")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Feed all tool results back to Claude in one message
            messages.append({"role": "user", "content": tool_results})

    log.warning("Agent hit max iterations without finishing")
    return {"error": "Agent did not complete within iteration limit"}


# ── CLI ────────────────────────────────────────────────────────────────────────

def print_report(report: dict):
    """Pretty-print the structured analysis report."""
    if "error" in report:
        print(f"\n❌ Error: {report['error']}")
        if "raw" in report:
            print(f"Raw output:\n{report['raw']}")
        return

    print(f"\n{'═'*65}")
    print(f"  📊 Financial Analysis: {report.get('ticker', 'N/A')}")
    print(f"  {report.get('analysis_date', '')}")
    print(f"{'═'*65}")

    print(f"\n  EXECUTIVE SUMMARY")
    print(f"  {report.get('executive_summary', 'N/A')}")

    print(f"\n  ✅ STRENGTHS")
    for s in report.get("strengths", []):
        print(f"    • {s['point']}")
        print(f"      Data: {s['supporting_data']}")

    print(f"\n  ⚠️  RISKS")
    for r in report.get("risks", []):
        print(f"    • {r['point']}")
        print(f"      Data: {r['supporting_data']}")

    anomalies = report.get("anomalies", [])
    if anomalies:
        print(f"\n  🚩 ANOMALIES")
        for a in anomalies:
            severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(a.get("severity", ""), "•")
            print(f"    {severity_emoji} [{a.get('severity')}] {a.get('flag')}")
            print(f"      {a.get('interpretation')}")

    rec = report.get("recommendation", {})
    stance_emoji = {"BULLISH": "📈", "CAUTIOUS": "➡️", "BEARISH": "📉"}.get(rec.get("stance", ""), "•")
    print(f"\n  {stance_emoji} RECOMMENDATION: {rec.get('stance')} (Confidence: {rec.get('confidence')})")
    print(f"  {rec.get('rationale', '')}")

    metrics_to_watch = rec.get("key_metrics_to_watch", [])
    if metrics_to_watch:
        print(f"\n  👁  WATCH: {', '.join(metrics_to_watch)}")

    print(f"\n  ⚖️  {report.get('disclaimer', '')}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the financial analysis agent for one or more tickers"
    )
    parser.add_argument("--ticker", required=True,
                        help="Primary ticker to analyze (e.g. AAPL)")
    parser.add_argument("--vs", nargs="*",
                        help="Optional tickers to compare against (e.g. --vs MSFT GOOGL)")
    parser.add_argument("--db", default=str(DB_PATH),
                        help=f"Path to SQLite database (default: {DB_PATH})")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted report")
    args = parser.parse_args()

    DB_PATH = Path(args.db)

    report = run_agent(args.ticker.upper(), compare_with=args.vs)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)