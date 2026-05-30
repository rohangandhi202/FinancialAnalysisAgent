"""
financial_agent/report_generator.py

Generates a polished, self-contained HTML report from:
  - The agent's JSON analysis output
  - Raw financial data from SQLite (for charts)

Charts are embedded as base64 PNGs — no external files needed.
Output is a single .html file you can open in any browser or email.

Usage:
    python -m financial_agent.report_generator --ticker AAPL
    python -m financial_agent.report_generator --ticker AAPL --vs MSFT
    python -m financial_agent.report_generator --ticker AAPL --out reports/aapl.html
"""

import argparse
import base64
import io
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI window
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from dotenv import load_dotenv

load_dotenv()

from financial_agent.agent import run_agent
from financial_agent.financial_analysis import analyze_ticker
from financial_agent.data_ingestion import init_db, get_recent_prices, get_annual_financials

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = Path("financial_data.db")

# ── Chart theme ────────────────────────────────────────────────────────────────

DARK_BG     = "#0a0e1a"
CARD_BG     = "#111827"
BORDER      = "#1f2937"
AMBER       = "#f59e0b"
EMERALD     = "#10b981"
ROSE        = "#f43f5e"
SKY         = "#38bdf8"
TEXT_PRI    = "#f9fafb"
TEXT_SEC    = "#9ca3af"

def _apply_chart_style(fig, ax):
    """Apply consistent dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_SEC, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SEC)
    ax.yaxis.label.set_color(TEXT_SEC)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.grid(axis="x", visible=False)
    return fig, ax


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Chart generators ───────────────────────────────────────────────────────────

def chart_price_history(ticker: str, conn: sqlite3.Connection) -> str | None:
    """90-day price history line chart."""
    prices = get_recent_prices(conn, ticker, days=90)
    if not prices:
        return None

    dates  = [p["date"] for p in reversed(prices)]
    closes = [p["close"] for p in reversed(prices)]

    fig, ax = plt.subplots(figsize=(7, 3))
    _apply_chart_style(fig, ax)

    ax.plot(dates, closes, color=AMBER, linewidth=1.8, zorder=3)
    ax.fill_between(range(len(closes)), closes,
                    min(closes) * 0.995,
                    alpha=0.15, color=AMBER, zorder=2)

    # Only show every ~15th date label to avoid crowding
    step = max(1, len(dates) // 6)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                       rotation=25, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))
    ax.set_title(f"{ticker} — 90-Day Price", color=TEXT_PRI,
                 fontsize=11, pad=10, loc="left")

    return _fig_to_base64(fig)


def chart_revenue_income(ticker: str, conn: sqlite3.Connection) -> str | None:
    """Annual revenue vs net income grouped bar chart."""
    rows = get_annual_financials(conn, ticker, years=5)
    if not rows:
        return None

    rows = list(reversed(rows))  # oldest → newest
    years   = [r["fiscal_year"][:4] for r in rows]
    revenue = [(r["total_revenue"] or 0) / 1e9 for r in rows]
    income  = [(r["net_income"] or 0) / 1e9 for r in rows]

    x = range(len(years))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7, 3))
    _apply_chart_style(fig, ax)

    ax.bar([i - w/2 for i in x], revenue, width=w, color=SKY,
           alpha=0.85, label="Revenue", zorder=3)
    ax.bar([i + w/2 for i in x], income, width=w, color=EMERALD,
           alpha=0.85, label="Net Income", zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(years, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:.0f}B"
    ))
    ax.set_title(f"{ticker} — Revenue & Net Income (Annual)",
                 color=TEXT_PRI, fontsize=11, pad=10, loc="left")

    legend = ax.legend(facecolor=CARD_BG, edgecolor=BORDER,
                       labelcolor=TEXT_SEC, fontsize=8)
    return _fig_to_base64(fig)


def chart_margins(ticker: str, conn: sqlite3.Connection) -> str | None:
    """Gross / operating / net margin trend over annual periods."""
    rows = get_annual_financials(conn, ticker, years=5)
    if not rows or len(rows) < 2:
        return None

    rows = list(reversed(rows))
    years = [r["fiscal_year"][:4] for r in rows]

    def pct(num, denom):
        if num and denom and denom != 0:
            return round(num / denom * 100, 1)
        return None

    gross_m = [pct(r["gross_profit"],      r["total_revenue"]) for r in rows]
    oper_m  = [pct(r["operating_income"],  r["total_revenue"]) for r in rows]
    net_m   = [pct(r["net_income"],        r["total_revenue"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 3))
    _apply_chart_style(fig, ax)

    def plot_series(values, color, label):
        clean = [(i, v) for i, v in enumerate(values) if v is not None]
        if clean:
            xs, ys = zip(*clean)
            ax.plot(xs, ys, color=color, linewidth=1.8,
                    marker="o", markersize=4, label=label, zorder=3)

    plot_series(gross_m, SKY,     "Gross Margin")
    plot_series(oper_m,  AMBER,   "Operating Margin")
    plot_series(net_m,   EMERALD, "Net Margin")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}%"
    ))
    ax.set_title(f"{ticker} — Margin Trends (Annual)",
                 color=TEXT_PRI, fontsize=11, pad=10, loc="left")
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER,
              labelcolor=TEXT_SEC, fontsize=8)

    return _fig_to_base64(fig)


def chart_fcf(ticker: str, conn: sqlite3.Connection) -> str | None:
    """Free cash flow vs net income comparison bar chart."""
    rows = get_annual_financials(conn, ticker, years=5)
    if not rows:
        return None

    rows = list(reversed(rows))
    years  = [r["fiscal_year"][:4] for r in rows]
    fcf    = [(r["free_cashflow"] or 0) / 1e9 for r in rows]
    income = [(r["net_income"] or 0) / 1e9 for r in rows]

    x = range(len(years))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7, 3))
    _apply_chart_style(fig, ax)

    ax.bar([i - w/2 for i in x], fcf,    width=w, color=AMBER,
           alpha=0.85, label="Free Cash Flow", zorder=3)
    ax.bar([i + w/2 for i in x], income, width=w, color=SKY,
           alpha=0.85, label="Net Income", zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(years, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:.0f}B"
    ))
    ax.set_title(f"{ticker} — Free Cash Flow vs Net Income",
                 color=TEXT_PRI, fontsize=11, pad=10, loc="left")
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER,
              labelcolor=TEXT_SEC, fontsize=8)

    return _fig_to_base64(fig)


# ── HTML builder ───────────────────────────────────────────────────────────────

def _fmt(val, suffix="", prefix="", decimals=1, na="—"):
    """Format a number cleanly, returning na string if None."""
    if val is None:
        return na
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def _stance_color(stance: str) -> str:
    return {"BULLISH": EMERALD, "BEARISH": ROSE}.get(stance.upper(), AMBER)


def _severity_color(severity: str) -> str:
    return {"HIGH": ROSE, "MEDIUM": AMBER, "LOW": EMERALD}.get(
        severity.upper(), AMBER)


def build_html(ticker: str, report: dict, metrics,
               img_price: str, img_rev: str,
               img_margins: str, img_fcf: str) -> str:
    """
    Assemble the full self-contained HTML report string.
    All charts are embedded as base64 data URIs.
    """
    rec       = report.get("recommendation", {})
    stance    = rec.get("stance", "CAUTIOUS")
    conf      = rec.get("confidence", "MEDIUM")
    sc        = _stance_color(stance)
    today     = datetime.now().strftime("%B %d, %Y")
    anomalies = report.get("anomalies", [])

    def metric_card(label, value, sub=""):
        return f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
        </div>"""

    def chart_block(b64, alt):
        if not b64:
            return ""
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" class="chart-img">'

    strengths_html = "".join(
        f"""<div class="finding-item">
              <div class="finding-title">{s.get('point','')}</div>
              <div class="finding-data">{s.get('supporting_data','')}</div>
           </div>"""
        for s in report.get("strengths", [])
    )

    risks_html = "".join(
        f"""<div class="finding-item risk">
              <div class="finding-title">{r.get('point','')}</div>
              <div class="finding-data">{r.get('supporting_data','')}</div>
           </div>"""
        for r in report.get("risks", [])
    )

    anomaly_html = ""
    for a in anomalies:
        color = _severity_color(a.get("severity", "MEDIUM"))
        anomaly_html += f"""
        <div class="anomaly-item" style="border-left-color:{color}">
          <span class="anomaly-badge" style="background:{color}20;color:{color}">
            {a.get('severity','—')}
          </span>
          <span class="anomaly-flag">{a.get('flag','—')}</span>
          <p class="anomaly-detail">{a.get('interpretation','')}</p>
        </div>"""

    if not anomaly_html:
        anomaly_html = '<p class="no-anomaly">✓ No significant anomalies detected</p>'

    watch_pills = "".join(
        f'<span class="pill">{w}</span>'
        for w in rec.get("key_metrics_to_watch", [])
    )

    m = metrics  # shorthand

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} — Financial Analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       {DARK_BG};
    --card:     {CARD_BG};
    --border:   {BORDER};
    --amber:    {AMBER};
    --emerald:  {EMERALD};
    --rose:     {ROSE};
    --sky:      {SKY};
    --text:     {TEXT_PRI};
    --muted:    {TEXT_SEC};
    --stance:   {sc};
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    padding: 2rem;
    max-width: 1100px;
    margin: 0 auto;
  }}

  /* ── Header ── */
  .header {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }}
  .header-left {{ display: flex; flex-direction: column; gap: 0.4rem; }}
  .label-tag {{
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--amber);
    text-transform: uppercase;
  }}
  .ticker-name {{
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
  }}
  .report-date {{ color: var(--muted); font-size: 11px; margin-top: 0.2rem; }}
  .stance-badge {{
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.5rem 1.2rem;
    border: 1.5px solid var(--stance);
    color: var(--stance);
    border-radius: 4px;
    background: color-mix(in srgb, var(--stance) 10%, transparent);
  }}
  .confidence {{ font-size: 10px; color: var(--muted); text-align: right; margin-top: 4px; }}

  /* ── Summary ── */
  .executive-summary {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    padding: 1.25rem 1.5rem;
    margin-bottom: 2rem;
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    line-height: 1.75;
    color: #e5e7eb;
  }}

  /* ── Metrics grid ── */
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin-bottom: 2rem;
  }}
  .metric-card {{
    background: var(--card);
    padding: 1rem 1.1rem;
  }}
  .metric-label {{
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }}
  .metric-value {{
    font-size: 1.45rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: -0.02em;
  }}
  .metric-sub {{ font-size: 10px; color: var(--muted); margin-top: 3px; }}

  /* ── Charts ── */
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    padding: 0.25rem;
  }}
  .chart-img {{ width: 100%; display: block; }}

  /* ── Findings ── */
  .findings-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .findings-col {{ display: flex; flex-direction: column; gap: 0; }}
  .findings-header {{
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.75rem;
  }}
  .finding-item {{
    border-left: 2px solid var(--emerald);
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.6rem;
    background: color-mix(in srgb, var(--emerald) 5%, var(--card));
  }}
  .finding-item.risk {{ border-left-color: var(--rose); background: color-mix(in srgb, var(--rose) 5%, var(--card)); }}
  .finding-title {{ font-weight: 500; color: var(--text); margin-bottom: 3px; }}
  .finding-data {{ font-size: 11px; color: var(--muted); }}

  /* ── Anomalies ── */
  .section-header {{
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
  }}
  .anomalies-list {{ display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 2rem; }}
  .anomaly-item {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    padding: 0.8rem 1rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem;
  }}
  .anomaly-badge {{
    font-size: 9px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 2px;
    font-weight: 500;
  }}
  .anomaly-flag {{ font-weight: 500; flex: 1; }}
  .anomaly-detail {{ width: 100%; font-size: 11px; color: var(--muted); margin-top: 4px; }}
  .no-anomaly {{ color: var(--emerald); font-size: 12px; }}

  /* ── Recommendation ── */
  .recommendation {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--stance);
    padding: 1.25rem 1.5rem;
    margin-bottom: 2rem;
  }}
  .rec-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
  }}
  .rec-stance {{
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: var(--stance);
    font-weight: 700;
  }}
  .rec-conf {{
    font-size: 10px;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 2px;
  }}
  .rec-rationale {{ color: #d1d5db; font-size: 13px; line-height: 1.7; }}
  .watch-section {{ margin-top: 1rem; }}
  .watch-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
  .pill {{
    font-size: 10px;
    padding: 3px 10px;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 2px;
  }}

  /* ── Footer ── */
  .disclaimer {{
    font-size: 10px;
    color: #4b5563;
    border-top: 1px solid var(--border);
    padding-top: 1rem;
    line-height: 1.6;
  }}
  .footer-brand {{
    font-size: 10px;
    color: #374151;
    margin-top: 0.5rem;
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <span class="label-tag">Autonomous Financial Analysis Report</span>
    <div class="ticker-name">{ticker}</div>
    <div class="report-date">Generated {today}</div>
  </div>
  <div style="text-align:right">
    <div class="stance-badge">{stance}</div>
    <div class="confidence">Confidence: {conf}</div>
  </div>
</div>

<!-- Executive Summary -->
<p class="executive-summary">{report.get('executive_summary', '')}</p>

<!-- Key Metrics Grid -->
<div class="metrics-grid">
  {metric_card("Revenue", _fmt(m.latest_revenue, prefix="$", suffix="B", decimals=1,
                               na="—") if m.latest_revenue is None else
               f"${m.latest_revenue/1e9:,.1f}B")}
  {metric_card("Net Income", f"${m.latest_net_income/1e9:,.1f}B"
               if m.latest_net_income else "—")}
  {metric_card("Free Cash Flow", f"${m.latest_fcf/1e9:,.1f}B"
               if m.latest_fcf else "—")}
  {metric_card("Net Margin", f"{m.net_margin*100:.1f}%"
               if m.net_margin else "—",
               sub=f"{'▲' if (m.margin_expansion or 0) > 0 else '▼'} {abs(m.margin_expansion or 0):.1f}pp YoY"
               if m.margin_expansion else "")}
  {metric_card("Debt / Equity", f"{m.debt_to_equity:.2f}x"
               if m.debt_to_equity else "—")}
  {metric_card("Interest Cov.", f"{m.interest_coverage:.1f}x"
               if m.interest_coverage else "—")}
  {metric_card("Current Ratio", f"{m.current_ratio:.2f}x"
               if m.current_ratio else "—")}
  {metric_card("FCF / NI", f"{m.fcf_to_net_income:.2f}x"
               if m.fcf_to_net_income else "—",
               sub="Earnings quality")}
  {metric_card("Revenue Growth", f"{m.revenue_growth_yoy*100:+.1f}%"
               if m.revenue_growth_yoy else "—", sub="YoY")}
  {metric_card("ROE", f"{m.roe*100:.1f}%"
               if m.roe else "—")}
  {metric_card("ROA", f"{m.roa*100:.1f}%"
               if m.roa else "—")}
  {metric_card("Total Debt", f"${m.latest_total_debt/1e9:,.1f}B"
               if m.latest_total_debt else "—")}
</div>

<!-- Charts -->
<div class="charts-grid">
  <div class="chart-card">{chart_block(img_price, "Price History")}</div>
  <div class="chart-card">{chart_block(img_rev, "Revenue and Net Income")}</div>
  <div class="chart-card">{chart_block(img_margins, "Margin Trends")}</div>
  <div class="chart-card">{chart_block(img_fcf, "Free Cash Flow vs Net Income")}</div>
</div>

<!-- Strengths & Risks -->
<div class="findings-grid">
  <div class="findings-col">
    <div class="findings-header">Strengths</div>
    {strengths_html or '<p style="color:var(--muted);font-size:12px">None identified</p>'}
  </div>
  <div class="findings-col">
    <div class="findings-header">Risks</div>
    {risks_html or '<p style="color:var(--muted);font-size:12px">None identified</p>'}
  </div>
</div>

<!-- Anomalies -->
<div class="section-header">Anomaly Flags</div>
<div class="anomalies-list">{anomaly_html}</div>

<!-- Recommendation -->
<div class="recommendation">
  <div class="rec-header">
    <span class="rec-stance">{stance}</span>
    <span class="rec-conf">Confidence: {conf}</span>
  </div>
  <p class="rec-rationale">{rec.get('rationale', '')}</p>
  {f'<div class="watch-section"><div class="watch-label">Key metrics to watch</div><div class="pills">{watch_pills}</div></div>' if watch_pills else ''}
</div>

<!-- Disclaimer -->
<p class="disclaimer">{report.get('disclaimer', 'For informational purposes only. Not financial advice.')}</p>
<p class="footer-brand">Generated by Autonomous Financial Analysis Agent · Claude claude-sonnet-4-6 · Alpha Vantage</p>

</body>
</html>"""


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_report(ticker: str, compare_with: list = None,
                    out_path: Path = None) -> Path:
    """
    Full pipeline: run agent → fetch chart data → build HTML → write file.
    Returns the path to the generated HTML file.
    """
    ticker = ticker.upper()
    log.info(f"Running agent for {ticker}...")
    report = run_agent(ticker, compare_with=compare_with)

    if "error" in report:
        raise RuntimeError(f"Agent failed: {report['error']}")

    log.info("Loading metrics and generating charts...")
    conn = init_db(DB_PATH)
    conn.row_factory = sqlite3.Row
    metrics = analyze_ticker(ticker, DB_PATH)

    img_price   = chart_price_history(ticker, conn)
    img_rev     = chart_revenue_income(ticker, conn)
    img_margins = chart_margins(ticker, conn)
    img_fcf     = chart_fcf(ticker, conn)
    conn.close()

    log.info("Building HTML report...")
    html = build_html(ticker, report, metrics,
                      img_price, img_rev, img_margins, img_fcf)

    # Determine output path
    if out_path is None:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        out_path = reports_dir / f"{ticker}_{date_str}.html"

    out_path.write_text(html, encoding="utf-8")
    log.info(f"Report written to: {out_path.resolve()}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an HTML financial analysis report"
    )
    parser.add_argument("--ticker", required=True,
                        help="Primary ticker to analyze (e.g. AAPL)")
    parser.add_argument("--vs", nargs="*",
                        help="Optional comparison tickers (e.g. --vs MSFT)")
    parser.add_argument("--out", type=str,
                        help="Output file path (default: reports/<TICKER>_<DATE>.html)")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    DB_PATH = Path(args.db)
    out = Path(args.out) if args.out else None

    report_path = generate_report(args.ticker, compare_with=args.vs, out_path=out)
    print(f"\n✅ Report ready: {report_path.resolve()}")
    print("   Open in your browser to view.\n")