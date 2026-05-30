"""
financial_agent/financial_analysis.py

Reads financial statements from SQLite and computes:
- Key ratios (P/E, P/B, EV/EBITDA, D/E, current ratio, FCF yield, etc.)
- Trend metrics (YoY growth, margin expansion/contraction, debt trajectory)
- Anomaly flags (earnings misses, sudden margin drops, FCF divergence)

Output: a clean pandas DataFrame ready for Claude API reasoning.

Usage:
    python -m financial_agent.financial_analysis --ticker AAPL
    python -m financial_agent.financial_analysis --ticker AAPL MSFT
"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = Path("financial_data.db")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class FinancialMetrics:
    """Container for computed metrics for a single company."""
    ticker: str
    
    # Latest annual metrics
    latest_revenue: Optional[float] = None
    latest_net_income: Optional[float] = None
    latest_fcf: Optional[float] = None
    latest_total_debt: Optional[float] = None
    latest_total_equity: Optional[float] = None
    latest_eps: Optional[float] = None
    latest_cash: Optional[float] = None
    latest_assets: Optional[float] = None
    latest_current_ratio: Optional[float] = None
    
    # Ratios (profitability)
    net_margin: Optional[float] = None       # net_income / revenue
    gross_margin: Optional[float] = None     # gross_profit / revenue
    operating_margin: Optional[float] = None # operating_income / revenue
    roe: Optional[float] = None              # net_income / total_equity
    roa: Optional[float] = None              # net_income / total_assets
    
    # Ratios (solvency)
    debt_to_equity: Optional[float] = None   # total_debt / total_equity
    debt_to_assets: Optional[float] = None   # total_debt / total_assets
    interest_coverage: Optional[float] = None # ebit / interest_expense
    
    # Ratios (liquidity)
    current_ratio: Optional[float] = None    # current_assets / current_liabilities
    
    # Cash flow
    fcf_yield: Optional[float] = None        # fcf / market_cap (requires price)
    fcf_to_net_income: Optional[float] = None # fcf / net_income
    operating_cf_to_net_income: Optional[float] = None
    
    # Trends (YoY change)
    revenue_growth_yoy: Optional[float] = None     # (latest - prior) / prior
    net_income_growth_yoy: Optional[float] = None
    fcf_growth_yoy: Optional[float] = None
    
    # Margin trends
    margin_expansion: Optional[float] = None # net_margin[latest] - net_margin[prior]
    ebitda_margin: Optional[float] = None    # ebitda / revenue
    
    # Debt trajectory
    debt_change_pct: Optional[float] = None  # (debt[latest] - debt[prior]) / debt[prior]
    
    # Anomaly flags
    fcf_divergence: bool = False             # |fcf - net_income| > 25% of net_income
    earnings_miss: bool = False              # net_income declined 20%+ YoY with revenue up
    margin_compression: bool = False         # net margin fell 3+ percentage points
    debt_spike: bool = False                 # debt grew 20%+ YoY
    


# ── Calculation helpers ────────────────────────────────────────────────────────

def safe_div(num: float, denom: float) -> Optional[float]:
    """Safe division — returns None instead of ZeroDivisionError."""
    if denom is None or denom == 0:
        return None
    if num is None:
        return None
    return num / denom


def yoy_change(latest: float, prior: float) -> Optional[float]:
    """Year-over-year percent change. Returns None if can't compute."""
    if prior is None or prior == 0:
        return None
    if latest is None:
        return None
    return (latest - prior) / abs(prior)


def compute_metrics(ticker: str, conn: sqlite3.Connection) -> FinancialMetrics:
    """
    Compute all financial metrics for a ticker from its stored financials.
    Returns a FinancialMetrics object.
    """
    ticker = ticker.upper()
    m = FinancialMetrics(ticker=ticker)
    
    # Fetch annual financials (latest 5 years)
    financials = conn.execute("""
        SELECT
            i.fiscal_year,
            i.total_revenue,
            i.gross_profit,
            i.net_income,
            i.ebitda,
            i.operating_income,
            i.interest_expense,
            i.eps,
            b.total_assets,
            b.total_equity,
            b.total_debt,
            b.cash_and_equivalents,
            b.current_assets,
            b.current_liabilities,
            c.operating_cashflow,
            c.free_cashflow
        FROM income_statements i
        LEFT JOIN balance_sheets b
            ON b.ticker = i.ticker
            AND b.fiscal_year = i.fiscal_year
            AND b.report_type = 'annual'
        LEFT JOIN cash_flows c
            ON c.ticker = i.ticker
            AND c.fiscal_year = i.fiscal_year
            AND c.report_type = 'annual'
        WHERE i.ticker = ?
          AND i.report_type = 'annual'
        ORDER BY i.fiscal_year DESC
        LIMIT 5
    """, (ticker,)).fetchall()
    
    if not financials:
        log.warning(f"No annual financials found for {ticker}")
        return m
    
    # Convert to list of dicts for easier access
    financials = [dict(row) for row in financials]
    latest = financials[0]
    prior = financials[1] if len(financials) > 1 else None
    
    # ── Latest annual values ─────────────────────────────────────────────────
    m.latest_revenue = latest.get("total_revenue")
    m.latest_net_income = latest.get("net_income")
    m.latest_fcf = latest.get("free_cashflow")
    m.latest_total_debt = latest.get("total_debt")
    m.latest_total_equity = latest.get("total_equity")
    m.latest_eps = latest.get("eps")
    m.latest_cash = latest.get("cash_and_equivalents")
    m.latest_assets = latest.get("total_assets")
    
    # ── Profitability ratios ─────────────────────────────────────────────────
    m.net_margin = safe_div(latest.get("net_income"), latest.get("total_revenue"))
    m.gross_margin = safe_div(latest.get("gross_profit"), latest.get("total_revenue"))
    m.operating_margin = safe_div(latest.get("operating_income"), latest.get("total_revenue"))
    m.ebitda_margin = safe_div(latest.get("ebitda"), latest.get("total_revenue"))
    m.roe = safe_div(latest.get("net_income"), latest.get("total_equity"))
    m.roa = safe_div(latest.get("net_income"), latest.get("total_assets"))
    
    # ── Solvency ratios ──────────────────────────────────────────────────────
    m.debt_to_equity = safe_div(latest.get("total_debt"), latest.get("total_equity"))
    m.debt_to_assets = safe_div(latest.get("total_debt"), latest.get("total_assets"))
    m.interest_coverage = safe_div(
        latest.get("ebit") or latest.get("operating_income"),
        latest.get("interest_expense")
    )
    
    # ── Liquidity ratios ─────────────────────────────────────────────────────
    m.current_ratio = safe_div(
        latest.get("current_assets"),
        latest.get("current_liabilities")
    )
    
    # ── Cash flow analysis ───────────────────────────────────────────────────
    m.fcf_to_net_income = safe_div(
        latest.get("free_cashflow"),
        latest.get("net_income")
    )
    m.operating_cf_to_net_income = safe_div(
        latest.get("operating_cashflow"),
        latest.get("net_income")
    )
    
    # ── Trends (YoY) ─────────────────────────────────────────────────────────
    if prior:
        m.revenue_growth_yoy = yoy_change(
            latest.get("total_revenue"),
            prior.get("total_revenue")
        )
        m.net_income_growth_yoy = yoy_change(
            latest.get("net_income"),
            prior.get("net_income")
        )
        m.fcf_growth_yoy = yoy_change(
            latest.get("free_cashflow"),
            prior.get("free_cashflow")
        )
        
        # Margin trends
        prior_net_margin = safe_div(prior.get("net_income"), prior.get("total_revenue"))
        if m.net_margin is not None and prior_net_margin is not None:
            # In percentage points, e.g. 25% -> 20% = -5pp
            m.margin_expansion = (m.net_margin - prior_net_margin) * 100
        
        # Debt trajectory
        m.debt_change_pct = yoy_change(
            latest.get("total_debt"),
            prior.get("total_debt")
        )
    
    # ── Anomaly detection ────────────────────────────────────────────────────
    if prior:
        # FCF divergence: Free cash flow and net income should be in the same ballpark
        # within 25% of each other (both positive or both negative)
        latest_ni = latest.get("net_income")
        latest_fcf = latest.get("free_cashflow")
        if latest_ni and latest_fcf and latest_ni != 0:
            divergence = abs(latest_fcf - latest_ni) / abs(latest_ni)
            if divergence > 0.25:
                m.fcf_divergence = True
        
        # Earnings miss: net income declined significantly despite revenue growth
        rev_growth = m.revenue_growth_yoy
        ni_growth = m.net_income_growth_yoy
        if (rev_growth and rev_growth > 0.05 and
            ni_growth and ni_growth < -0.20):
            m.earnings_miss = True
        
        # Margin compression: net margin fell 3+ percentage points
        if m.margin_expansion is not None and m.margin_expansion < -3:
            m.margin_compression = True
        
        # Debt spike: debt grew 20%+ YoY
        if m.debt_change_pct and m.debt_change_pct > 0.20:
            m.debt_spike = True
    
    return m


def metrics_to_df(metrics: FinancialMetrics) -> pd.DataFrame:
    """
    Convert a FinancialMetrics object to a single-row DataFrame.
    Useful for aggregating multiple tickers into one DataFrame.
    """
    return pd.DataFrame([{
        "ticker": metrics.ticker,
        
        # Latest absolute values
        "revenue_B": (metrics.latest_revenue / 1e9) if metrics.latest_revenue else None,
        "net_income_B": (metrics.latest_net_income / 1e9) if metrics.latest_net_income else None,
        "fcf_B": (metrics.latest_fcf / 1e9) if metrics.latest_fcf else None,
        "total_debt_B": (metrics.latest_total_debt / 1e9) if metrics.latest_total_debt else None,
        "total_equity_B": (metrics.latest_total_equity / 1e9) if metrics.latest_total_equity else None,
        "cash_B": (metrics.latest_cash / 1e9) if metrics.latest_cash else None,
        "eps": metrics.latest_eps,
        "current_ratio": metrics.current_ratio,
        
        # Profitability
        "net_margin_%": (metrics.net_margin * 100) if metrics.net_margin else None,
        "gross_margin_%": (metrics.gross_margin * 100) if metrics.gross_margin else None,
        "operating_margin_%": (metrics.operating_margin * 100) if metrics.operating_margin else None,
        "ebitda_margin_%": (metrics.ebitda_margin * 100) if metrics.ebitda_margin else None,
        "roe_%": (metrics.roe * 100) if metrics.roe else None,
        "roa_%": (metrics.roa * 100) if metrics.roa else None,
        
        # Solvency
        "debt_to_equity": metrics.debt_to_equity,
        "debt_to_assets": metrics.debt_to_assets,
        "interest_coverage": metrics.interest_coverage,
        
        # Cash flow
        "fcf_to_ni": metrics.fcf_to_net_income,
        "operating_cf_to_ni": metrics.operating_cf_to_net_income,
        
        # Growth
        "revenue_growth_yoy_%": (metrics.revenue_growth_yoy * 100) if metrics.revenue_growth_yoy else None,
        "net_income_growth_yoy_%": (metrics.net_income_growth_yoy * 100) if metrics.net_income_growth_yoy else None,
        "fcf_growth_yoy_%": (metrics.fcf_growth_yoy * 100) if metrics.fcf_growth_yoy else None,
        
        # Trends
        "margin_expansion_pp": metrics.margin_expansion,
        "debt_change_yoy_%": (metrics.debt_change_pct * 100) if metrics.debt_change_pct else None,
        
        # Flags
        "flag_fcf_divergence": metrics.fcf_divergence,
        "flag_earnings_miss": metrics.earnings_miss,
        "flag_margin_compression": metrics.margin_compression,
        "flag_debt_spike": metrics.debt_spike,
    }])


# ── Main API ───────────────────────────────────────────────────────────────────

def analyze_ticker(ticker: str, db_path: Path = DB_PATH) -> FinancialMetrics:
    """
    Analyze a single ticker. Returns FinancialMetrics object.
    
    Usage:
        metrics = analyze_ticker("AAPL")
        print(metrics.net_margin)  # profit margin as decimal
        print(metrics.debt_to_equity)
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    metrics = compute_metrics(ticker, conn)
    conn.close()
    
    return metrics


def analyze_tickers(tickers: list[str], db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Analyze multiple tickers. Returns a DataFrame with one row per ticker.
    
    Usage:
        df = analyze_tickers(["AAPL", "MSFT", "GOOGL"])
        print(df)
        print(df.loc[df["flag_debt_spike"], ["ticker", "debt_change_yoy_%"]])
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    dfs = []
    for ticker in tickers:
        metrics = compute_metrics(ticker, conn)
        df = metrics_to_df(metrics)
        dfs.append(df)
    
    conn.close()
    
    result = pd.concat(dfs, ignore_index=True)
    return result


def print_summary(metrics: FinancialMetrics):
    """Pretty-print a metrics object."""
    print(f"\n{'═'*60}")
    print(f"  Financial Analysis: {metrics.ticker}")
    print(f"{'═'*60}")
    
    print(f"\n  Latest Annual (in billions):")
    print(f"    Revenue:      ${metrics.latest_revenue/1e9:>8.2f}B")
    print(f"    Net Income:   ${metrics.latest_net_income/1e9:>8.2f}B" if metrics.latest_net_income else "    Net Income:   N/A")
    print(f"    Free CF:      ${metrics.latest_fcf/1e9:>8.2f}B" if metrics.latest_fcf else "    Free CF:      N/A")
    print(f"    Total Debt:   ${metrics.latest_total_debt/1e9:>8.2f}B" if metrics.latest_total_debt else "    Total Debt:   N/A")
    print(f"    Equity:       ${metrics.latest_total_equity/1e9:>8.2f}B" if metrics.latest_total_equity else "    Equity:       N/A")
    print(f"    Cash:         ${metrics.latest_cash/1e9:>8.2f}B" if metrics.latest_cash else "    Cash:         N/A")
    
    print(f"\n  Profitability Ratios:")
    print(f"    Net Margin:   {metrics.net_margin*100:>7.2f}%" if metrics.net_margin else "    Net Margin:   N/A")
    print(f"    Gross Margin: {metrics.gross_margin*100:>7.2f}%" if metrics.gross_margin else "    Gross Margin: N/A")
    print(f"    EBITDA Marg:  {metrics.ebitda_margin*100:>7.2f}%" if metrics.ebitda_margin else "    EBITDA Marg:  N/A")
    print(f"    ROE:          {metrics.roe*100:>7.2f}%" if metrics.roe else "    ROE:          N/A")
    print(f"    ROA:          {metrics.roa*100:>7.2f}%" if metrics.roa else "    ROA:          N/A")
    
    print(f"\n  Solvency:")
    print(f"    D/E Ratio:    {metrics.debt_to_equity:>8.2f}x" if metrics.debt_to_equity else "    D/E Ratio:    N/A")
    print(f"    D/A Ratio:    {metrics.debt_to_assets:>8.2f}x" if metrics.debt_to_assets else "    D/A Ratio:    N/A")
    print(f"    Int. Cov.:    {metrics.interest_coverage:>8.2f}x" if metrics.interest_coverage else "    Int. Cov.:    N/A")
    
    print(f"\n  Liquidity:")
    print(f"    Current:      {metrics.current_ratio:>8.2f}x" if metrics.current_ratio else "    Current:      N/A")
    
    print(f"\n  Cash Flow:")
    print(f"    FCF/NI:       {metrics.fcf_to_net_income:>8.2f}x" if metrics.fcf_to_net_income else "    FCF/NI:       N/A")
    print(f"    Op CF/NI:     {metrics.operating_cf_to_net_income:>8.2f}x" if metrics.operating_cf_to_net_income else "    Op CF/NI:     N/A")
    
    print(f"\n  YoY Growth:")
    print(f"    Revenue:      {metrics.revenue_growth_yoy*100:>7.2f}%" if metrics.revenue_growth_yoy else "    Revenue:      N/A")
    print(f"    Net Income:   {metrics.net_income_growth_yoy*100:>7.2f}%" if metrics.net_income_growth_yoy else "    Net Income:   N/A")
    print(f"    Free CF:      {metrics.fcf_growth_yoy*100:>7.2f}%" if metrics.fcf_growth_yoy else "    Free CF:      N/A")
    
    print(f"\n  Margin Trends:")
    print(f"    Expansion:    {metrics.margin_expansion:>7.2f}pp" if metrics.margin_expansion else "    Expansion:    N/A")
    print(f"    Debt Change:  {metrics.debt_change_pct*100:>7.2f}%" if metrics.debt_change_pct else "    Debt Change:  N/A")
    
    print(f"\n  ⚠️  Anomaly Flags:")
    flags = [
        ("FCF Divergence", metrics.fcf_divergence),
        ("Earnings Miss", metrics.earnings_miss),
        ("Margin Compression", metrics.margin_compression),
        ("Debt Spike", metrics.debt_spike),
    ]
    any_flagged = False
    for name, flagged in flags:
        if flagged:
            print(f"    🚩 {name}")
            any_flagged = True
    if not any_flagged:
        print(f"    ✓ No major anomalies detected")
    
    print(f"{'═'*60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze financial metrics for one or more tickers"
    )
    parser.add_argument("--ticker", nargs="+", required=True,
                        help="One or more ticker symbols (e.g. AAPL MSFT)")
    parser.add_argument("--db", default=str(DB_PATH),
                        help=f"Path to SQLite database (default: {DB_PATH})")
    parser.add_argument("--csv", help="Export results to CSV file")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    
    # Analyze each ticker individually (for pretty printing)
    for ticker in args.ticker:
        try:
            metrics = analyze_ticker(ticker, db_path)
            print_summary(metrics)
        except Exception as e:
            log.error(f"Failed to analyze {ticker}: {e}")
    
    # Export to CSV if requested
    if args.csv:
        df = analyze_tickers(args.ticker, db_path)
        df.to_csv(args.csv, index=False)
        log.info(f"Exported {len(df)} tickers to {args.csv}")