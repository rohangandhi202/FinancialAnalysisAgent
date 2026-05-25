"""
financial_agent/data_ingestion.py

Fetches income statement, balance sheet, cash flow, and price history
from Alpha Vantage and stores them in a local SQLite database.

Free tier limits: 25 API calls/day
Strategy: cache everything aggressively — never re-fetch data you already have.

Usage:
    python data_ingestion.py --ticker AAPL
    python data_ingestion.py --ticker AAPL --force-refresh
    python data_ingestion.py --ticker AAPL MSFT GOOGL
"""

import sqlite3
import requests
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

API_KEY = "YOUR_API_KEY_HERE"          # Free at alphavantage.co
BASE_URL = "https://www.alphavantage.co/query"
DB_PATH = Path("financial_data.db")

# How old cached data can be before we consider re-fetching (days)
CACHE_TTL_DAYS = {
    "income_statement": 90,    # Quarterly reports don't change
    "balance_sheet": 90,
    "cash_flow": 90,
    "price_history": 1,        # Daily prices refresh often
}

RATE_LIMIT_DELAY = 12          # seconds between calls (free tier: 5 calls/min)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── Database setup ─────────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Create all tables if they don't exist.
    Returns an open connection with WAL mode enabled for better concurrency.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row    # lets you access columns by name

    conn.executescript("""
        -- Raw API responses (source of truth — never lose the original data)
        CREATE TABLE IF NOT EXISTS raw_responses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            data_type   TEXT    NOT NULL,   -- e.g. 'income_statement'
            fetched_at  TEXT    NOT NULL,   -- ISO timestamp
            response    TEXT    NOT NULL    -- full JSON blob
        );
        CREATE INDEX IF NOT EXISTS idx_raw_ticker_type
            ON raw_responses(ticker, data_type, fetched_at DESC);

        -- Annual income statements (one row per fiscal year)
        CREATE TABLE IF NOT EXISTS income_statements (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker              TEXT    NOT NULL,
            fiscal_year         TEXT    NOT NULL,   -- e.g. '2023-12-31'
            report_type         TEXT    NOT NULL,   -- 'annual' | 'quarterly'
            total_revenue       REAL,
            gross_profit        REAL,
            ebitda              REAL,
            ebit                REAL,
            net_income          REAL,
            eps                 REAL,
            research_dev        REAL,
            operating_income    REAL,
            interest_expense    REAL,
            income_tax          REAL,
            fetched_at          TEXT    NOT NULL,
            UNIQUE(ticker, fiscal_year, report_type)
        );

        -- Balance sheets
        CREATE TABLE IF NOT EXISTS balance_sheets (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker                  TEXT    NOT NULL,
            fiscal_year             TEXT    NOT NULL,
            report_type             TEXT    NOT NULL,
            total_assets            REAL,
            total_liabilities       REAL,
            total_equity            REAL,
            cash_and_equivalents    REAL,
            short_term_debt         REAL,
            long_term_debt          REAL,
            total_debt              REAL,
            current_assets          REAL,
            current_liabilities     REAL,
            inventory               REAL,
            retained_earnings       REAL,
            fetched_at              TEXT    NOT NULL,
            UNIQUE(ticker, fiscal_year, report_type)
        );

        -- Cash flow statements
        CREATE TABLE IF NOT EXISTS cash_flows (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker                      TEXT    NOT NULL,
            fiscal_year                 TEXT    NOT NULL,
            report_type                 TEXT    NOT NULL,
            operating_cashflow          REAL,
            capex                       REAL,
            free_cashflow               REAL,   -- computed: operating - capex
            dividends_paid              REAL,
            net_income                  REAL,
            depreciation_amortization   REAL,
            stock_buybacks              REAL,
            fetched_at                  TEXT    NOT NULL,
            UNIQUE(ticker, fiscal_year, report_type)
        );

        -- Daily price history (OHLCV)
        CREATE TABLE IF NOT EXISTS price_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            UNIQUE(ticker, date)
        );

        -- Metadata / cache tracking
        CREATE TABLE IF NOT EXISTS fetch_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            data_type   TEXT    NOT NULL,
            fetched_at  TEXT    NOT NULL,
            status      TEXT    NOT NULL,   -- 'success' | 'error' | 'cached'
            note        TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_fetch_log_ticker
            ON fetch_log(ticker, data_type, fetched_at DESC);
    """)
    conn.commit()
    log.info(f"Database ready: {db_path.resolve()}")
    return conn


# ── Cache helpers ──────────────────────────────────────────────────────────────

def is_cached(conn: sqlite3.Connection, ticker: str, data_type: str) -> bool:
    """
    Returns True if we have a recent enough fetch for this ticker + data_type.
    'Recent enough' is defined by CACHE_TTL_DAYS.
    """
    ttl = CACHE_TTL_DAYS.get(data_type, 7)
    cutoff = (datetime.utcnow() - timedelta(days=ttl)).isoformat()

    row = conn.execute("""
        SELECT fetched_at FROM fetch_log
        WHERE ticker = ? AND data_type = ? AND status = 'success'
          AND fetched_at > ?
        ORDER BY fetched_at DESC
        LIMIT 1
    """, (ticker, data_type, cutoff)).fetchone()

    if row:
        log.info(f"Cache hit: {ticker} / {data_type} (last fetched {row['fetched_at'][:10]})")
        return True
    return False


def log_fetch(conn: sqlite3.Connection, ticker: str, data_type: str,
              status: str, note: str = ""):
    conn.execute("""
        INSERT INTO fetch_log (ticker, data_type, fetched_at, status, note)
        VALUES (?, ?, ?, ?, ?)
    """, (ticker, data_type, datetime.utcnow().isoformat(), status, note))
    conn.commit()


# ── Alpha Vantage API calls ────────────────────────────────────────────────────

def _fetch(function: str, ticker: str, extra_params: dict = None) -> dict:
    """
    Raw HTTP call to Alpha Vantage. Returns parsed JSON.
    Raises on HTTP errors or API error messages.
    """
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": API_KEY,
    }
    if extra_params:
        params.update(extra_params)

    log.info(f"Fetching {function} for {ticker}...")
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    # Alpha Vantage returns errors as JSON keys, not HTTP status codes
    if "Error Message" in data:
        raise ValueError(f"API error for {ticker}/{function}: {data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Rate limit hit: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"API limit: {data['Information']}")

    return data


def _to_float(val) -> float | None:
    """Convert AV string values ('None', '', numbers) to float or None."""
    if val in (None, "None", "", "N/A"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Fetch + store functions ───────────────────────────────────────────────────

def fetch_income_statement(conn: sqlite3.Connection, ticker: str,
                           force: bool = False) -> bool:
    data_type = "income_statement"
    if not force and is_cached(conn, ticker, data_type):
        return True

    try:
        data = _fetch("INCOME_STATEMENT", ticker)

        # Store raw response
        conn.execute("""
            INSERT INTO raw_responses (ticker, data_type, fetched_at, response)
            VALUES (?, ?, ?, ?)
        """, (ticker, data_type, datetime.utcnow().isoformat(), json.dumps(data)))

        # Parse annual and quarterly reports
        for report_type, key in [("annual", "annualReports"), ("quarterly", "quarterlyReports")]:
            for report in data.get(key, []):
                conn.execute("""
                    INSERT OR REPLACE INTO income_statements
                        (ticker, fiscal_year, report_type, total_revenue, gross_profit,
                         ebitda, ebit, net_income, eps, research_dev, operating_income,
                         interest_expense, income_tax, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    report.get("fiscalDateEnding"),
                    report_type,
                    _to_float(report.get("totalRevenue")),
                    _to_float(report.get("grossProfit")),
                    _to_float(report.get("ebitda")),
                    _to_float(report.get("ebit")),
                    _to_float(report.get("netIncome")),
                    _to_float(report.get("reportedEPS")),
                    _to_float(report.get("researchAndDevelopment")),
                    _to_float(report.get("operatingIncome")),
                    _to_float(report.get("interestExpense")),
                    _to_float(report.get("incomeTaxExpense")),
                    datetime.utcnow().isoformat(),
                ))

        conn.commit()
        log_fetch(conn, ticker, data_type, "success")
        log.info(f"Stored income statements for {ticker}")
        return True

    except Exception as e:
        log_fetch(conn, ticker, data_type, "error", str(e))
        log.error(f"Failed to fetch income statement for {ticker}: {e}")
        return False


def fetch_balance_sheet(conn: sqlite3.Connection, ticker: str,
                        force: bool = False) -> bool:
    data_type = "balance_sheet"
    if not force and is_cached(conn, ticker, data_type):
        return True

    try:
        data = _fetch("BALANCE_SHEET", ticker)

        conn.execute("""
            INSERT INTO raw_responses (ticker, data_type, fetched_at, response)
            VALUES (?, ?, ?, ?)
        """, (ticker, data_type, datetime.utcnow().isoformat(), json.dumps(data)))

        for report_type, key in [("annual", "annualReports"), ("quarterly", "quarterlyReports")]:
            for report in data.get(key, []):
                total_debt = (
                    (_to_float(report.get("shortTermDebt")) or 0) +
                    (_to_float(report.get("longTermDebt")) or 0)
                )
                conn.execute("""
                    INSERT OR REPLACE INTO balance_sheets
                        (ticker, fiscal_year, report_type, total_assets, total_liabilities,
                         total_equity, cash_and_equivalents, short_term_debt, long_term_debt,
                         total_debt, current_assets, current_liabilities, inventory,
                         retained_earnings, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    report.get("fiscalDateEnding"),
                    report_type,
                    _to_float(report.get("totalAssets")),
                    _to_float(report.get("totalLiabilities")),
                    _to_float(report.get("totalShareholderEquity")),
                    _to_float(report.get("cashAndCashEquivalentsAtCarryingValue")),
                    _to_float(report.get("shortTermDebt")),
                    _to_float(report.get("longTermDebt")),
                    total_debt if total_debt > 0 else None,
                    _to_float(report.get("totalCurrentAssets")),
                    _to_float(report.get("totalCurrentLiabilities")),
                    _to_float(report.get("inventory")),
                    _to_float(report.get("retainedEarnings")),
                    datetime.utcnow().isoformat(),
                ))

        conn.commit()
        log_fetch(conn, ticker, data_type, "success")
        log.info(f"Stored balance sheets for {ticker}")
        return True

    except Exception as e:
        log_fetch(conn, ticker, data_type, "error", str(e))
        log.error(f"Failed to fetch balance sheet for {ticker}: {e}")
        return False


def fetch_cash_flow(conn: sqlite3.Connection, ticker: str,
                    force: bool = False) -> bool:
    data_type = "cash_flow"
    if not force and is_cached(conn, ticker, data_type):
        return True

    try:
        data = _fetch("CASH_FLOW", ticker)

        conn.execute("""
            INSERT INTO raw_responses (ticker, data_type, fetched_at, response)
            VALUES (?, ?, ?, ?)
        """, (ticker, data_type, datetime.utcnow().isoformat(), json.dumps(data)))

        for report_type, key in [("annual", "annualReports"), ("quarterly", "quarterlyReports")]:
            for report in data.get(key, []):
                operating_cf = _to_float(report.get("operatingCashflow"))
                capex = _to_float(report.get("capitalExpenditures"))
                # FCF = Operating CF - CapEx (capex is often negative in the API — normalize)
                if operating_cf is not None and capex is not None:
                    free_cf = operating_cf - abs(capex)
                else:
                    free_cf = None

                conn.execute("""
                    INSERT OR REPLACE INTO cash_flows
                        (ticker, fiscal_year, report_type, operating_cashflow, capex,
                         free_cashflow, dividends_paid, net_income,
                         depreciation_amortization, stock_buybacks, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    report.get("fiscalDateEnding"),
                    report_type,
                    operating_cf,
                    capex,
                    free_cf,
                    _to_float(report.get("dividendPayout")),
                    _to_float(report.get("netIncome")),
                    _to_float(report.get("depreciationDepletionAndAmortization")),
                    _to_float(report.get("paymentsForRepurchaseOfCommonStock")),
                    datetime.utcnow().isoformat(),
                ))

        conn.commit()
        log_fetch(conn, ticker, data_type, "success")
        log.info(f"Stored cash flows for {ticker}")
        return True

    except Exception as e:
        log_fetch(conn, ticker, data_type, "error", str(e))
        log.error(f"Failed to fetch cash flow for {ticker}: {e}")
        return False


def fetch_price_history(conn: sqlite3.Connection, ticker: str,
                        force: bool = False, compact: bool = True) -> bool:
    """
    Fetches daily OHLCV price history.
    compact=True  → last 100 trading days (1 API call)
    compact=False → full history, up to 20 years (1 API call, larger response)
    """
    data_type = "price_history"
    if not force and is_cached(conn, ticker, data_type):
        return True

    try:
        output_size = "compact" if compact else "full"
        data = _fetch("TIME_SERIES_DAILY", ticker, {"outputsize": output_size})

        conn.execute("""
            INSERT INTO raw_responses (ticker, data_type, fetched_at, response)
            VALUES (?, ?, ?, ?)
        """, (ticker, data_type, datetime.utcnow().isoformat(), json.dumps(data)))

        time_series = data.get("Time Series (Daily)", {})
        rows = []
        for date_str, values in time_series.items():
            rows.append((
                ticker,
                date_str,
                _to_float(values.get("1. open")),
                _to_float(values.get("2. high")),
                _to_float(values.get("3. low")),
                _to_float(values.get("4. close")),
                int(values.get("5. volume", 0) or 0),
            ))

        conn.executemany("""
            INSERT OR IGNORE INTO price_history
                (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)

        conn.commit()
        log_fetch(conn, ticker, data_type, "success",
                  f"{len(rows)} trading days stored")
        log.info(f"Stored {len(rows)} price records for {ticker}")
        return True

    except Exception as e:
        log_fetch(conn, ticker, data_type, "error", str(e))
        log.error(f"Failed to fetch price history for {ticker}: {e}")
        return False


# ── Full pipeline ──────────────────────────────────────────────────────────────

def ingest_ticker(ticker: str, conn: sqlite3.Connection,
                  force: bool = False) -> dict:
    """
    Run the full ingestion pipeline for a single ticker.
    Respects the free-tier rate limit with a delay between calls.
    Returns a summary dict of what succeeded.
    """
    ticker = ticker.upper().strip()
    log.info(f"{'─'*40}")
    log.info(f"Starting ingestion for {ticker}")

    results = {}
    fetchers = [
        ("income_statement", fetch_income_statement),
        ("balance_sheet",    fetch_balance_sheet),
        ("cash_flow",        fetch_cash_flow),
        ("price_history",    fetch_price_history),
    ]

    for i, (name, fn) in enumerate(fetchers):
        results[name] = fn(conn, ticker, force=force)
        # Rate limit: wait between uncached calls (skip wait if result was cached)
        if results[name] and i < len(fetchers) - 1:
            if not is_cached(conn, ticker, fetchers[i+1][0]):
                log.info(f"Waiting {RATE_LIMIT_DELAY}s (rate limit)...")
                time.sleep(RATE_LIMIT_DELAY)

    successes = sum(v for v in results.values())
    log.info(f"Ingestion complete for {ticker}: {successes}/{len(fetchers)} succeeded")
    return results


# ── Quick query helpers ────────────────────────────────────────────────────────

def get_annual_financials(conn: sqlite3.Connection, ticker: str,
                          years: int = 5) -> list[dict]:
    """
    Returns a merged view of income + balance + cash flow for the last N annual periods.
    Useful for ratio calculation in the analysis layer.
    """
    rows = conn.execute("""
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
            b.inventory,
            c.operating_cashflow,
            c.free_cashflow,
            c.capex,
            c.dividends_paid
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
        LIMIT ?
    """, (ticker.upper(), years)).fetchall()

    return [dict(row) for row in rows]


def get_recent_prices(conn: sqlite3.Connection, ticker: str,
                      days: int = 90) -> list[dict]:
    """Returns the most recent N trading days of price data."""
    rows = conn.execute("""
        SELECT date, open, high, low, close, volume
        FROM price_history
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
    """, (ticker.upper(), days)).fetchall()

    return [dict(row) for row in rows]


def print_summary(conn: sqlite3.Connection, ticker: str):
    """Print a quick data availability summary for a ticker."""
    ticker = ticker.upper()
    print(f"\n{'═'*45}")
    print(f"  Data summary: {ticker}")
    print(f"{'═'*45}")

    # Income statement
    rows = conn.execute("""
        SELECT COUNT(*) as cnt, MIN(fiscal_year) as oldest, MAX(fiscal_year) as newest
        FROM income_statements WHERE ticker = ? AND report_type = 'annual'
    """, (ticker,)).fetchone()
    print(f"  Income statements:  {rows['cnt']} annual ({rows['oldest']} → {rows['newest']})")

    # Balance sheet
    rows = conn.execute("""
        SELECT COUNT(*) as cnt FROM balance_sheets
        WHERE ticker = ? AND report_type = 'annual'
    """, (ticker,)).fetchone()
    print(f"  Balance sheets:     {rows['cnt']} annual records")

    # Cash flow
    rows = conn.execute("""
        SELECT COUNT(*) as cnt FROM cash_flows
        WHERE ticker = ? AND report_type = 'annual'
    """, (ticker,)).fetchone()
    print(f"  Cash flow stmts:    {rows['cnt']} annual records")

    # Price history
    rows = conn.execute("""
        SELECT COUNT(*) as cnt, MIN(date) as oldest, MAX(date) as newest
        FROM price_history WHERE ticker = ?
    """, (ticker,)).fetchone()
    print(f"  Price history:      {rows['cnt']} trading days ({rows['oldest']} → {rows['newest']})")
    print(f"{'═'*45}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest financial data from Alpha Vantage into SQLite"
    )
    parser.add_argument("--ticker", nargs="+", required=True,
                        help="One or more ticker symbols (e.g. AAPL MSFT)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Bypass cache and re-fetch everything")
    parser.add_argument("--db", default=str(DB_PATH),
                        help=f"Path to SQLite database (default: {DB_PATH})")
    parser.add_argument("--full-history", action="store_true",
                        help="Fetch full price history instead of last 100 days")
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = init_db(db_path)

    for ticker in args.ticker:
        ingest_ticker(ticker, conn, force=args.force_refresh)
        print_summary(conn, ticker)

        # Peek at the data
        financials = get_annual_financials(conn, ticker, years=3)
        if financials:
            print(f"  Most recent annual revenue: "
                  f"${financials[0]['total_revenue']:,.0f}" if financials[0]['total_revenue'] else "N/A")
            print(f"  Most recent FCF:            "
                  f"${financials[0]['free_cashflow']:,.0f}" if financials[0]['free_cashflow'] else "N/A")
        print()

    conn.close()