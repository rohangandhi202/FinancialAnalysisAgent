"""
financial_agent/test_agent.py

Tests the agent's tool execution layer and JSON output parsing.
Mocks the Claude API so no real API calls are made.

Usage:
    python -m financial_agent.test_agent
"""

import sys
import json
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Point DB_PATH at our test DB before importing agent
import financial_agent.agent as agent_module
from financial_agent import agent as ag


# ── Test DB setup (same schema as real DB) ─────────────────────────────────────

def create_test_db(db_path: Path):
    """Minimal test DB with realistic AAPL-like numbers."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE income_statements (
            ticker TEXT, fiscal_year TEXT, report_type TEXT,
            total_revenue REAL, gross_profit REAL, ebitda REAL, ebit REAL,
            net_income REAL, eps REAL, research_dev REAL, operating_income REAL,
            interest_expense REAL, income_tax REAL, fetched_at TEXT,
            PRIMARY KEY(ticker, fiscal_year, report_type)
        );
        CREATE TABLE balance_sheets (
            ticker TEXT, fiscal_year TEXT, report_type TEXT,
            total_assets REAL, total_liabilities REAL, total_equity REAL,
            cash_and_equivalents REAL, short_term_debt REAL, long_term_debt REAL,
            total_debt REAL, current_assets REAL, current_liabilities REAL,
            inventory REAL, retained_earnings REAL, fetched_at TEXT,
            PRIMARY KEY(ticker, fiscal_year, report_type)
        );
        CREATE TABLE cash_flows (
            ticker TEXT, fiscal_year TEXT, report_type TEXT,
            operating_cashflow REAL, capex REAL, free_cashflow REAL,
            dividends_paid REAL, net_income REAL,
            depreciation_amortization REAL, stock_buybacks REAL, fetched_at TEXT,
            PRIMARY KEY(ticker, fiscal_year, report_type)
        );
        CREATE TABLE price_history (
            ticker TEXT, date TEXT, open REAL, high REAL,
            low REAL, close REAL, volume INTEGER,
            PRIMARY KEY(ticker, date)
        );
        CREATE TABLE fetch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, data_type TEXT, fetched_at TEXT, status TEXT, note TEXT
        );
    """)

    now = datetime.utcnow().isoformat()

    # Two years of data for AAPL-like company
    for year, rev, ni, fcf, debt in [
        ("2023-12-31", 383e9, 97e9, 101e9, 105e9),
        ("2022-12-31", 394e9, 100e9, 111e9,  97e9),
    ]:
        conn.execute("""INSERT INTO income_statements VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("AAPL", year, "annual", rev, 169e9, 130e9, 114e9, ni,
             6.13, 30e9, 114e9, 3.9e9, 30e9, now))
        conn.execute("""INSERT INTO balance_sheets VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("AAPL", year, "annual", 352e9, 290e9, 62e9, 30e9,
             10e9, 95e9, debt, 143e9, 145e9, 6e9, -214e9, now))
        conn.execute("""INSERT INTO cash_flows VALUES
            (?,?,?,?,?,?,?,?,?,?,?)""",
            ("AAPL", year, "annual", 113e9, 11e9, fcf,
             15e9, ni, 11e9, 77e9, now))

    # Price history
    prices = [
        ("2024-01-03", 184.25), ("2024-01-02", 185.20),
        ("2024-01-01", 192.53), ("2023-12-29", 193.58),
        ("2023-12-28", 193.03), ("2023-12-27", 193.15),
    ]
    for date, close in prices:
        conn.execute(
            "INSERT INTO price_history VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("AAPL", date, close - 1, close + 1, close - 2, close, 50_000_000)
        )

    conn.commit()
    conn.close()


# ── Test runner ────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  ✓ {name}")

    def fail(self, name, reason):
        self.failed += 1
        self.errors.append(f"{name}: {reason}")
        print(f"  ✗ {name}")
        print(f"      → {reason}")


def run_tests():
    r = TestResult()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        create_test_db(db_path)

        # Patch DB_PATH in the agent module
        agent_module.DB_PATH = db_path

        # ── 1. Tool: get_financial_metrics ─────────────────────────────────────
        try:
            result_str = ag.execute_tool("get_financial_metrics", {"ticker": "AAPL"})
            result = json.loads(result_str)

            assert result["ticker"] == "AAPL"
            assert "profitability" in result
            assert "solvency" in result
            assert "growth_yoy" in result
            assert result["latest_financials_billions"]["revenue"] is not None

            # Spot-check a specific value: revenue ≈ 383B
            rev = result["latest_financials_billions"]["revenue"]
            assert 380 < rev < 390, f"Revenue should be ~383B, got {rev}B"

            r.ok("get_financial_metrics returns correct structure and values")
        except Exception as e:
            r.fail("get_financial_metrics", str(e))

        # ── 2. Tool: get_price_history ─────────────────────────────────────────
        try:
            result_str = ag.execute_tool("get_price_history", {"ticker": "AAPL", "days": 10})
            result = json.loads(result_str)

            assert result["ticker"] == "AAPL"
            assert result["latest_close"] is not None
            assert result["52_week_high"] is not None
            assert result["52_week_low"] is not None
            assert result["52_week_high"] >= result["52_week_low"]
            assert "recent_prices_sample" in result
            assert len(result["recent_prices_sample"]) > 0

            r.ok("get_price_history returns correct structure")
        except Exception as e:
            r.fail("get_price_history", str(e))

        # ── 3. Tool: compare_companies ─────────────────────────────────────────
        try:
            # Add a second company (MSFT-like) to the DB
            conn = sqlite3.connect(db_path)
            now = datetime.utcnow().isoformat()
            conn.execute("""INSERT INTO income_statements VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("MSFT", "2023-12-31", "annual", 211e9, 140e9, 100e9, 88e9,
                 72e9, 9.65, 27e9, 88e9, 1.9e9, 18e9, now))
            conn.execute("""INSERT INTO balance_sheets VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("MSFT", "2023-12-31", "annual", 411e9, 205e9, 206e9, 75e9,
                 0, 42e9, 42e9, 184e9, 125e9, 2.5e9, 118e9, now))
            conn.execute("""INSERT INTO cash_flows VALUES
                (?,?,?,?,?,?,?,?,?,?,?)""",
                ("MSFT", "2023-12-31", "annual", 87e9, 28e9, 59e9,
                 11e9, 72e9, 14e9, 22e9, now))
            conn.commit()
            conn.close()

            result_str = ag.execute_tool("compare_companies", {"tickers": ["AAPL", "MSFT"]})
            result = json.loads(result_str)

            assert isinstance(result, list)
            assert len(result) == 2
            tickers = {item["ticker"] for item in result}
            assert tickers == {"AAPL", "MSFT"}

            # Each item should have key comparison fields
            for item in result:
                assert "net_margin_%" in item
                assert "debt_to_equity" in item
                assert "flags" in item

            r.ok("compare_companies returns comparison for both tickers")
        except Exception as e:
            r.fail("compare_companies", str(e))

        # ── 4. Tool: get_anomaly_flags ─────────────────────────────────────────
        try:
            result_str = ag.execute_tool("get_anomaly_flags", {"ticker": "AAPL"})
            result = json.loads(result_str)

            assert result["ticker"] == "AAPL"
            assert "anomalies" in result
            assert isinstance(result["anomalies"], list)
            assert len(result["anomalies"]) > 0

            # Each anomaly should have required fields
            for anomaly in result["anomalies"]:
                assert "flag" in anomaly
                assert "severity" in anomaly
                assert "detail" in anomaly

            r.ok("get_anomaly_flags returns structured anomaly list")
        except Exception as e:
            r.fail("get_anomaly_flags", str(e))

        # ── 5. Unknown tool returns error gracefully ───────────────────────────
        try:
            result_str = ag.execute_tool("nonexistent_tool", {})
            result = json.loads(result_str)
            assert "error" in result
            r.ok("Unknown tool returns error gracefully (no crash)")
        except Exception as e:
            r.fail("unknown tool handling", str(e))

        # ── 6. JSON output parsing ─────────────────────────────────────────────
        try:
            # Simulate a valid agent JSON output
            mock_report = {
                "ticker": "AAPL",
                "analysis_date": "2024-01-03",
                "executive_summary": "Apple demonstrates strong profitability.",
                "strengths": [{"point": "High net margin", "supporting_data": "25.3%"}],
                "risks": [{"point": "Revenue decline", "supporting_data": "-2.8% YoY"}],
                "anomalies": [],
                "recommendation": {
                    "stance": "CAUTIOUS",
                    "confidence": "MEDIUM",
                    "rationale": "Strong margins but declining revenue.",
                    "key_metrics_to_watch": ["revenue_growth", "fcf_trend"]
                },
                "disclaimer": "For informational purposes only."
            }

            # Test that print_report doesn't crash on valid input
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                ag.print_report(mock_report)
            output = f.getvalue()

            assert "AAPL" in output
            assert "CAUTIOUS" in output
            assert "MEDIUM" in output
            assert "High net margin" in output

            r.ok("print_report formats valid report correctly")
        except Exception as e:
            r.fail("print_report", str(e))

        # ── 7. print_report handles error dict gracefully ─────────────────────
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                ag.print_report({"error": "Something went wrong"})
            output = f.getvalue()
            assert "Error" in output
            r.ok("print_report handles error dict without crashing")
        except Exception as e:
            r.fail("print_report error handling", str(e))

        # ── 8. Validate tool input schemas ────────────────────────────────────
        try:
            tool_names = {t["name"] for t in ag.TOOLS}
            expected = {
                "get_financial_metrics",
                "get_price_history",
                "compare_companies",
                "get_anomaly_flags"
            }
            assert tool_names == expected, f"Tool mismatch: {tool_names}"

            for tool in ag.TOOLS:
                assert "name" in tool
                assert "description" in tool
                assert "input_schema" in tool
                assert len(tool["description"]) > 20, "Tool description too short"

            r.ok("All 4 tools present with valid schemas")
        except Exception as e:
            r.fail("tool schema validation", str(e))

    # ── Results ────────────────────────────────────────────────────────────────
    total = r.passed + r.failed
    print(f"\n{'═'*45}")
    print(f"  Results: {r.passed}/{total} passed")
    if r.errors:
        print(f"\n  Failures:")
        for err in r.errors:
            print(f"    • {err}")
    print(f"{'═'*45}")

    return r.failed == 0


if __name__ == "__main__":
    print("\nRunning agent tests (no Claude API calls)...\n")
    success = run_tests()
    sys.exit(0 if success else 1)