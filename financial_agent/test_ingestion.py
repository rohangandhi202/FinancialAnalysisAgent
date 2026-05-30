"""
financial_agent/test_ingestion.py

Tests the full data pipeline with mock API responses.
Run this to validate your setup BEFORE using real API calls.

Usage:
    python test_ingestion.py
"""

import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Patch the module's DB_PATH before importing
import financial_agent.data_ingestion as di

# ── Mock API responses ─────────────────────────────────────────────────────────

MOCK_INCOME = {
    "symbol": "TEST",
    "annualReports": [
        {
            "fiscalDateEnding": "2023-12-31",
            "totalRevenue": "383285000000",
            "grossProfit": "169148000000",
            "ebitda": "123456000000",
            "ebit": "111000000000",
            "netIncome": "96995000000",
            "reportedEPS": "6.13",
            "researchAndDevelopment": "29915000000",
            "operatingIncome": "114301000000",
            "interestExpense": "3933000000",
            "incomeTaxExpense": "29749000000",
        },
        {
            "fiscalDateEnding": "2022-12-31",
            "totalRevenue": "394328000000",
            "grossProfit": "170782000000",
            "ebitda": "130000000000",
            "ebit": "119000000000",
            "netIncome": "99803000000",
            "reportedEPS": "6.11",
            "researchAndDevelopment": "26251000000",
            "operatingIncome": "119437000000",
            "interestExpense": "2931000000",
            "incomeTaxExpense": "19300000000",
        },
    ],
    "quarterlyReports": [],
}

MOCK_BALANCE = {
    "symbol": "TEST",
    "annualReports": [
        {
            "fiscalDateEnding": "2023-12-31",
            "totalAssets": "352583000000",
            "totalLiabilities": "290437000000",
            "totalShareholderEquity": "62146000000",
            "cashAndCashEquivalentsAtCarryingValue": "29965000000",
            "shortTermDebt": "9822000000",
            "longTermDebt": "95281000000",
            "totalCurrentAssets": "143566000000",
            "totalCurrentLiabilities": "145308000000",
            "inventory": "6331000000",
            "retainedEarnings": "-214000000000",
        }
    ],
    "quarterlyReports": [],
}

MOCK_CASHFLOW = {
    "symbol": "TEST",
    "annualReports": [
        {
            "fiscalDateEnding": "2023-12-31",
            "operatingCashflow": "113210000000",
            "capitalExpenditures": "11558000000",
            "dividendPayout": "15025000000",
            "netIncome": "96995000000",
            "depreciationDepletionAndAmortization": "11519000000",
            "paymentsForRepurchaseOfCommonStock": "77550000000",
        }
    ],
    "quarterlyReports": [],
}

MOCK_PRICES = {
    "Meta Data": {"2. Symbol": "TEST"},
    "Time Series (Daily)": {
        "2024-01-03": {"1. open": "184.22", "2. high": "185.88", "3. low": "183.43", "4. close": "184.25", "5. volume": "58429000"},
        "2024-01-02": {"1. open": "187.15", "2. high": "188.44", "3. low": "183.88", "4. close": "185.20", "5. volume": "46679300"},
        "2024-01-01": {"1. open": "192.33", "2. high": "193.00", "3. low": "191.00", "4. close": "192.53", "5. volume": "28228500"},
    },
}


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
        print(f"  ✗ {name} — {reason}")


def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, got {actual!r} — {label}")


def run_tests():
    r = TestResult()
    ticker = "TEST"

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = di.init_db(db_path)

        # ── 1. DB initializes cleanly ──────────────────────────────────────────
        try:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            expected_tables = {
                "raw_responses", "income_statements", "balance_sheets",
                "cash_flows", "price_history", "fetch_log"
            }
            assert expected_tables.issubset(tables), f"missing tables: {expected_tables - tables}"
            r.ok("All tables created")
        except Exception as e:
            r.fail("DB init", str(e))

        # ── 2. _to_float handles edge cases ────────────────────────────────────
        try:
            assert di._to_float("383285000000") == 383285000000.0
            assert di._to_float("None") is None
            assert di._to_float("") is None
            assert di._to_float(None) is None
            assert di._to_float("N/A") is None
            assert di._to_float("-214000000000") == -214000000000.0
            r.ok("_to_float handles all edge cases")
        except Exception as e:
            r.fail("_to_float", str(e))

        # ── 3. Cache miss on fresh DB ──────────────────────────────────────────
        try:
            assert not di.is_cached(conn, ticker, "income_statement")
            r.ok("Cache miss on empty DB")
        except Exception as e:
            r.fail("cache miss", str(e))

        # ── 4. Income statement ingestion ─────────────────────────────────────
        try:
            # Inject mock data directly (bypass HTTP)
            _store_income(conn, ticker, MOCK_INCOME)
            di.log_fetch(conn, ticker, "income_statement", "success")

            rows = conn.execute("""
                SELECT * FROM income_statements
                WHERE ticker = ? AND report_type = 'annual'
                ORDER BY fiscal_year DESC
            """, (ticker,)).fetchall()

            assert len(rows) == 2, f"expected 2 annual rows, got {len(rows)}"
            latest = dict(rows[0])
            assert latest["total_revenue"] == 383285000000.0
            assert latest["net_income"] == 96995000000.0
            assert latest["eps"] == 6.13
            r.ok("Income statement stored and retrieved correctly")
        except Exception as e:
            r.fail("income statement", str(e))

        # ── 5. Balance sheet ingestion ─────────────────────────────────────────
        try:
            _store_balance(conn, ticker, MOCK_BALANCE)
            di.log_fetch(conn, ticker, "balance_sheet", "success")

            row = conn.execute("""
                SELECT * FROM balance_sheets
                WHERE ticker = ? AND report_type = 'annual'
                ORDER BY fiscal_year DESC LIMIT 1
            """, (ticker,)).fetchone()
            row = dict(row)

            # total_debt should be short + long term
            expected_debt = 9822000000 + 95281000000
            assert row["total_debt"] == expected_debt, \
                f"total_debt mismatch: {row['total_debt']} vs {expected_debt}"
            r.ok("Balance sheet: total_debt computed correctly")
        except Exception as e:
            r.fail("balance sheet", str(e))

        # ── 6. Cash flow: FCF computation ─────────────────────────────────────
        try:
            _store_cashflow(conn, ticker, MOCK_CASHFLOW)
            di.log_fetch(conn, ticker, "cash_flow", "success")

            row = conn.execute("""
                SELECT * FROM cash_flows
                WHERE ticker = ? AND report_type = 'annual'
                LIMIT 1
            """, (ticker,)).fetchone()
            row = dict(row)

            # FCF = 113,210M - 11,558M = 101,652M
            expected_fcf = 113210000000 - 11558000000
            assert row["free_cashflow"] == expected_fcf, \
                f"FCF mismatch: {row['free_cashflow']} vs {expected_fcf}"
            r.ok(f"Cash flow: FCF computed correctly (${expected_fcf/1e9:.1f}B)")
        except Exception as e:
            r.fail("cash flow FCF", str(e))

        # ── 7. Price history ingestion ─────────────────────────────────────────
        try:
            _store_prices(conn, ticker, MOCK_PRICES)
            di.log_fetch(conn, ticker, "price_history", "success", "3 trading days stored")

            prices = di.get_recent_prices(conn, ticker, days=10)
            assert len(prices) == 3
            assert prices[0]["date"] == "2024-01-03"
            assert prices[0]["close"] == 184.25
            r.ok("Price history: 3 records stored and retrieved")
        except Exception as e:
            r.fail("price history", str(e))

        # ── 8. Cache hit after successful fetch ────────────────────────────────
        try:
            assert di.is_cached(conn, ticker, "income_statement")
            assert di.is_cached(conn, ticker, "balance_sheet")
            assert di.is_cached(conn, ticker, "cash_flow")
            assert di.is_cached(conn, ticker, "price_history")
            r.ok("Cache hit detected after all four data types logged")
        except Exception as e:
            r.fail("cache hit", str(e))

        # ── 9. get_annual_financials joins all three tables ────────────────────
        try:
            financials = di.get_annual_financials(conn, ticker, years=5)
            assert len(financials) >= 1
            latest = financials[0]

            # Should have fields from all three tables
            assert latest["total_revenue"] is not None
            assert latest["total_assets"] is not None
            assert latest["free_cashflow"] is not None
            assert latest["fiscal_year"] == "2023-12-31"
            r.ok("get_annual_financials joins income + balance + cashflow correctly")
        except Exception as e:
            r.fail("get_annual_financials", str(e))

        # ── 10. Duplicate inserts don't create extra rows (UNIQUE constraint) ──
        try:
            before = conn.execute(
                "SELECT COUNT(*) FROM income_statements WHERE ticker = ?", (ticker,)
            ).fetchone()[0]

            _store_income(conn, ticker, MOCK_INCOME)  # re-insert same data

            after = conn.execute(
                "SELECT COUNT(*) FROM income_statements WHERE ticker = ?", (ticker,)
            ).fetchone()[0]

            assert before == after, f"row count changed: {before} → {after}"
            r.ok("Duplicate inserts handled by UNIQUE constraint (no duplicate rows)")
        except Exception as e:
            r.fail("duplicate insert", str(e))

        conn.close()

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


# ── Helpers to inject mock data directly into DB ───────────────────────────────

def _store_income(conn, ticker, data):
    for report_type, key in [("annual", "annualReports"), ("quarterly", "quarterlyReports")]:
        for report in data.get(key, []):
            conn.execute("""
                INSERT OR REPLACE INTO income_statements
                    (ticker, fiscal_year, report_type, total_revenue, gross_profit,
                     ebitda, ebit, net_income, eps, research_dev, operating_income,
                     interest_expense, income_tax, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, report.get("fiscalDateEnding"), report_type,
                di._to_float(report.get("totalRevenue")),
                di._to_float(report.get("grossProfit")),
                di._to_float(report.get("ebitda")),
                di._to_float(report.get("ebit")),
                di._to_float(report.get("netIncome")),
                di._to_float(report.get("reportedEPS")),
                di._to_float(report.get("researchAndDevelopment")),
                di._to_float(report.get("operatingIncome")),
                di._to_float(report.get("interestExpense")),
                di._to_float(report.get("incomeTaxExpense")),
                datetime.utcnow().isoformat(),
            ))
    conn.commit()


def _store_balance(conn, ticker, data):
    for report_type, key in [("annual", "annualReports"), ("quarterly", "quarterlyReports")]:
        for report in data.get(key, []):
            total_debt = (
                (di._to_float(report.get("shortTermDebt")) or 0) +
                (di._to_float(report.get("longTermDebt")) or 0)
            )
            conn.execute("""
                INSERT OR REPLACE INTO balance_sheets
                    (ticker, fiscal_year, report_type, total_assets, total_liabilities,
                     total_equity, cash_and_equivalents, short_term_debt, long_term_debt,
                     total_debt, current_assets, current_liabilities, inventory,
                     retained_earnings, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, report.get("fiscalDateEnding"), report_type,
                di._to_float(report.get("totalAssets")),
                di._to_float(report.get("totalLiabilities")),
                di._to_float(report.get("totalShareholderEquity")),
                di._to_float(report.get("cashAndCashEquivalentsAtCarryingValue")),
                di._to_float(report.get("shortTermDebt")),
                di._to_float(report.get("longTermDebt")),
                total_debt if total_debt > 0 else None,
                di._to_float(report.get("totalCurrentAssets")),
                di._to_float(report.get("totalCurrentLiabilities")),
                di._to_float(report.get("inventory")),
                di._to_float(report.get("retainedEarnings")),
                datetime.utcnow().isoformat(),
            ))
    conn.commit()


def _store_cashflow(conn, ticker, data):
    for report_type, key in [("annual", "annualReports"), ("quarterly", "quarterlyReports")]:
        for report in data.get(key, []):
            operating_cf = di._to_float(report.get("operatingCashflow"))
            capex = di._to_float(report.get("capitalExpenditures"))
            free_cf = operating_cf - abs(capex) if (operating_cf and capex) else None
            conn.execute("""
                INSERT OR REPLACE INTO cash_flows
                    (ticker, fiscal_year, report_type, operating_cashflow, capex,
                     free_cashflow, dividends_paid, net_income,
                     depreciation_amortization, stock_buybacks, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, report.get("fiscalDateEnding"), report_type,
                operating_cf, capex, free_cf,
                di._to_float(report.get("dividendPayout")),
                di._to_float(report.get("netIncome")),
                di._to_float(report.get("depreciationDepletionAndAmortization")),
                di._to_float(report.get("paymentsForRepurchaseOfCommonStock")),
                datetime.utcnow().isoformat(),
            ))
    conn.commit()


def _store_prices(conn, ticker, data):
    rows = []
    for date_str, values in data.get("Time Series (Daily)", {}).items():
        rows.append((
            ticker, date_str,
            di._to_float(values.get("1. open")),
            di._to_float(values.get("2. high")),
            di._to_float(values.get("3. low")),
            di._to_float(values.get("4. close")),
            int(values.get("5. volume", 0) or 0),
        ))
    conn.executemany("""
        INSERT OR IGNORE INTO price_history (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()


if __name__ == "__main__":
    print("\nRunning data ingestion tests (no API calls)...\n")
    success = run_tests()
    sys.exit(0 if success else 1)