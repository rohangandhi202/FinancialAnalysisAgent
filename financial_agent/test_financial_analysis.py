"""
financial_agent/test_financial_analysis.py

Tests the ratio calculation engine with realistic mock data.
No database reads — all data is mocked.

Usage:
    python -m financial_agent.test_financial_analysis
"""

import sys
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime

from financial_agent import financial_analysis as fa


# ── Mock financials ────────────────────────────────────────────────────────────

def create_test_db(db_path: Path):
    """Create a test DB with two years of mock financials for 'TEST' ticker."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    
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
    """)
    
    # Year 2 (prior) — solid financials
    conn.execute("""
        INSERT INTO income_statements VALUES
        ('TEST', '2022-12-31', 'annual', 100e9, 40e9, 30e9, 25e9,
         20e9, 1.50, 10e9, 25e9, 2e9, 5e9, ?)
    """, (datetime.utcnow().isoformat(),))
    
    conn.execute("""
        INSERT INTO balance_sheets VALUES
        ('TEST', '2022-12-31', 'annual', 200e9, 80e9, 120e9, 20e9,
         10e9, 30e9, 40e9, 80e9, 40e9, 5e9, 50e9, ?)
    """, (datetime.utcnow().isoformat(),))
    
    conn.execute("""
        INSERT INTO cash_flows VALUES
        ('TEST', '2022-12-31', 'annual', 22e9, 5e9, 17e9,
         3e9, 20e9, 4e9, 5e9, ?)
    """, (datetime.utcnow().isoformat(),))
    
    # Year 1 (latest) — revenue up 10%, but margins compressed
    conn.execute("""
        INSERT INTO income_statements VALUES
        ('TEST', '2023-12-31', 'annual', 110e9, 40e9, 31e9, 24e9,
         19e9, 1.45, 12e9, 24e9, 2.5e9, 5e9, ?)
    """, (datetime.utcnow().isoformat(),))
    
    conn.execute("""
        INSERT INTO balance_sheets VALUES
        ('TEST', '2023-12-31', 'annual', 220e9, 90e9, 130e9, 25e9,
         10e9, 40e9, 50e9, 90e9, 50e9, 6e9, 55e9, ?)
    """, (datetime.utcnow().isoformat(),))
    
    conn.execute("""
        INSERT INTO cash_flows VALUES
        ('TEST', '2023-12-31', 'annual', 20e9, 6e9, 14e9,
         3e9, 19e9, 5e9, 8e9, ?)
    """, (datetime.utcnow().isoformat(),))
    
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


def approx_eq(actual, expected, tolerance=0.01, label=""):
    """Assert float equality within tolerance."""
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"expected {expected:.4f}, got {actual:.4f} (diff: {abs(actual - expected):.4f}) — {label}"
        )
    return True


def run_tests():
    r = TestResult()
    ticker = "TEST"

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        create_test_db(db_path)

        # ── 1. Load mock data ──────────────────────────────────────────────────
        try:
            metrics = fa.analyze_ticker(ticker, db_path)
            assert metrics is not None
            r.ok("analyze_ticker loads without crashing")
        except Exception as e:
            r.fail("load metrics", str(e))
            return r

        # ── 2. Latest absolute values ──────────────────────────────────────────
        try:
            approx_eq(metrics.latest_revenue, 110e9, label="revenue")
            approx_eq(metrics.latest_net_income, 19e9, label="net income")
            approx_eq(metrics.latest_fcf, 14e9, label="fcf")
            approx_eq(metrics.latest_total_debt, 50e9, label="debt")
            approx_eq(metrics.latest_total_equity, 130e9, label="equity")
            r.ok("Latest annual values extracted correctly")
        except Exception as e:
            r.fail("latest values", str(e))

        # ── 3. Profitability ratios ────────────────────────────────────────────
        try:
            # Net margin: 19B / 110B ≈ 0.1727
            approx_eq(metrics.net_margin, 19e9 / 110e9, tolerance=0.001, label="net margin")
            
            # Gross margin: 40B / 110B ≈ 0.3636
            approx_eq(metrics.gross_margin, 40e9 / 110e9, tolerance=0.001, label="gross margin")
            
            # EBITDA margin: 31B / 110B ≈ 0.2818
            approx_eq(metrics.ebitda_margin, 31e9 / 110e9, tolerance=0.001, label="ebitda margin")
            
            # ROE: 19B / 130B ≈ 0.1462
            approx_eq(metrics.roe, 19e9 / 130e9, tolerance=0.001, label="roe")
            
            # ROA: 19B / 220B ≈ 0.0864
            approx_eq(metrics.roa, 19e9 / 220e9, tolerance=0.001, label="roa")
            
            r.ok("Profitability ratios computed correctly")
        except Exception as e:
            r.fail("profitability ratios", str(e))

        # ── 4. Solvency ratios ─────────────────────────────────────────────────
        try:
            # D/E: 50B / 130B ≈ 0.3846
            approx_eq(metrics.debt_to_equity, 50e9 / 130e9, tolerance=0.001, label="d/e")
            
            # D/A: 50B / 220B ≈ 0.2273
            approx_eq(metrics.debt_to_assets, 50e9 / 220e9, tolerance=0.001, label="d/a")
            
            # Interest coverage: 24B / 2.5B = 9.6x
            approx_eq(metrics.interest_coverage, 24e9 / 2.5e9, tolerance=0.1, label="interest cov")
            
            r.ok("Solvency ratios computed correctly")
        except Exception as e:
            r.fail("solvency ratios", str(e))

        # ── 5. Liquidity ratios ────────────────────────────────────────────────
        try:
            # Current ratio: 90B / 50B = 1.8x
            approx_eq(metrics.current_ratio, 90e9 / 50e9, tolerance=0.001, label="current ratio")
            r.ok("Liquidity ratios computed correctly")
        except Exception as e:
            r.fail("liquidity ratios", str(e))

        # ── 6. Cash flow metrics ───────────────────────────────────────────────
        try:
            # FCF / NI: 14B / 19B ≈ 0.7368
            approx_eq(metrics.fcf_to_net_income, 14e9 / 19e9, tolerance=0.001, label="fcf/ni")
            
            # Op CF / NI: 20B / 19B ≈ 1.0526
            approx_eq(metrics.operating_cf_to_net_income, 20e9 / 19e9, tolerance=0.001, label="op cf/ni")
            
            r.ok("Cash flow metrics computed correctly")
        except Exception as e:
            r.fail("cash flow metrics", str(e))

        # ── 7. YoY growth rates ────────────────────────────────────────────────
        try:
            # Revenue growth: (110B - 100B) / 100B = 0.10 (10%)
            approx_eq(metrics.revenue_growth_yoy, 0.10, tolerance=0.001, label="rev growth")
            
            # Net income growth: (19B - 20B) / 20B = -0.05 (-5%)
            approx_eq(metrics.net_income_growth_yoy, -0.05, tolerance=0.001, label="ni growth")
            
            # FCF growth: (14B - 17B) / 17B ≈ -0.176
            approx_eq(metrics.fcf_growth_yoy, (14e9 - 17e9) / 17e9, tolerance=0.001, label="fcf growth")
            
            r.ok("YoY growth rates computed correctly")
        except Exception as e:
            r.fail("yoy growth", str(e))

        # ── 8. Margin expansion ────────────────────────────────────────────────
        try:
            # Prior margin: 20B / 100B = 0.20 (20%)
            # Latest margin: 19B / 110B ≈ 0.1727 (17.27%)
            # Expansion: (0.1727 - 0.20) * 100 ≈ -2.73pp
            expected_expansion = (19e9 / 110e9 - 20e9 / 100e9) * 100
            approx_eq(metrics.margin_expansion, expected_expansion, tolerance=0.1, label="margin exp")
            r.ok("Margin expansion computed correctly")
        except Exception as e:
            r.fail("margin expansion", str(e))

        # ── 9. Debt change ────────────────────────────────────────────────────
        try:
            # Debt growth: (50B - 40B) / 40B = 0.25 (25%)
            approx_eq(metrics.debt_change_pct, 0.25, tolerance=0.001, label="debt change")
            r.ok("Debt change computed correctly")
        except Exception as e:
            r.fail("debt change", str(e))

        # ── 10. Anomaly detection ──────────────────────────────────────────────
        try:
            # FCF divergence: |14B - 19B| / |19B| ≈ 0.26 (26%) > 25% threshold → flagged
            # (Revenue +10%, NI -5%: not an earnings miss because NI declined, not up)
            # Margin compression: -2.73pp (not < -3pp threshold)
            # Debt spike: +25% > 20% threshold → flagged
            
            assert metrics.fcf_divergence == True, "FCF divergence not flagged"
            assert metrics.earnings_miss == False, "Earnings miss should not be flagged (NI down with rev up needs bigger decline)"
            assert metrics.margin_compression == False, "Margin compression not significant enough"
            assert metrics.debt_spike == True, "Debt spike not flagged"
            
            r.ok("Anomaly detection flags set correctly")
        except Exception as e:
            r.fail("anomaly detection", str(e))

        # ── 11. metrics_to_df conversion ───────────────────────────────────────
        try:
            df = fa.metrics_to_df(metrics)
            assert len(df) == 1, f"expected 1 row, got {len(df)}"
            assert df.iloc[0]["ticker"] == "TEST"
            assert df.iloc[0]["flag_fcf_divergence"] == True
            assert df.iloc[0]["flag_debt_spike"] == True
            r.ok("metrics_to_df creates proper single-row DataFrame")
        except Exception as e:
            r.fail("metrics_to_df", str(e))

        # ── 12. analyze_tickers (multiple) ─────────────────────────────────────
        try:
            # Add another ticker to the DB for testing
            conn = sqlite3.connect(db_path)
            conn.execute("""
                INSERT INTO income_statements VALUES
                ('TEST2', '2023-12-31', 'annual', 50e9, 20e9, 15e9, 12e9,
                 10e9, 0.75, 5e9, 12e9, 1e9, 2.5e9, ?)
            """, (datetime.utcnow().isoformat(),))
            conn.execute("""
                INSERT INTO balance_sheets VALUES
                ('TEST2', '2023-12-31', 'annual', 100e9, 40e9, 60e9, 10e9,
                 5e9, 20e9, 25e9, 40e9, 25e9, 3e9, 30e9, ?)
            """, (datetime.utcnow().isoformat(),))
            conn.execute("""
                INSERT INTO cash_flows VALUES
                ('TEST2', '2023-12-31', 'annual', 11e9, 3e9, 8e9,
                 1.5e9, 10e9, 2.5e9, 2e9, ?)
            """, (datetime.utcnow().isoformat(),))
            conn.commit()
            conn.close()
            
            df = fa.analyze_tickers(["TEST", "TEST2"], db_path)
            assert len(df) == 2, f"expected 2 rows, got {len(df)}"
            assert set(df["ticker"].values) == {"TEST", "TEST2"}
            r.ok("analyze_tickers handles multiple tickers")
        except Exception as e:
            r.fail("analyze_tickers", str(e))

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
    print("\nRunning financial analysis tests...\n")
    success = run_tests()
    sys.exit(0 if success else 1)