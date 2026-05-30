"""
financial_agent/test_report_generator.py

Tests the report generator: chart creation, HTML assembly, and file output.
No Claude API calls — agent output is mocked.

Usage:
    python -m financial_agent.test_report_generator
"""

import sys
import json
import sqlite3
import tempfile
import base64
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import financial_agent.report_generator as rg_module
from financial_agent import report_generator as rg
from financial_agent.financial_analysis import FinancialMetrics


# ── Mock data ──────────────────────────────────────────────────────────────────

MOCK_REPORT = {
    "ticker": "AAPL",
    "analysis_date": "2024-01-03",
    "executive_summary": (
        "Apple demonstrates exceptional cash generation with $101B in free cash flow "
        "and a 25.3% net margin, though revenue contracted 2.8% YoY signaling near-term "
        "growth headwinds."
    ),
    "strengths": [
        {"point": "Best-in-class free cash flow", "supporting_data": "FCF of $101B, FCF/NI ratio of 1.04x"},
        {"point": "Strong interest coverage", "supporting_data": "29.3x — debt comfortably serviceable"},
    ],
    "risks": [
        {"point": "Revenue contraction", "supporting_data": "Revenue declined 2.8% YoY to $383B"},
        {"point": "Negative book equity", "supporting_data": "Total equity of -$62B due to buybacks"},
    ],
    "anomalies": [
        {
            "flag": "FCF_DIVERGENCE",
            "severity": "MEDIUM",
            "interpretation": "FCF exceeds net income by >25%, indicating strong non-cash charges relative to earnings."
        }
    ],
    "recommendation": {
        "stance": "CAUTIOUS",
        "confidence": "MEDIUM",
        "rationale": "Strong cash generation and margins offset by declining revenue and elevated leverage.",
        "key_metrics_to_watch": ["revenue_growth_yoy", "gross_margin", "fcf_trend"]
    },
    "disclaimer": "For informational purposes only. Not financial advice."
}

MOCK_METRICS = FinancialMetrics(
    ticker="AAPL",
    latest_revenue=383e9,
    latest_net_income=97e9,
    latest_fcf=101e9,
    latest_total_debt=105e9,
    latest_total_equity=-62e9,
    latest_eps=6.13,
    latest_cash=30e9,
    latest_assets=352e9,
    net_margin=0.253,
    gross_margin=0.441,
    operating_margin=0.298,
    ebitda_margin=0.320,
    roe=-1.56,
    roa=0.275,
    debt_to_equity=-1.69,
    debt_to_assets=0.298,
    interest_coverage=29.3,
    current_ratio=0.99,
    fcf_to_net_income=1.04,
    operating_cf_to_net_income=1.17,
    revenue_growth_yoy=-0.028,
    net_income_growth_yoy=-0.029,
    fcf_growth_yoy=-0.091,
    margin_expansion=-0.4,
    debt_change_pct=0.082,
    fcf_divergence=True,
    earnings_miss=False,
    margin_compression=False,
    debt_spike=False,
)


def create_test_db(db_path: Path):
    """Create a minimal test DB with AAPL financials and price history."""
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
    for year, rev, ni, fcf in [
        ("2023-12-31", 383e9, 97e9, 101e9),
        ("2022-12-31", 394e9, 100e9, 111e9),
        ("2021-12-31", 366e9, 95e9,  93e9),
    ]:
        conn.execute(
            "INSERT INTO income_statements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", year, "annual", rev, 169e9, 123e9, 114e9,
             ni, 6.13, 30e9, 114e9, 3.9e9, 30e9, now)
        )
        conn.execute(
            "INSERT INTO balance_sheets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", year, "annual", 352e9, 290e9, -62e9, 30e9,
             10e9, 95e9, 105e9, 143e9, 145e9, 6e9, -214e9, now)
        )
        conn.execute(
            "INSERT INTO cash_flows VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", year, "annual", 113e9, 11e9, fcf,
             15e9, ni, 11e9, 77e9, now)
        )

    # 10 days of price history
    prices = [
        ("2024-01-10", 185.92), ("2024-01-09", 185.14),
        ("2024-01-08", 188.32), ("2024-01-05", 187.68),
        ("2024-01-04", 182.52), ("2024-01-03", 184.25),
        ("2024-01-02", 185.20), ("2023-12-29", 193.58),
        ("2023-12-28", 193.03), ("2023-12-27", 193.15),
    ]
    for date, close in prices:
        conn.execute(
            "INSERT INTO price_history VALUES (?,?,?,?,?,?,?)",
            ("AAPL", date, close - 1, close + 1, close - 2, close, 55_000_000)
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

        # Patch DB_PATH in the module
        rg_module.DB_PATH = db_path

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # ── 1. chart_price_history returns valid base64 PNG ────────────────────
        try:
            b64 = rg.chart_price_history("AAPL", conn)
            assert b64 is not None, "returned None"
            decoded = base64.b64decode(b64)
            assert decoded[:8] == b'\x89PNG\r\n\x1a\n', "not a valid PNG"
            assert len(decoded) > 5000, f"PNG too small ({len(decoded)} bytes)"
            r.ok("chart_price_history returns valid base64 PNG")
        except Exception as e:
            r.fail("chart_price_history", str(e))

        # ── 2. chart_revenue_income returns valid base64 PNG ──────────────────
        try:
            b64 = rg.chart_revenue_income("AAPL", conn)
            assert b64 is not None, "returned None"
            decoded = base64.b64decode(b64)
            assert decoded[:8] == b'\x89PNG\r\n\x1a\n', "not a valid PNG"
            r.ok("chart_revenue_income returns valid base64 PNG")
        except Exception as e:
            r.fail("chart_revenue_income", str(e))

        # ── 3. chart_margins returns valid base64 PNG ─────────────────────────
        try:
            b64 = rg.chart_margins("AAPL", conn)
            assert b64 is not None, "returned None"
            decoded = base64.b64decode(b64)
            assert decoded[:8] == b'\x89PNG\r\n\x1a\n', "not a valid PNG"
            r.ok("chart_margins returns valid base64 PNG")
        except Exception as e:
            r.fail("chart_margins", str(e))

        # ── 4. chart_fcf returns valid base64 PNG ────────────────────────────
        try:
            b64 = rg.chart_fcf("AAPL", conn)
            assert b64 is not None, "returned None"
            decoded = base64.b64decode(b64)
            assert decoded[:8] == b'\x89PNG\r\n\x1a\n', "not a valid PNG"
            r.ok("chart_fcf returns valid base64 PNG")
        except Exception as e:
            r.fail("chart_fcf", str(e))

        # ── 5. chart functions handle missing ticker gracefully ───────────────
        try:
            result = rg.chart_price_history("ZZZZ", conn)
            assert result is None, f"expected None for unknown ticker, got {type(result)}"
            r.ok("chart_price_history returns None for unknown ticker")
        except Exception as e:
            r.fail("chart graceful fallback", str(e))

        conn.close()

        # ── 6. build_html produces valid HTML structure ───────────────────────
        try:
            html = rg.build_html(
                "AAPL", MOCK_REPORT, MOCK_METRICS,
                img_price=None, img_rev=None,
                img_margins=None, img_fcf=None
            )
            assert "<!DOCTYPE html>" in html
            assert "<title>AAPL" in html
            assert "CAUTIOUS" in html
            assert "Apple demonstrates exceptional" in html
            assert "Best-in-class free cash flow" in html
            assert "Revenue contraction" in html
            assert "FCF_DIVERGENCE" in html
            assert "revenue_growth_yoy" in html
            r.ok("build_html produces complete HTML with all sections")
        except Exception as e:
            r.fail("build_html structure", str(e))

        # ── 7. build_html embeds metrics correctly ────────────────────────────
        try:
            html = rg.build_html(
                "AAPL", MOCK_REPORT, MOCK_METRICS,
                None, None, None, None
            )
            # Revenue ~383B
            assert "383" in html, "Revenue not found in HTML"
            # Net margin 25.3%
            assert "25.3%" in html, "Net margin not found"
            # Interest coverage
            assert "29.3x" in html, "Interest coverage not found"
            r.ok("build_html embeds metric values correctly")
        except Exception as e:
            r.fail("build_html metrics", str(e))

        # ── 8. build_html with charts embeds base64 data URIs ────────────────
        try:
            # Create a tiny 1x1 PNG base64 as a fake chart
            fake_b64 = base64.b64encode(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
                b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
                b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
                b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
            ).decode()

            html = rg.build_html(
                "AAPL", MOCK_REPORT, MOCK_METRICS,
                img_price=fake_b64, img_rev=fake_b64,
                img_margins=fake_b64, img_fcf=fake_b64
            )
            assert html.count("data:image/png;base64,") == 4, \
                f"Expected 4 chart embeds, found {html.count('data:image/png;base64,')}"
            r.ok("build_html embeds 4 charts as base64 data URIs")
        except Exception as e:
            r.fail("build_html chart embedding", str(e))

        # ── 9. build_html handles missing/empty sections gracefully ──────────
        try:
            empty_report = {
                "ticker": "TEST",
                "analysis_date": "2024-01-01",
                "executive_summary": "Test summary.",
                "strengths": [],
                "risks": [],
                "anomalies": [],
                "recommendation": {
                    "stance": "BULLISH",
                    "confidence": "HIGH",
                    "rationale": "Strong fundamentals.",
                    "key_metrics_to_watch": []
                },
                "disclaimer": "Not advice."
            }
            empty_metrics = FinancialMetrics(ticker="TEST")
            html = rg.build_html("TEST", empty_report, empty_metrics,
                                  None, None, None, None)
            assert "<!DOCTYPE html>" in html
            assert "No significant anomalies detected" in html
            r.ok("build_html handles empty strengths/risks/anomalies gracefully")
        except Exception as e:
            r.fail("build_html empty sections", str(e))

        # ── 10. generate_report writes file to disk ───────────────────────────
        try:
            from unittest.mock import patch

            out_path = Path(tmpdir) / "test_report.html"

            # Mock run_agent so we don't make real API calls
            with patch("financial_agent.report_generator.run_agent",
                       return_value=MOCK_REPORT):
                result_path = rg.generate_report(
                    "AAPL",
                    out_path=out_path,
                )

            assert result_path == out_path, f"Wrong path returned: {result_path}"
            assert out_path.exists(), "HTML file not written to disk"
            content = out_path.read_text(encoding="utf-8")
            assert len(content) > 10_000, f"File too small: {len(content)} chars"
            assert "<!DOCTYPE html>" in content
            assert "AAPL" in content
            r.ok("generate_report writes valid HTML file to disk")
        except Exception as e:
            r.fail("generate_report file output", str(e))

        # ── 11. _fmt helper handles None and values correctly ─────────────────
        try:
            assert rg._fmt(None) == "—"
            assert rg._fmt(25.312, suffix="%", decimals=1) == "25.3%"
            assert rg._fmt(1500.0, prefix="$", decimals=0) == "$1,500"
            assert rg._fmt(0.0, suffix="x") == "0.0x"
            r.ok("_fmt helper handles None, suffixes, and formatting correctly")
        except Exception as e:
            r.fail("_fmt helper", str(e))

        # ── 12. Stance and severity color helpers ─────────────────────────────
        try:
            assert rg._stance_color("BULLISH")  == rg.EMERALD
            assert rg._stance_color("BEARISH")  == rg.ROSE
            assert rg._stance_color("CAUTIOUS") == rg.AMBER
            assert rg._severity_color("HIGH")   == rg.ROSE
            assert rg._severity_color("MEDIUM") == rg.AMBER
            assert rg._severity_color("LOW")    == rg.EMERALD
            r.ok("Stance and severity color helpers return correct colors")
        except Exception as e:
            r.fail("color helpers", str(e))

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
    print("\nRunning report generator tests (no API calls)...\n")
    success = run_tests()
    sys.exit(0 if success else 1)