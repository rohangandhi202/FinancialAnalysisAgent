"""
app.py  ·  Autonomous Financial Analysis Agent — Streamlit UI

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect repo on share.streamlit.io → set secrets
"""

import streamlit as st
import sqlite3
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Financial Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Environment & secrets ─────────────────────────────────────────────────────
# On Streamlit Cloud, keys live in st.secrets.
# Locally they come from .env (loaded by dotenv in the submodules).
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
if "ALPHA_VANTAGE_API_KEY" in st.secrets:
    os.environ["ALPHA_VANTAGE_API_KEY"] = st.secrets["ALPHA_VANTAGE_API_KEY"]

# Import after env is set so dotenv picks up local keys too
from dotenv import load_dotenv
load_dotenv()

from financial_agent.data_ingestion import init_db, ingest_ticker
from financial_agent.financial_analysis import analyze_ticker
from financial_agent.agent import run_agent
from financial_agent.report_generator import (
    chart_price_history,
    chart_revenue_income,
    chart_margins,
    chart_fcf,
    build_html,
)

import matplotlib
matplotlib.use("Agg")

# ── DB path: use a temp dir on Streamlit Cloud (read-only filesystem) ─────────
if "STREAMLIT_SHARING_MODE" in os.environ or not Path("financial_data.db").parent.exists():
    _db_dir = tempfile.mkdtemp()
    DB_PATH = Path(_db_dir) / "financial_data.db"
else:
    DB_PATH = Path("financial_data.db")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace !important;
}
.stApp {
    background: #0a0e1a;
    color: #f9fafb;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* ── Header ── */
.agent-header {
    border-bottom: 1px solid #1f2937;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.agent-eyebrow {
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #f59e0b;
    margin-bottom: 0.5rem;
}
.agent-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.1;
    color: #f9fafb;
    margin: 0;
}
.agent-sub {
    font-size: 12px;
    color: #6b7280;
    margin-top: 0.5rem;
}

/* ── Input card ── */
.input-card {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 1.5rem;
    margin-bottom: 2rem;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 1rem 1.1rem;
}
.metric-tile .label {
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 6px;
}
.metric-tile .value {
    font-size: 1.5rem;
    font-weight: 500;
    color: #f9fafb;
    letter-spacing: -0.02em;
}
.metric-tile .sub {
    font-size: 10px;
    color: #9ca3af;
    margin-top: 3px;
}

/* ── Stance badge ── */
.stance-bullish  { color: #10b981; border: 1.5px solid #10b981; }
.stance-cautious { color: #f59e0b; border: 1.5px solid #f59e0b; }
.stance-bearish  { color: #f43f5e; border: 1.5px solid #f43f5e; }
.stance-badge {
    display: inline-block;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.4rem 1.1rem;
    font-weight: 500;
}

/* ── Finding cards ── */
.finding-strength {
    border-left: 2px solid #10b981;
    background: rgba(16,185,129,0.05);
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
}
.finding-risk {
    border-left: 2px solid #f43f5e;
    background: rgba(244,63,94,0.05);
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
}
.finding-title { font-weight: 500; font-size: 13px; color: #f9fafb; }
.finding-data  { font-size: 11px; color: #9ca3af; margin-top: 3px; }

/* ── Anomaly cards ── */
.anomaly-high   { border-left: 3px solid #f43f5e; }
.anomaly-medium { border-left: 3px solid #f59e0b; }
.anomaly-low    { border-left: 3px solid #10b981; }
.anomaly-card {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.anomaly-badge-high   { background: rgba(244,63,94,0.15);  color: #f43f5e; }
.anomaly-badge-medium { background: rgba(245,158,11,0.15); color: #f59e0b; }
.anomaly-badge-low    { background: rgba(16,185,129,0.15); color: #10b981; }
.anomaly-badge {
    font-size: 9px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 8px;
    font-weight: 500;
    border-radius: 2px;
    margin-right: 8px;
}

/* ── Section headers ── */
.section-label {
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4b5563;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stTextInput"] input {
    background: #0a0e1a !important;
    border: 1px solid #374151 !important;
    color: #f9fafb !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 0 !important;
    font-size: 14px !important;
}
div[data-testid="stButton"] button {
    background: #f59e0b !important;
    color: #0a0e1a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 0 !important;
    letter-spacing: 0.06em !important;
    font-size: 12px !important;
    padding: 0.6rem 1.5rem !important;
    text-transform: uppercase !important;
}
div[data-testid="stButton"] button:hover {
    background: #d97706 !important;
}
div[data-testid="stSelectbox"] > div {
    background: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 0 !important;
    color: #f9fafb !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1f2937 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #f59e0b !important;
    border-bottom: 2px solid #f59e0b !important;
    background: transparent !important;
}
.stSpinner > div { border-top-color: #f59e0b !important; }
div[data-testid="stImage"] img { width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="agent-header">
    <div class="agent-eyebrow">Autonomous AI · Financial Analysis</div>
    <div class="agent-title">Financial Analysis Agent</div>
    <div class="agent-sub">
        Powered by Claude claude-sonnet-4-6 · Alpha Vantage · Real company data
    </div>
</div>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(val, prefix="", suffix="", decimals=1):
    if val is None:
        return "—"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def stance_badge(stance: str) -> str:
    cls = f"stance-{stance.lower()}"
    return f'<span class="stance-badge {cls}">{stance}</span>'


def severity_cls(s: str) -> str:
    return s.lower() if s.lower() in ("high", "medium", "low") else "medium"


def check_api_keys() -> tuple[bool, str]:
    ak = os.getenv("ANTHROPIC_API_KEY", "")
    av = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not ak:
        return False, "ANTHROPIC_API_KEY not set — add it to your .env file or Streamlit secrets."
    if not av:
        return False, "ALPHA_VANTAGE_API_KEY not set — add it to your .env file or Streamlit secrets."
    return True, ""


# ── Input panel ───────────────────────────────────────────────────────────────
keys_ok, keys_err = check_api_keys()
if not keys_ok:
    st.error(f"⚠️ {keys_err}")
    st.stop()

col_input, col_compare, col_btn = st.columns([3, 2, 1])

with col_input:
    ticker_raw = st.text_input(
        "Ticker symbol",
        placeholder="AAPL",
        label_visibility="collapsed",
    )

with col_compare:
    compare_raw = st.text_input(
        "Compare with (optional)",
        placeholder="MSFT  (optional)",
        label_visibility="collapsed",
    )

with col_btn:
    run_btn = st.button("Analyze →")

# Popular tickers as quick-launch pills
st.markdown(
    "<div style='font-size:10px;color:#4b5563;letter-spacing:0.1em;"
    "text-transform:uppercase;margin-bottom:6px'>Quick launch</div>",
    unsafe_allow_html=True
)
pill_cols = st.columns(8)
quick_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]
for i, qt in enumerate(quick_tickers):
    with pill_cols[i]:
        if st.button(qt, key=f"pill_{qt}"):
            ticker_raw = qt
            run_btn = True

st.markdown("<div style='border-bottom:1px solid #1f2937;margin:1.5rem 0'></div>",
            unsafe_allow_html=True)


# ── Main analysis flow ────────────────────────────────────────────────────────
if run_btn and ticker_raw.strip():
    ticker = ticker_raw.strip().upper()
    compare_with = [c.strip().upper() for c in compare_raw.split()
                    if c.strip()] if compare_raw.strip() else None

    # ── Step 1: Ingest data ───────────────────────────────────────────────────
    with st.spinner(f"Fetching financial data for {ticker}..."):
        try:
            conn = init_db(DB_PATH)
            conn.close()
            result = ingest_ticker(ticker, init_db(DB_PATH))
            if compare_with:
                for ct in compare_with:
                    ingest_ticker(ct, init_db(DB_PATH))
        except Exception as e:
            st.error(f"Data ingestion failed: {e}")
            st.stop()

    # ── Step 2: Run the agent ─────────────────────────────────────────────────
    with st.spinner("Running AI analysis... (this takes 20–40 seconds)"):
        try:
            # Temporarily set DB_PATH in agent and report_generator modules
            import financial_agent.agent as agent_mod
            import financial_agent.report_generator as rg_mod
            agent_mod.DB_PATH = DB_PATH
            rg_mod.DB_PATH   = DB_PATH

            report = run_agent(ticker, compare_with=compare_with)
            if "error" in report:
                st.error(f"Agent error: {report['error']}")
                if "raw" in report:
                    with st.expander("Raw output"):
                        st.text(report["raw"])
                st.stop()
        except Exception as e:
            st.error(f"Agent failed: {e}")
            st.stop()

    # ── Step 3: Compute metrics & charts ─────────────────────────────────────
    with st.spinner("Generating charts..."):
        metrics = analyze_ticker(ticker, DB_PATH)
        db_conn = init_db(DB_PATH)
        db_conn.row_factory = sqlite3.Row

        img_price   = chart_price_history(ticker, db_conn)
        img_rev     = chart_revenue_income(ticker, db_conn)
        img_margins = chart_margins(ticker, db_conn)
        img_fcf     = chart_fcf(ticker, db_conn)
        db_conn.close()

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    rec    = report.get("recommendation", {})
    stance = rec.get("stance", "CAUTIOUS")
    conf   = rec.get("confidence", "MEDIUM")

    # ── Report header ─────────────────────────────────────────────────────────
    h_left, h_right = st.columns([3, 1])
    with h_left:
        st.markdown(
            f"<div style='font-family:Playfair Display,serif;font-size:2.2rem;"
            f"font-weight:700;color:#f9fafb;letter-spacing:-0.02em'>{ticker}</div>"
            f"<div style='font-size:11px;color:#6b7280;margin-top:4px'>"
            f"Analysis generated {datetime.now().strftime('%B %d, %Y · %H:%M')}</div>",
            unsafe_allow_html=True
        )
    with h_right:
        st.markdown(
            f"<div style='text-align:right;padding-top:0.5rem'>"
            f"{stance_badge(stance)}"
            f"<div style='font-size:10px;color:#6b7280;margin-top:6px'>"
            f"Confidence: {conf}</div></div>",
            unsafe_allow_html=True
        )

    # ── Executive summary ─────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#111827;border:1px solid #1f2937;"
        f"border-left:3px solid #f59e0b;padding:1.25rem 1.5rem;margin:1.5rem 0;"
        f"font-family:Playfair Display,serif;font-size:15px;line-height:1.75;"
        f"color:#e5e7eb'>{report.get('executive_summary','')}</div>",
        unsafe_allow_html=True
    )

    # ── Key metrics grid ──────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Key Metrics</div>", unsafe_allow_html=True)
    m = metrics
    tiles = [
        ("Revenue",      f"${m.latest_revenue/1e9:,.1f}B"   if m.latest_revenue   else "—", ""),
        ("Net Income",   f"${m.latest_net_income/1e9:,.1f}B" if m.latest_net_income else "—", ""),
        ("Free Cash Flow", f"${m.latest_fcf/1e9:,.1f}B"     if m.latest_fcf        else "—", ""),
        ("Net Margin",   fmt(m.net_margin, suffix="%", decimals=1) if m.net_margin else "—",
            f"{'▲' if (m.margin_expansion or 0) > 0 else '▼'} {abs(m.margin_expansion or 0):.1f}pp YoY"
            if m.margin_expansion else ""),
        ("D/E Ratio",    fmt(m.debt_to_equity, suffix="x")   if m.debt_to_equity   else "—", ""),
        ("Interest Cov.", fmt(m.interest_coverage, suffix="x") if m.interest_coverage else "—", ""),
        ("Current Ratio", fmt(m.current_ratio, suffix="x")   if m.current_ratio    else "—", ""),
        ("FCF / NI",     fmt(m.fcf_to_net_income, suffix="x") if m.fcf_to_net_income else "—",
            "Earnings quality"),
        ("Revenue Growth", fmt(m.revenue_growth_yoy, prefix="+", suffix="%") if (m.revenue_growth_yoy or 0) >= 0
            else fmt(m.revenue_growth_yoy, suffix="%") if m.revenue_growth_yoy else "—", "YoY"),
        ("ROE",          fmt(m.roe, suffix="%")               if m.roe              else "—", ""),
        ("ROA",          fmt(m.roa, suffix="%")               if m.roa              else "—", ""),
        ("Total Debt",   f"${m.latest_total_debt/1e9:,.1f}B" if m.latest_total_debt else "—", ""),
    ]

    tile_cols = st.columns(6)
    for i, (label, value, sub) in enumerate(tiles):
        with tile_cols[i % 6]:
            st.markdown(
                f"<div class='metric-tile'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value'>{value}</div>"
                f"{'<div class=sub>' + sub + '</div>' if sub else ''}"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── Tabs: Charts / Strengths & Risks / Anomalies / Recommendation ────────
    st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
    tab_charts, tab_findings, tab_anomalies, tab_rec, tab_download = st.tabs([
        "Charts", "Strengths & Risks", "Anomalies", "Recommendation", "Download Report"
    ])

    # ── Tab 1: Charts ─────────────────────────────────────────────────────────
    with tab_charts:
        import base64, io
        from PIL import Image

        def b64_to_image(b64str):
            if not b64str:
                return None
            return Image.open(io.BytesIO(base64.b64decode(b64str)))

        c1, c2 = st.columns(2)
        with c1:
            img = b64_to_image(img_price)
            if img:
                st.image(img, use_container_width=True)
            img = b64_to_image(img_margins)
            if img:
                st.image(img, use_container_width=True)
        with c2:
            img = b64_to_image(img_rev)
            if img:
                st.image(img, use_container_width=True)
            img = b64_to_image(img_fcf)
            if img:
                st.image(img, use_container_width=True)

    # ── Tab 2: Strengths & Risks ──────────────────────────────────────────────
    with tab_findings:
        f_left, f_right = st.columns(2)

        with f_left:
            st.markdown("<div class='section-label'>Strengths</div>",
                        unsafe_allow_html=True)
            strengths = report.get("strengths", [])
            if strengths:
                for s in strengths:
                    st.markdown(
                        f"<div class='finding-strength'>"
                        f"<div class='finding-title'>{s.get('point','')}</div>"
                        f"<div class='finding-data'>{s.get('supporting_data','')}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<p style='color:#6b7280;font-size:12px'>None identified</p>",
                            unsafe_allow_html=True)

        with f_right:
            st.markdown("<div class='section-label'>Risks</div>",
                        unsafe_allow_html=True)
            risks = report.get("risks", [])
            if risks:
                for r in risks:
                    st.markdown(
                        f"<div class='finding-risk'>"
                        f"<div class='finding-title'>{r.get('point','')}</div>"
                        f"<div class='finding-data'>{r.get('supporting_data','')}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<p style='color:#6b7280;font-size:12px'>None identified</p>",
                            unsafe_allow_html=True)

    # ── Tab 3: Anomalies ──────────────────────────────────────────────────────
    with tab_anomalies:
        st.markdown("<div class='section-label'>Anomaly Flags</div>",
                    unsafe_allow_html=True)
        anomalies = report.get("anomalies", [])
        if anomalies:
            for a in anomalies:
                sev = severity_cls(a.get("severity", "medium"))
                st.markdown(
                    f"<div class='anomaly-card anomaly-{sev}'>"
                    f"<span class='anomaly-badge anomaly-badge-{sev}'>{a.get('severity','—')}</span>"
                    f"<strong style='font-size:13px'>{a.get('flag','—')}</strong>"
                    f"<p style='font-size:12px;color:#9ca3af;margin-top:6px;margin-bottom:0'>"
                    f"{a.get('interpretation','')}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<p style='color:#10b981;font-size:13px'>✓ No significant anomalies detected</p>",
                unsafe_allow_html=True
            )

    # ── Tab 4: Recommendation ─────────────────────────────────────────────────
    with tab_rec:
        sc = {"BULLISH": "#10b981", "BEARISH": "#f43f5e"}.get(stance, "#f59e0b")
        st.markdown(
            f"<div style='background:#111827;border:1px solid #1f2937;"
            f"border-left:3px solid {sc};padding:1.5rem;margin-bottom:1rem'>"
            f"<div style='font-family:Playfair Display,serif;font-size:1.6rem;"
            f"color:{sc};font-weight:700;margin-bottom:0.5rem'>{stance}</div>"
            f"<div style='font-size:10px;color:#6b7280;border:1px solid #1f2937;"
            f"display:inline-block;padding:2px 8px;margin-bottom:1rem'>"
            f"Confidence: {conf}</div>"
            f"<p style='color:#d1d5db;font-size:13px;line-height:1.75;margin-bottom:1rem'>"
            f"{rec.get('rationale','')}</p>",
            unsafe_allow_html=True
        )
        watch = rec.get("key_metrics_to_watch", [])
        if watch:
            st.markdown(
                "<div style='font-size:10px;color:#6b7280;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-bottom:0.5rem'>Key metrics to watch</div>",
                unsafe_allow_html=True
            )
            pills_html = "".join(
                f"<span style='font-size:10px;padding:3px 10px;border:1px solid #1f2937;"
                f"color:#9ca3af;display:inline-block;margin:2px'>{w}</span>"
                for w in watch
            )
            st.markdown(f"<div>{pills_html}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"<p style='font-size:10px;color:#374151;margin-top:1rem'>"
            f"{report.get('disclaimer','')}</p>",
            unsafe_allow_html=True
        )

    # ── Tab 5: Download ───────────────────────────────────────────────────────
    with tab_download:
        st.markdown(
            "<div class='section-label'>Download Full Report</div>",
            unsafe_allow_html=True
        )
        html_report = build_html(
            ticker, report, metrics,
            img_price, img_rev, img_margins, img_fcf
        )
        date_str = datetime.now().strftime("%Y%m%d")
        st.download_button(
            label="Download HTML Report →",
            data=html_report.encode("utf-8"),
            file_name=f"{ticker}_{date_str}_analysis.html",
            mime="text/html",
        )
        st.markdown(
            "<p style='font-size:11px;color:#6b7280;margin-top:0.75rem'>"
            "Self-contained HTML file — open in any browser, no internet required.</p>",
            unsafe_allow_html=True
        )

elif run_btn and not ticker_raw.strip():
    st.warning("Enter a ticker symbol first.")

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="border:1px dashed #1f2937;padding:3rem;text-align:center;margin-top:1rem">
        <div style="font-size:10px;letter-spacing:0.14em;text-transform:uppercase;
                    color:#374151;margin-bottom:1rem">How it works</div>
        <div style="display:flex;justify-content:center;gap:3rem;flex-wrap:wrap">
            <div style="text-align:center">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">01</div>
                <div style="font-size:11px;color:#6b7280">Enter a ticker<br>symbol above</div>
            </div>
            <div style="color:#1f2937;font-size:1.5rem;align-self:center">→</div>
            <div style="text-align:center">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">02</div>
                <div style="font-size:11px;color:#6b7280">Agent fetches real<br>financial data</div>
            </div>
            <div style="color:#1f2937;font-size:1.5rem;align-self:center">→</div>
            <div style="text-align:center">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">03</div>
                <div style="font-size:11px;color:#6b7280">Claude reasons over<br>ratios & trends</div>
            </div>
            <div style="color:#1f2937;font-size:1.5rem;align-self:center">→</div>
            <div style="text-align:center">
                <div style="font-size:1.5rem;margin-bottom:0.5rem">04</div>
                <div style="font-size:11px;color:#6b7280">Structured report<br>with charts</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)