# Autonomous Financial Analysis Agent

An end-to-end AI system that fetches real company financials, computes 15+ financial ratios, detects anomalies, and generates structured investment reports — orchestrated by a Claude-powered reasoning agent with a live Streamlit UI.

Built to demonstrate agentic AI design patterns, production data engineering, and applied financial domain knowledge.

---

## What It Does

Type a ticker symbol. The agent:

1. Fetches income statements, balance sheets, cash flow statements, and price history from Alpha Vantage
2. Computes 15+ financial ratios and detects anomalies automatically
3. Runs a Claude claude-sonnet-4-6 reasoning agent that calls tools iteratively, then produces a structured investment report
4. Renders a polished HTML report with embedded charts — downloadable as a single file


```
═══════════════════════════════════════════════════════════════════
  📊 Financial Analysis: AAPL                          2026-05-30
═══════════════════════════════════════════════════════════════════

  EXECUTIVE SUMMARY
  Apple demonstrates exceptional cash generation with $98.8B in free
  cash flow and a 26.9% net margin. Revenue grew 6.4% YoY while net
  income surged 19.5%, reflecting meaningful margin expansion.

  ✅ STRENGTHS
    • Industry-leading profitability
      Data: Net margin 26.9%, operating margin 32.0% — well above
      S&P 500 median of ~10%
    • High-quality earnings
      Data: FCF/NI ratio of 0.88x confirms cash-backed income

  ⚠️  RISKS
    • Stock near 52-week high
      Data: $312.06 vs $312.51 high — limited valuation margin of safety

  📈 RECOMMENDATION: CAUTIOUS  (Confidence: MEDIUM)
  Exceptional fundamentals but valuation entry risk at current levels.
  Watch FCF recovery and revenue growth acceleration.

  👁  WATCH: fcf_growth_yoy, revenue_growth_yoy, current_ratio
═══════════════════════════════════════════════════════════════════
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               Streamlit UI  /  CLI                           │
│                      app.py                                  │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                      Agent Layer                             │
│                  financial_agent/agent.py                    │
│                                                              │
│   Claude claude-sonnet-4-6 + tool use (agentic loop)         │
│   Tools: get_financial_metrics · get_price_history           │
│          compare_companies · get_anomaly_flags               │
│   Output: structured JSON with fixed schema                  │
└────────┬──────────────────┬──────────────────────────────────┘
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌──────────────────────────────────────────┐
│  Report Layer   │  │           Analysis Engine                │
│  report_        │  │     financial_agent/financial_analysis.py│
│  generator.py   │  │                                          │
│                 │  │  15+ ratios: net margin, ROE, D/E,       │
│  4 matplotlib   │  │  current ratio, FCF/NI, interest cov.    │
│  charts → HTML  │  │  YoY trends · anomaly detection          │
└─────────────────┘  └──────────────────┬───────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────────┐
                     │              Data Layer                  │
                     │    financial_agent/data_ingestion.py     │
                     │                                          │
                     │  Alpha Vantage API → SQLite              │
                     │  90-day cache · raw JSON preserved       │
                     │  Income · Balance · Cash Flow · Prices   │
                     └──────────────────┬───────────────────────┘
                                        │
                                        ▼
                               financial_data.db
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI / Reasoning | Claude API (`claude-sonnet-4-6`) with tool use |
| Financial Data | Alpha Vantage (free tier — 25 calls/day) |
| Data Processing | Python, pandas |
| Storage | SQLite with 90-day cache |
| Visualization | matplotlib (4 embedded charts) |
| UI | Streamlit |
| Deploy | Streamlit Cloud (free) |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/yourusername/FinancialAnalysisAgent.git
cd FinancialAnalysisAgent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get API keys (both free)

| Service | Sign up | Free tier |
|---|---|---|
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use (~$0.01 per report) |
| Alpha Vantage | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | 25 API calls/day |

### 3. Configure environment

```bash
cp .env.example .env
# Add your keys to .env
```

Your `.env` should contain:
```
ANTHROPIC_API_KEY=sk-ant-...
ALPHA_VANTAGE_API_KEY=your-key-here
```

---

## Usage

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter any ticker and hit Analyze.

### CLI — fetch data

```bash
# Single ticker (~4 of your 25 daily API calls)
python3 -m financial_agent.data_ingestion --ticker AAPL

# Multiple tickers
python3 -m financial_agent.data_ingestion --ticker AAPL MSFT GOOGL

# Force re-fetch (bypass 90-day cache)
python3 -m financial_agent.data_ingestion --ticker AAPL --force-refresh
```

### CLI — run the agent

```bash
# Formatted terminal report
python3 -m financial_agent.agent --ticker AAPL

# Compare two companies
python3 -m financial_agent.agent --ticker AAPL --vs MSFT

# Raw JSON output
python3 -m financial_agent.agent --ticker AAPL --json
```

### CLI — generate HTML report

```bash
# Creates reports/AAPL_<date>.html
python3 -m financial_agent.report_generator --ticker AAPL

# With comparison
python3 -m financial_agent.report_generator --ticker AAPL --vs MSFT
```

### Run all tests

```bash
python3 -m financial_agent.test_ingestion              # 10/10
python3 -m financial_agent.test_financial_analysis     # 12/12
python3 -m financial_agent.test_agent                  # 8/8
python3 -m financial_agent.test_report_generator       # 12/12
```

---

## Financial Metrics

**Profitability** — net margin, gross margin, operating margin, EBITDA margin, ROE, ROA

**Solvency** — debt-to-equity, debt-to-assets, interest coverage ratio

**Liquidity** — current ratio

**Cash Flow Quality** — FCF/net income ratio, operating CF/net income ratio

**Growth (YoY)** — revenue, net income, FCF, margin expansion (pp), debt trajectory

**Anomaly Detection**

| Flag | Trigger | Severity |
|---|---|---|
| `FCF_DIVERGENCE` | \|FCF − Net Income\| > 25% of Net Income | MEDIUM |
| `EARNINGS_MISS` | Revenue +5% YoY but Net Income −20% YoY | HIGH |
| `MARGIN_COMPRESSION` | Net margin fell 3+ percentage points | MEDIUM |
| `DEBT_SPIKE` | Total debt grew 20%+ YoY | MEDIUM |

---

## Agent Design

The Claude agent uses **tool use** (function calling) to reason over financial data before generating its report — the same pattern used in production agentic systems.

**Agentic loop:**
1. Claude receives a user query (e.g. "Analyze AAPL")
2. Claude decides which tools to call and in what order
3. Tool results are fed back into the conversation
4. Claude reasons over accumulated data and produces a structured JSON report

**Why this matters:** Claude never sees raw data upfront. It actively retrieves what it needs, which means the reasoning is grounded in verified numbers from each tool call. Every claim in the output cites a specific data point — no hallucinated figures.

**Tools available to the agent:**

| Tool | What it returns |
|---|---|
| `get_financial_metrics` | All computed ratios, trends, and anomaly flags |
| `get_price_history` | 52-week range, latest close, recent price sample |
| `compare_companies` | Side-by-side ratio comparison across tickers |
| `get_anomaly_flags` | Focused risk flag summary with severity and detail |

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **New app** → select repo → set main file to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ALPHA_VANTAGE_API_KEY = "your-key-here"
   ```
5. Deploy — you get a permanent public URL

---