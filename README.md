# 📊 Autonomous Financial Analysis Agent

An AI-powered financial analysis system that fetches real company data, computes key financial ratios, detects anomalies, and generates structured investment reports — all orchestrated by a Claude-powered reasoning agent.

Built as a portfolio project demonstrating **agentic AI design patterns**, **data engineering**, and **financial domain knowledge**.

---

## Demo

```
python3 -m financial_agent.agent --ticker AAPL
```

```
═══════════════════════════════════════════════════════════════════
  📊 Financial Analysis: AAPL
  2025-01-15
═══════════════════════════════════════════════════════════════════

  EXECUTIVE SUMMARY
  Apple demonstrates exceptional cash generation and profitability,
  with a 26.4% net margin and $101B in free cash flow. However,
  revenue contracted 2.8% YoY, signaling near-term growth headwinds.

  ✅ STRENGTHS
    • Best-in-class free cash flow generation
      Data: FCF of $101B, FCF/NI ratio of 1.04x indicates high earnings quality
    • Strong interest coverage
      Data: Interest coverage of 29.3x — debt is comfortably serviceable

  ⚠️  RISKS
    • Revenue contraction
      Data: Revenue declined 2.8% YoY from $394B to $383B

  📈 RECOMMENDATION: CAUTIOUS (Confidence: MEDIUM)
  Strong fundamentals and cash generation offset by declining revenue.
  Watch for return to growth before turning bullish.

  👁  WATCH: revenue_growth_yoy, fcf_trend, gross_margin
═══════════════════════════════════════════════════════════════════
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User / CLI                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer                              │
│              (financial_agent/agent.py)                     │
│                                                             │
│  • Claude claude-sonnet-4-6 with tool use                   │
│  • Agentic loop: calls tools → reasons → generates report   │
│  • Structured JSON output with fixed schema                 │
└──────┬────────────────┬───────────────┬────────────────────┘
       │                │               │
       ▼                ▼               ▼
  get_metrics    get_prices    compare_companies
  get_anomalies
       │                │               │
       └────────────────┴───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Analysis Engine                            │
│           (financial_agent/financial_analysis.py)           │
│                                                             │
│  • 15+ financial ratios (P/E, D/E, ROE, FCF yield, etc.)   │
│  • YoY trend computation                                    │
│  • Anomaly detection (FCF divergence, debt spike, etc.)     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
│            (financial_agent/data_ingestion.py)              │
│                                                             │
│  • Alpha Vantage API (income statement, balance sheet,      │
│    cash flow, price history)                                │
│  • SQLite storage with 90-day cache (free tier safe)        │
│  • Raw JSON preservation for reprocessing                   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
              financial_data.db (SQLite)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI / Reasoning | Claude API (`claude-sonnet-4-6`) with tool use |
| Financial Data | Alpha Vantage (free tier) |
| Data Processing | Python, pandas |
| Storage | SQLite (zero cost, local) |
| Visualization | matplotlib, plotly *(Day 4)* |
| UI / Demo | Streamlit *(Day 5)* |

---

## Project Structure

```
FinancialAnalysisAgent/
├── financial_agent/
│   ├── __init__.py
│   ├── data_ingestion.py        # Alpha Vantage API → SQLite pipeline
│   ├── financial_analysis.py    # Ratio computation + anomaly detection
│   ├── agent.py                 # Claude agent with tool use
│   ├── test_ingestion.py        # Tests for data layer (10/10)
│   ├── test_financial_analysis.py  # Tests for analysis engine (12/12)
│   └── test_agent.py            # Tests for agent tools (8/8)
├── financial_data.db            # SQLite database (gitignored)
├── .env                         # API keys (gitignored)
├── .env.example                 # Key template (committed)
├── requirements.txt
└── README.md
```

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

| Service | Where to get it | Free tier |
|---------|----------------|-----------|
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use |
| Alpha Vantage | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | 25 calls/day |

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your real keys
```

Also set your Alpha Vantage key in `financial_agent/data_ingestion.py`:

```python
API_KEY = "your-alpha-vantage-key-here"
```

---

## Usage

### Fetch financial data

```bash
# Fetch a single ticker (~4 of your 25 daily API calls)
python3 -m financial_agent.data_ingestion --ticker AAPL

# Fetch multiple tickers
python3 -m financial_agent.data_ingestion --ticker AAPL MSFT GOOGL

# Force re-fetch (bypass cache)
python3 -m financial_agent.data_ingestion --ticker AAPL --force-refresh
```

### Run the analysis engine

```bash
# View computed ratios for a ticker
python3 -m financial_agent.financial_analysis --ticker AAPL

# Export to CSV
python3 -m financial_agent.financial_analysis --ticker AAPL MSFT --csv analysis.csv
```

### Run the AI agent

```bash
# Formatted report
python3 -m financial_agent.agent --ticker AAPL

# Compare two companies
python3 -m financial_agent.agent --ticker AAPL --vs MSFT

# Raw JSON output
python3 -m financial_agent.agent --ticker AAPL --json
```

### Run the test suite

```bash
python3 -m financial_agent.test_ingestion          # 10/10
python3 -m financial_agent.test_financial_analysis # 12/12
python3 -m financial_agent.test_agent              # 8/8
```

---

## Financial Metrics Computed

**Profitability**
- Net margin, gross margin, operating margin, EBITDA margin
- Return on equity (ROE), return on assets (ROA)

**Solvency**
- Debt-to-equity, debt-to-assets
- Interest coverage ratio

**Liquidity**
- Current ratio

**Cash Flow Quality**
- FCF-to-net-income ratio (earnings quality signal)
- Operating CF-to-net-income ratio

**Growth (YoY)**
- Revenue growth, net income growth, FCF growth
- Margin expansion/compression (percentage points)
- Debt trajectory

**Anomaly Flags**
| Flag | Trigger | Severity |
|------|---------|---------|
| `FCF_DIVERGENCE` | \|FCF - Net Income\| > 25% of Net Income | MEDIUM |
| `EARNINGS_MISS` | Revenue +5% but Net Income -20% | HIGH |
| `MARGIN_COMPRESSION` | Net margin fell 3+ percentage points | MEDIUM |
| `DEBT_SPIKE` | Total debt grew 20%+ YoY | MEDIUM |

---

## Agent Design

The Claude agent uses **tool use** (function calling) to reason over financial data before generating its report. This mirrors production agentic architectures:

1. **Tool selection** — Claude decides which tools to call and in what order
2. **Iterative reasoning** — results from each tool inform subsequent calls
3. **Structured output** — fixed JSON schema enforced via system prompt
4. **Confidence calibration** — HIGH/MEDIUM/LOW confidence based on signal consistency

The system prompt enforces that every claim in the report is grounded in a specific data point from a tool result — no hallucinated numbers.

---

## Roadmap

- [x] Day 1 — Data ingestion layer (Alpha Vantage → SQLite)
- [x] Day 2 — Financial analysis engine (ratios + anomaly detection)
- [x] Day 3 — Claude agent with tool use
- [ ] Day 4 — HTML report with matplotlib/plotly charts
- [ ] Day 5 — Streamlit UI + deployment

---

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own due diligence before making investment decisions.