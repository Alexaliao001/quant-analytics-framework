# Quantitative Analytics Framework

A production-grade Python framework for building reliable, self-evolving quantitative analysis systems.

Born from real-world needs: when you're fetching data from 25+ API sources in parallel, running Bayesian calibrations against 12,000+ signals, and monitoring live positions around the clock — you need infrastructure that doesn't break at 3 AM.

## What's Inside

### `core/` — API Reliability Layer

The foundation everything else builds on. Solves the unglamorous but critical problems of working with rate-limited REST APIs at scale.

```python
from core import ReliableAPIClient

client = ReliableAPIClient(
    base_url="https://api.polygon.io",
    api_key=os.environ["POLYGON_API_KEY"],
    max_connections=10,     # HTTP connection pool size
    max_concurrent=8,       # Semaphore-limited concurrent requests
    max_retries=3,          # Exponential backoff retry
)

# Safe to call from 20 ThreadPoolExecutor workers simultaneously
# Semaphore ensures only 8 are in-flight at once
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(client.get, f"/v2/aggs/ticker/{t}/range/1/day/2024-01-01/2024-12-31"): t
               for t in tickers}
    for f in as_completed(futures):
        result = f.result()  # Never throws — returns {"_failed": True, "error": "..."} on failure

# After batch: check health
stats = client.get_stats()
print(f"Calls: {stats['total_calls']}, Error rate: {stats['error_rate']:.1%}")
```

**Key design decisions:**
- Semaphore released BEFORE backoff sleep (don't hold connection slots while waiting)
- Never throws on API failure — returns structured error dict (caller decides policy)
- Thread-safe call counting enables monitoring without external instrumentation

Also includes:
- **`MultiFallbackResolver`** — Priority-ordered resolution chain (e.g., ticker → sector mapping with 4-layer fallback)
- **`DataQualityTracker`** — Batch-level data quality monitoring (high/medium/low reliability scoring)

### `calibration/` — Bayesian Signal Calibration

Adaptive weight optimization using Bayesian shrinkage estimation. Blends backtest priors with live outcomes — critical for any system that needs to evolve without overfitting.

```python
from calibration import BayesianCalibrator

priors = {
    ("STRONG_BUY", "bull"):  {"weight": 0.16, "n_backtest": 1077, "ev": 2.70, "win_pct": 0.616},
    ("BUY", "bull"):         {"weight": 0.12, "n_backtest": 2448, "ev": 2.98, "win_pct": 0.585},
}

calibrator = BayesianCalibrator(priors)
results = calibrator.update({
    ("STRONG_BUY", "bull"): [0.05, -0.02, 0.08, 0.03, -0.01, 0.06],
})

for cell_key, cell in results["cells"].items():
    print(f"{cell_key}: {cell['prior_weight']:.3f} → {cell['posterior_weight']:.3f} "
          f"(λ={cell['lambda']:.2f}, {cell['alert_level']})")
```

**Key components:**
- **`BayesianCalibrator`** — Shrinkage estimator: `posterior = λ × live + (1-λ) × prior`, where λ adapts to sample size
- **`compute_wilson_ci()`** — Wilson score confidence intervals (more accurate than normal approximation for small samples)
- **`compute_kelly_weight()`** — Half-Kelly position sizing from observed returns
- **Drift detection** — 4-tier alert system (OK → WATCH → REVIEW → ALERT) with configurable thresholds

### `risk/` — Portfolio Risk Analysis

9-dimensional risk model with estimated Greeks (no Black-Scholes dependency required).

```python
from risk import PortfolioRiskAnalyzer, Position

positions = [
    Position(ticker="AAPL", sector="tech", option_type="call",
             strike=180.0, stock_price=185.0, entry_price=5.50,
             expiry="2026-06-20", entry_date="2026-03-01",
             qty=2, total_cost=1100.0),
]

analyzer = PortfolioRiskAnalyzer(account_size=50000.0)
report = analyzer.analyze(positions)

print(f"Effective bets: {report['concentration']['effective_bets']}")
print(f"Daily theta: ${report['theta']['total_daily_theta_usd']:.0f}")
```

**9 Risk Dimensions:**
1. Sector concentration (weighted by invested capital)
2. Single-name concentration
3. Correlation clustering (HHI-based effective bets)
4. Delta exposure (logistic sigmoid approximation — no IV input needed)
5. Theta decay (3-tier acceleration model)
6. Vega sensitivity (sqrt-scaled by DTE)
7. Premium remaining (sqrt time decay)
8. Scenario stress testing (SPY ±10%, VIX +20, theta bleed)
9. Position health scoring (GREEN/YELLOW/RED)

Also includes **`ExitSignalDetector`** — 8-layer priority-ordered exit framework:

```python
from risk import ExitSignalDetector

detector = ExitSignalDetector()
result = detector.analyze(position_dict, indicator_dict)
# → {"overall_action": "REDUCE", "signals": [...], "checks_run": 8}
```

Exit checks (in priority order): HARD_STOP → TARGET_HIT → MOMENTUM_FADE → TREND_BREAK → OBV_DIVERGENCE → TRAILING_STOP → TIME_DECAY → IV_COLLAPSE

### `screening/` — Two-Phase Opportunity Scanner

Fast screen → Deep analysis pipeline. Screens 100+ tickers efficiently by filtering aggressively in Phase 1.

```python
from screening import TwoPhaseScanner, QuickScorer, BigWinnerScorer

# Phase 1: Score with 4-component framework (100 pts)
scorer = QuickScorer()
score = scorer.score(indicator_summary)
# → {"volume_price": 28, "momentum": 18, "trend": 16, "risk": 14, "total": 76}

# Big winner overlay (0-75 bonus pts)
bw_scorer = BigWinnerScorer(big_winner_types={"smallcap_speculative", "crypto_mining"})
bw = bw_scorer.score(summary, stock_type="smallcap_speculative", consecutive_signals=3)
# → {"hv_high": 20, "stock_type_match": 15, ..., "total": 55}

# Full pipeline: Phase 1 rank → Phase 2 deep analysis
scanner = TwoPhaseScanner(deep_analyze_fn=my_analysis_fn)
results = scanner.run(phase1_results, top_n=20)
```

**Scoring framework:**
- Base 100: Volume/Price (35) + Momentum (25) + Trend (20) + Risk (20)
- Big Winner overlay: up to 75 bonus points for high-volatility, mean-reversion, consecutive-signal setups
- Composite = quick × 0.4 + big_winner × 0.6

## Architecture

```
Data Sources (25+)
       │
       ▼
┌─────────────────────┐
│  API Reliability     │  ← Connection pool + Semaphore + Retry
│  Layer               │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐ ┌────────┐
│ Screen │ │ Monitor│   ← Two-phase scan / Exit signal detection
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌────────────────────┐
│ Signal Calibration │   ← Bayesian shrinkage + Wilson CIs
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Risk Analysis      │   ← 9 dimensions + Scenario stress tests
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Self-Evolving Loop │   ← Weekly: harvest → update → detect drift
└────────────────────┘
```

## Performance

| Component | Metric |
|-----------|--------|
| Data aggregation | 15s for 25+ sources (80x vs sequential) |
| Calibration | 12,708 signals, 50K Monte Carlo rounds |
| Risk analysis | 9 dimensions in <3s |
| Screening | 183 tickers in ~3 min (Phase 1) |
| Full pipeline | <3 min end-to-end |

## Tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

```
148 passed in 0.5s
```

## Built With

Python 3.12+, requests, threading, concurrent.futures

## License

MIT
