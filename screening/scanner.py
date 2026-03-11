"""
Two-Phase Opportunity Scanner — Fast screen → Deep analysis pipeline.

Designed for scanning large ticker universes efficiently:
  Phase 1 (Quick Screen): Lightweight scoring on pre-fetched daily data.
           Uses 4-component mechanical score (100 pts) + big-winner overlay (75 pts).
           Runs in parallel, screens 100+ tickers in minutes.
  Phase 2 (Deep Analysis): Full multi-timeframe analysis on top candidates only.
           Configurable: pass your own deep analysis function.

The two-phase approach saves ~80% of API calls by filtering aggressively
in Phase 1. Only the top N candidates proceed to expensive Phase 2.

Scoring framework (Phase 1):
  Base 100 points:
    - Volume & Price action: 35 pts (volume ratio, OBV, BB position, EMA arrangement)
    - Momentum: 25 pts (RSI, MACD cross, MACD histogram)
    - Trend: 20 pts (EMA arrangement, pullback depth)
    - Risk: 20 pts (RSI-SMA, HV20, ADX, BB width)

  Big Winner overlay (0-75 bonus points):
    - High volatility (HV>80): +20
    - Stock type match: +15
    - Mean reversion setup: +10
    - Consecutive same signals: +10
    - Low-score negative adjustment: +10
    - Low IV environment: +5
    - Pullback in sweet zone: +5

  Composite = quick_score × 0.4 + big_winner_score × 0.6

Example:
    from screening import TwoPhaseScanner, QuickScorer

    # Phase 1: Score pre-fetched data
    scorer = QuickScorer()
    for ticker, summary in daily_summaries.items():
        score = scorer.score(summary)
        print(f"{ticker}: {score['total']}/100")

    # Full pipeline with custom deep analyzer
    def my_deep_analyzer(ticker: str) -> dict:
        # Your full analysis logic here
        return {"ticker": ticker, "score": 85, "signal": "BUY"}

    scanner = TwoPhaseScanner(deep_analyze_fn=my_deep_analyzer)
    results = scanner.run(
        quick_results=[...],  # Phase 1 results
        top_n=10,
    )
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ─── Scoring Components ───────────────────────────────────────────────────


class QuickScorer:
    """Phase 1 mechanical scorer: 4-component 100-point framework.

    Input is a dict of indicator values (from any technical analysis library).
    Output is a structured score breakdown.

    Expected indicator keys:
        volume_ratio, obv_status, bb_position_pct, ema_arrangement,
        rsi, macd_cross, macd_histogram, pullback_pct, rsi_sma,
        hv_20, adx, bb_upper, bb_lower, price
    """

    def score(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Compute 100-point mechanical score from indicator summary."""
        scores = {}

        # ─── Volume & Price (35 pts) ──────────────────────────────
        vp = 0
        vol_ratio = summary.get("volume_ratio")
        if vol_ratio is not None:
            if vol_ratio > 2.0:
                vp += 10
            elif vol_ratio > 1.2:
                vp += 7
            elif vol_ratio > 0.8:
                vp += 4
            else:
                vp += 1

        obv_status = summary.get("obv_status")
        if obv_status == "above_ma":
            vp += 8
        else:
            vp += 2

        bb_pct = summary.get("bb_position_pct")
        if bb_pct is not None:
            if 40 <= bb_pct <= 70:
                vp += 8  # healthy middle
            elif bb_pct > 80:
                vp += 3  # overbought
            elif bb_pct < 20:
                vp += 5  # oversold, mean-reversion
            else:
                vp += 6
        scores["volume_price"] = min(vp + self._ema_vp_bonus(summary), 35)

        # ─── Momentum (25 pts) — RSI + MACD ──────────────────────
        mom = 0
        rsi = summary.get("rsi")
        if rsi is not None:
            if 45 <= rsi <= 65:
                mom += 10
            elif 30 <= rsi < 45:
                mom += 7
            elif rsi < 30:
                mom += 5
            elif rsi <= 75:
                mom += 6
            else:
                mom += 2

        macd_cross = summary.get("macd_cross")
        mom += 8 if macd_cross == "bullish" else 3

        hist = summary.get("macd_histogram")
        if hist is not None:
            if hist > 0:
                mom += 7
            elif hist > -0.5:
                mom += 4
            else:
                mom += 2
        scores["momentum"] = min(mom, 25)

        # ─── Trend (20 pts) ──────────────────────────────────────
        trend = 0
        arrangement = summary.get("ema_arrangement")
        if arrangement == "bullish":
            trend += 12
        elif arrangement == "mixed":
            trend += 6
        else:
            trend += 2

        pullback = summary.get("pullback_pct")
        if pullback is not None:
            if 3 <= pullback <= 10:
                trend += 8
            elif pullback < 3:
                trend += 5
            elif pullback <= 20:
                trend += 4
            else:
                trend += 1
        scores["trend"] = min(trend, 20)

        # ─── Risk (20 pts) ───────────────────────────────────────
        risk = 0
        rsi_sma = summary.get("rsi_sma")
        if rsi_sma is not None:
            if rsi_sma > 50:
                risk += 5
            elif rsi_sma > 40:
                risk += 3
            else:
                risk += 1

        hv = summary.get("hv_20")
        if hv is not None:
            if hv < 30:
                risk += 5
            elif hv < 50:
                risk += 4
            elif hv < 80:
                risk += 3
            else:
                risk += 2

        adx = summary.get("adx")
        if adx is not None:
            if 20 <= adx <= 40:
                risk += 5
            elif adx < 20:
                risk += 3
            else:
                risk += 2

        bb_upper = summary.get("bb_upper")
        bb_lower = summary.get("bb_lower")
        price = summary.get("price", 0)
        if bb_upper and bb_lower and price:
            bb_width_pct = (bb_upper - bb_lower) / price * 100
            if bb_width_pct < 10:
                risk += 5
            elif bb_width_pct < 20:
                risk += 3
            else:
                risk += 1
        scores["risk"] = min(risk, 20)

        scores["total"] = (
            scores["volume_price"]
            + scores["momentum"]
            + scores["trend"]
            + scores["risk"]
        )
        return scores

    @staticmethod
    def _ema_vp_bonus(summary: dict) -> int:
        arrangement = summary.get("ema_arrangement")
        if arrangement == "bullish":
            return 9
        elif arrangement == "mixed":
            return 5
        return 1


class BigWinnerScorer:
    """Big-winner overlay scorer (0-75 bonus points).

    Identifies setups with outsized return potential based on
    backtest evidence of what characteristics big winners share.

    Args:
        big_winner_types: Set of stock type strings that historically
                         produce outsized returns. User-configurable.
    """

    def __init__(self, big_winner_types: Optional[set[str]] = None):
        self.big_winner_types = big_winner_types or set()

    def score(
        self,
        summary: dict[str, Any],
        stock_type: str = "",
        consecutive_signals: int = 0,
        quick_score: int = 50,
    ) -> dict[str, Any]:
        """Compute big-winner overlay score.

        Args:
            summary: Indicator summary dict (needs hv_20, pullback_pct)
            stock_type: Detected stock classification
            consecutive_signals: Number of consecutive same signals
            quick_score: Phase 1 quick score (for mean reversion detection)

        Returns:
            Dict with component scores and total (0-75).
        """
        bw: dict[str, Any] = {}
        total = 0

        # HV20 > 80 → +20 (high volatility = big move potential)
        hv = summary.get("hv_20")
        pts = 20 if (hv is not None and hv > 80) else 0
        bw["hv_high"] = pts
        total += pts

        # Stock type match → +15
        pts = 15 if stock_type in self.big_winner_types else 0
        bw["stock_type_match"] = pts
        total += pts

        # Mean reversion proxy → +10
        pts = 10 if quick_score < 50 else 0
        bw["mean_reversion"] = pts
        total += pts

        # Consecutive same signal ≥ 3 → +10
        pts = 10 if consecutive_signals >= 3 else 0
        bw["consecutive_signal"] = pts
        total += pts

        # Deep negative score → +10
        pts = 10 if quick_score < 45 else 0
        bw["negative_adj"] = pts
        total += pts

        # Low IV (cheap options) → +5
        pts = 5 if (hv is not None and hv < 25) else 0
        bw["low_iv"] = pts
        total += pts

        # Pullback in sweet zone → +5
        pullback = summary.get("pullback_pct")
        pts = 5 if (pullback is not None and 5 <= pullback <= 15) else 0
        bw["pullback_zone"] = pts
        total += pts

        bw["total"] = total
        return bw


# ─── Signal Journal Helpers ────────────────────────────────────────────────


def count_consecutive_signals(entries: list[dict[str, Any]]) -> int:
    """Count consecutive same-signal entries (most recent streak).

    Args:
        entries: List of journal entries with "date" and "signal" keys.

    Returns:
        Length of the most recent consecutive same-signal streak.
    """
    if not entries:
        return 0
    sorted_entries = sorted(
        entries, key=lambda e: e.get("date", ""), reverse=True
    )
    last_signal = sorted_entries[0].get("signal", "")
    count = 0
    for entry in sorted_entries:
        if entry.get("signal") == last_signal:
            count += 1
        else:
            break
    return count


# ─── Two-Phase Scanner ────────────────────────────────────────────────────


class TwoPhaseScanner:
    """Orchestrates the two-phase scan: quick screen → deep analysis.

    Phase 1 results are passed in (user handles data fetching).
    Phase 2 uses a user-provided deep analysis function.

    Args:
        deep_analyze_fn: Callable that takes a ticker string and returns
                        a result dict (or None on failure).
        max_workers: Thread pool size for Phase 2 parallel analysis.
        composite_weights: (quick_weight, bw_weight) for composite score.
                          Default: (0.4, 0.6) — big winner weighted more.
    """

    def __init__(
        self,
        deep_analyze_fn: Optional[Callable[[str], Optional[dict]]] = None,
        max_workers: int = 5,
        composite_weights: tuple[float, float] = (0.4, 0.6),
    ):
        self.deep_analyze_fn = deep_analyze_fn
        self.max_workers = max_workers
        self.quick_weight, self.bw_weight = composite_weights

    def compute_composite(self, quick_score: float, bw_score: float) -> float:
        """Compute weighted composite score."""
        return round(
            quick_score * self.quick_weight + bw_score * self.bw_weight, 1
        )

    def rank_phase1(
        self, results: list[dict[str, Any]], top_n: int = 20
    ) -> list[dict[str, Any]]:
        """Rank Phase 1 results by composite score, return top N."""
        for r in results:
            if "composite" not in r:
                r["composite"] = self.compute_composite(
                    r.get("quick_score", 0), r.get("big_winner_score", 0)
                )
        ranked = sorted(results, key=lambda r: r["composite"], reverse=True)
        return ranked[:top_n]

    def run_phase2(
        self, tickers: list[str]
    ) -> list[dict[str, Any]]:
        """Run Phase 2 deep analysis on selected tickers in parallel.

        Returns list of successful analysis results.
        Requires deep_analyze_fn to be set.
        """
        if not self.deep_analyze_fn:
            raise ValueError("deep_analyze_fn not set — cannot run Phase 2")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.deep_analyze_fn, t): t for t in tickers
            }
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if result:
                        results.append(result)
                except Exception:
                    pass  # Deep analysis failure → skip ticker
        return results

    def run(
        self,
        quick_results: list[dict[str, Any]],
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Run the full two-phase pipeline.

        Args:
            quick_results: Phase 1 scoring results (list of dicts with
                          'ticker', 'quick_score', 'big_winner_score')
            top_n: Number of candidates to send to Phase 2

        Returns:
            Dict with phase1_ranked, phase2_results (if deep_analyze_fn set),
            and summary statistics.
        """
        # Phase 1: Rank
        ranked = self.rank_phase1(quick_results, top_n)

        result = {
            "phase1_total_screened": len(quick_results),
            "phase1_top_n": len(ranked),
            "phase1_ranked": ranked,
            "phase2_results": [],
        }

        # Phase 2: Deep analysis (if function provided)
        if self.deep_analyze_fn and ranked:
            tickers = [r["ticker"] for r in ranked]
            deep_results = self.run_phase2(tickers)
            result["phase2_results"] = deep_results
            result["phase2_analyzed"] = len(deep_results)

        return result
