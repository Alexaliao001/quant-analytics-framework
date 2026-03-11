"""Tests for Two-Phase Opportunity Scanner.

Tests cover:
- QuickScorer 4-component scoring
- BigWinnerScorer overlay scoring
- Consecutive signal counting
- TwoPhaseScanner ranking and pipeline
- Edge cases (empty data, missing fields)
"""

import pytest

from screening.scanner import (
    BigWinnerScorer,
    QuickScorer,
    TwoPhaseScanner,
    count_consecutive_signals,
)


# ─── QuickScorer Tests ────────────────────────────────────────────────────


class TestQuickScorer:
    """Test 4-component mechanical scoring."""

    @pytest.fixture
    def scorer(self):
        return QuickScorer()

    def test_bullish_setup_high_score(self, scorer):
        summary = {
            "volume_ratio": 2.5,
            "obv_status": "above_ma",
            "bb_position_pct": 55,
            "ema_arrangement": "bullish",
            "rsi": 55,
            "macd_cross": "bullish",
            "macd_histogram": 0.5,
            "pullback_pct": 5,
            "rsi_sma": 55,
            "hv_20": 25,
            "adx": 30,
            "bb_upper": 110,
            "bb_lower": 100,
            "price": 105,
        }
        score = scorer.score(summary)
        assert score["total"] > 70

    def test_bearish_setup_low_score(self, scorer):
        summary = {
            "volume_ratio": 0.3,
            "obv_status": "below_ma",
            "bb_position_pct": 90,
            "ema_arrangement": "bearish",
            "rsi": 85,
            "macd_cross": "bearish",
            "macd_histogram": -2.0,
            "pullback_pct": 25,
            "rsi_sma": 35,
            "hv_20": 90,
            "adx": 50,
            "bb_upper": 130,
            "bb_lower": 90,
            "price": 100,
        }
        score = scorer.score(summary)
        assert score["total"] < 50

    def test_all_components_present(self, scorer):
        score = scorer.score({"ema_arrangement": "bullish"})
        assert "volume_price" in score
        assert "momentum" in score
        assert "trend" in score
        assert "risk" in score
        assert "total" in score

    def test_total_is_sum_of_components(self, scorer):
        summary = {
            "volume_ratio": 1.5,
            "ema_arrangement": "mixed",
            "rsi": 50,
            "macd_cross": "bullish",
            "macd_histogram": 0.3,
        }
        score = scorer.score(summary)
        assert score["total"] == (
            score["volume_price"]
            + score["momentum"]
            + score["trend"]
            + score["risk"]
        )

    def test_empty_summary(self, scorer):
        score = scorer.score({})
        assert score["total"] >= 0
        assert score["total"] <= 100

    def test_components_bounded(self, scorer):
        summary = {
            "volume_ratio": 100,
            "obv_status": "above_ma",
            "bb_position_pct": 55,
            "ema_arrangement": "bullish",
            "rsi": 55,
            "macd_cross": "bullish",
            "macd_histogram": 10,
            "pullback_pct": 5,
            "rsi_sma": 80,
            "hv_20": 10,
            "adx": 30,
            "bb_upper": 110,
            "bb_lower": 105,
            "price": 107,
        }
        score = scorer.score(summary)
        assert score["volume_price"] <= 35
        assert score["momentum"] <= 25
        assert score["trend"] <= 20
        assert score["risk"] <= 20


# ─── BigWinnerScorer Tests ────────────────────────────────────────────────


class TestBigWinnerScorer:
    """Test big-winner overlay scoring."""

    @pytest.fixture
    def scorer(self):
        return BigWinnerScorer(
            big_winner_types={"smallcap_speculative", "crypto_mining"}
        )

    def test_all_criteria_met(self, scorer):
        summary = {"hv_20": 90, "pullback_pct": 8}
        result = scorer.score(
            summary,
            stock_type="smallcap_speculative",
            consecutive_signals=4,
            quick_score=40,
        )
        assert result["hv_high"] == 20
        assert result["stock_type_match"] == 15
        assert result["mean_reversion"] == 10
        assert result["consecutive_signal"] == 10
        assert result["negative_adj"] == 10
        assert result["pullback_zone"] == 5
        assert result["total"] == 70

    def test_no_criteria_met(self, scorer):
        summary = {"hv_20": 30}
        result = scorer.score(
            summary,
            stock_type="tech",
            consecutive_signals=1,
            quick_score=70,
        )
        assert result["total"] == 0

    def test_low_iv_bonus(self, scorer):
        summary = {"hv_20": 20}
        result = scorer.score(summary, quick_score=60)
        assert result["low_iv"] == 5

    def test_empty_big_winner_types(self):
        scorer = BigWinnerScorer()
        result = scorer.score({"hv_20": 90}, stock_type="anything")
        assert result["stock_type_match"] == 0

    def test_max_score_75(self, scorer):
        summary = {"hv_20": 90, "pullback_pct": 10}
        result = scorer.score(
            summary,
            stock_type="smallcap_speculative",
            consecutive_signals=5,
            quick_score=30,
        )
        assert result["total"] <= 75


# ─── Consecutive Signals Tests ────────────────────────────────────────────


class TestConsecutiveSignals:
    def test_three_same_signals(self):
        entries = [
            {"date": "2026-03-01", "signal": "BUY"},
            {"date": "2026-03-02", "signal": "BUY"},
            {"date": "2026-03-03", "signal": "BUY"},
        ]
        assert count_consecutive_signals(entries) == 3

    def test_signal_change_breaks_streak(self):
        entries = [
            {"date": "2026-03-01", "signal": "SELL"},
            {"date": "2026-03-02", "signal": "BUY"},
            {"date": "2026-03-03", "signal": "BUY"},
        ]
        assert count_consecutive_signals(entries) == 2

    def test_empty_entries(self):
        assert count_consecutive_signals([]) == 0

    def test_single_entry(self):
        assert count_consecutive_signals([{"date": "2026-03-01", "signal": "BUY"}]) == 1

    def test_unsorted_entries(self):
        entries = [
            {"date": "2026-03-03", "signal": "BUY"},
            {"date": "2026-03-01", "signal": "SELL"},
            {"date": "2026-03-02", "signal": "BUY"},
        ]
        # Most recent is 03-03 (BUY), then 03-02 (BUY), then 03-01 (SELL)
        assert count_consecutive_signals(entries) == 2


# ─── TwoPhaseScanner Tests ────────────────────────────────────────────────


class TestTwoPhaseScanner:
    """Test the two-phase scanning pipeline."""

    def test_rank_phase1_sorts_by_composite(self):
        scanner = TwoPhaseScanner()
        results = [
            {"ticker": "AAA", "quick_score": 60, "big_winner_score": 30},
            {"ticker": "BBB", "quick_score": 80, "big_winner_score": 50},
            {"ticker": "CCC", "quick_score": 40, "big_winner_score": 20},
        ]
        ranked = scanner.rank_phase1(results, top_n=2)
        assert len(ranked) == 2
        assert ranked[0]["ticker"] == "BBB"

    def test_composite_calculation(self):
        scanner = TwoPhaseScanner(composite_weights=(0.4, 0.6))
        assert scanner.compute_composite(80, 50) == 62.0

    def test_custom_weights(self):
        scanner = TwoPhaseScanner(composite_weights=(0.7, 0.3))
        assert scanner.compute_composite(80, 50) == 71.0

    def test_run_phase2_no_function(self):
        scanner = TwoPhaseScanner()
        with pytest.raises(ValueError):
            scanner.run_phase2(["AAPL", "NVDA"])

    def test_run_phase2_with_function(self):
        def mock_deep(ticker: str) -> dict:
            return {"ticker": ticker, "deep_score": 85}

        scanner = TwoPhaseScanner(deep_analyze_fn=mock_deep)
        results = scanner.run_phase2(["AAPL", "NVDA"])
        assert len(results) == 2
        tickers = {r["ticker"] for r in results}
        assert tickers == {"AAPL", "NVDA"}

    def test_run_phase2_handles_failures(self):
        call_count = {"n": 0}
        def failing_deep(ticker: str) -> dict:
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise RuntimeError("API error")
            return {"ticker": ticker}

        scanner = TwoPhaseScanner(deep_analyze_fn=failing_deep)
        results = scanner.run_phase2(["A", "B", "C", "D"])
        # Some succeed, some fail
        assert len(results) >= 1
        assert len(results) <= 4

    def test_full_pipeline(self):
        def mock_deep(ticker: str) -> dict:
            return {"ticker": ticker, "final_score": 85}

        scanner = TwoPhaseScanner(deep_analyze_fn=mock_deep)
        quick_results = [
            {"ticker": "AAA", "quick_score": 70, "big_winner_score": 40},
            {"ticker": "BBB", "quick_score": 50, "big_winner_score": 60},
            {"ticker": "CCC", "quick_score": 90, "big_winner_score": 10},
        ]
        result = scanner.run(quick_results, top_n=2)

        assert result["phase1_total_screened"] == 3
        assert result["phase1_top_n"] == 2
        assert len(result["phase2_results"]) == 2

    def test_full_pipeline_no_deep_fn(self):
        scanner = TwoPhaseScanner()
        quick_results = [
            {"ticker": "AAA", "quick_score": 70, "big_winner_score": 40},
        ]
        result = scanner.run(quick_results, top_n=5)
        assert result["phase2_results"] == []
        assert len(result["phase1_ranked"]) == 1

    def test_empty_input(self):
        scanner = TwoPhaseScanner()
        result = scanner.run([], top_n=5)
        assert result["phase1_total_screened"] == 0
        assert result["phase1_ranked"] == []
