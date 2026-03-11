"""Tests for Exit Signal Detector.

Tests cover:
- All 8 individual exit checks
- Check priority ordering (highest severity wins)
- Configurable thresholds
- Edge cases (empty data, zero values, missing fields)
- Full detector pipeline
"""

from datetime import date, timedelta

import pytest

from risk.exit_signals import (
    ExitSignalDetector,
    ExitThresholds,
    check_hard_stop,
    check_iv_collapse,
    check_momentum_fade,
    check_obv_divergence,
    check_target_hit,
    check_time_decay,
    check_trailing_stop,
    check_trend_break,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _future_date(days: int) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y-%m-%d")


# ─── Check 1: Hard Stop ────────────────────────────────────────────────────


class TestHardStop:
    def test_below_stop(self):
        result = check_hard_stop(2.00, 2.50)
        assert result is not None
        assert result["action"] == "EXIT"
        assert result["severity"] == "CRITICAL"

    def test_at_stop(self):
        result = check_hard_stop(2.50, 2.50)
        assert result is not None
        assert result["action"] == "EXIT"

    def test_above_stop(self):
        assert check_hard_stop(3.00, 2.50) is None

    def test_no_stop_set(self):
        assert check_hard_stop(3.00, 0) is None

    def test_no_option_price(self):
        assert check_hard_stop(0, 2.50) is None


# ─── Check 2: Target Hit ──────────────────────────────────────────────────


class TestTargetHit:
    def test_target2_exit(self):
        result = check_target_hit(12.00, target1=8.00, target2=11.00)
        assert result is not None
        assert result["action"] == "EXIT"
        assert "TARGET2" in result["detail"]

    def test_target1_reduce(self):
        result = check_target_hit(8.50, target1=8.00, target2=11.00)
        assert result is not None
        assert result["action"] == "REDUCE"
        assert "TARGET1" in result["detail"]

    def test_below_targets(self):
        assert check_target_hit(5.00, target1=8.00, target2=11.00) is None

    def test_no_targets(self):
        assert check_target_hit(5.00) is None

    def test_no_price(self):
        assert check_target_hit(0, target1=8.00) is None


# ─── Check 3: Momentum Fade ───────────────────────────────────────────────


class TestMomentumFade:
    def test_rsi_drop_and_macd_negative(self):
        rsi = [65, 70, 75, 68, 55]  # peak=75, current=55, drop=20
        macd = [0.5, 0.2, -0.1, -0.3, -0.2]
        result = check_momentum_fade(rsi, macd, ExitThresholds())
        assert result is not None
        assert result["action"] == "REDUCE"

    def test_rsi_drop_only_no_macd(self):
        rsi = [65, 70, 75, 68, 55]
        macd = [0.5, 0.2, 0.1, -0.3, 0.1]  # not all negative
        assert check_momentum_fade(rsi, macd, ExitThresholds()) is None

    def test_macd_negative_only_no_rsi_drop(self):
        rsi = [60, 62, 65, 63, 62]  # peak=65, current=62, drop=3 (< 15)
        macd = [-0.1, -0.2, -0.3, -0.4, -0.5]
        assert check_momentum_fade(rsi, macd, ExitThresholds()) is None

    def test_empty_data(self):
        assert check_momentum_fade([], [], ExitThresholds()) is None
        assert check_momentum_fade(None, None, ExitThresholds()) is None

    def test_too_few_bars(self):
        assert check_momentum_fade([70, 55], [0.1], ExitThresholds()) is None


# ─── Check 4: Trend Break ─────────────────────────────────────────────────


class TestTrendBreak:
    def test_bearish_triggers(self):
        result = check_trend_break("bearish")
        assert result is not None
        assert result["action"] == "REDUCE"

    def test_mixed_triggers(self):
        result = check_trend_break("mixed")
        assert result is not None

    def test_bullish_no_trigger(self):
        assert check_trend_break("bullish") is None


# ─── Check 5: OBV Divergence ──────────────────────────────────────────────


class TestOBVDivergence:
    def test_divergence_detected(self):
        # Price at high, OBV not
        close = [100, 102, 104, 103, 105, 104, 106, 107, 108, 110]
        obv = [1000, 1050, 1080, 1060, 1070, 1040, 1030, 1020, 1010, 1000]
        result = check_obv_divergence(close, obv, ExitThresholds())
        assert result is not None
        assert result["action"] == "WATCH"

    def test_no_divergence(self):
        close = [100, 102, 104, 103, 105, 104, 106, 107, 108, 110]
        obv = [1000, 1020, 1040, 1030, 1050, 1040, 1060, 1070, 1080, 1100]
        assert check_obv_divergence(close, obv, ExitThresholds()) is None

    def test_too_few_bars(self):
        assert check_obv_divergence([100, 102], [1000, 1020], ExitThresholds()) is None


# ─── Check 6: Trailing Stop ───────────────────────────────────────────────


class TestTrailingStop:
    def test_large_pullback(self):
        highs = [100, 105, 110, 108, 107]
        result = check_trailing_stop(highs, 100.0, ExitThresholds())
        # pullback: (110 - 100) / 110 * 100 = 9.09% > 7%
        assert result is not None
        assert result["action"] == "REDUCE"

    def test_small_pullback(self):
        highs = [100, 105, 108]
        assert check_trailing_stop(highs, 105.0, ExitThresholds()) is None

    def test_empty_highs(self):
        assert check_trailing_stop([], 100.0, ExitThresholds()) is None

    def test_single_bar(self):
        assert check_trailing_stop([100], 95.0, ExitThresholds()) is None


# ─── Check 7: Time Decay ──────────────────────────────────────────────────


class TestTimeDecay:
    def test_low_dte_triggers(self):
        result = check_time_decay(
            _future_date(5), 14, today=date.today()
        )
        assert result is not None
        assert result["action"] == "EXIT"

    def test_high_dte_no_trigger(self):
        result = check_time_decay(
            _future_date(60), 14, today=date.today()
        )
        assert result is None

    def test_expired(self):
        past = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
        result = check_time_decay(past, 14, today=date.today())
        assert result is not None
        assert result["action"] == "EXIT"

    def test_empty_expiry(self):
        assert check_time_decay("", 14) is None

    def test_invalid_expiry(self):
        assert check_time_decay("not-a-date", 14) is None


# ─── Check 8: IV Collapse ─────────────────────────────────────────────────


class TestIVCollapse:
    def test_contraction_detected(self):
        result = check_iv_collapse(20.0, 40.0, ExitThresholds())
        # 20 < 40 * 0.80 = 32 → contracting
        assert result is not None
        assert result["action"] == "WATCH"

    def test_no_contraction(self):
        assert check_iv_collapse(35.0, 40.0, ExitThresholds()) is None

    def test_none_values(self):
        assert check_iv_collapse(None, 40.0, ExitThresholds()) is None
        assert check_iv_collapse(35.0, None, ExitThresholds()) is None

    def test_zero_hv60(self):
        assert check_iv_collapse(20.0, 0.0, ExitThresholds()) is None


# ─── Detector Pipeline Tests ──────────────────────────────────────────────


class TestExitSignalDetector:
    """Test the full ExitSignalDetector pipeline."""

    @pytest.fixture
    def detector(self):
        return ExitSignalDetector()

    def test_healthy_position_hold(self, detector):
        position = {
            "option_price": 5.50,
            "stop_loss": 2.50,
            "target1": 8.00,
            "target2": 12.00,
            "expiry": _future_date(60),
        }
        indicators = {
            "rsi": [55, 58, 60, 62, 65],
            "macd_histogram": [0.1, 0.2, 0.3, 0.2, 0.1],
            "ema_arrangement": "bullish",
            "close_prices": [180, 181, 182, 183, 184, 185, 186, 187, 188, 189],
            "high_prices": [181, 182, 183, 184, 185, 186, 187, 188, 189, 190],
            "obv": [1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180],
            "hv_20": 30.0,
            "hv_60": 30.0,
        }
        result = detector.analyze(position, indicators)
        assert result["overall_action"] == "HOLD"
        assert result["checks_run"] == 8

    def test_stop_hit_overrides_all(self, detector):
        position = {
            "option_price": 2.00,
            "stop_loss": 2.50,
            "target1": 8.00,
            "target2": 12.00,
            "expiry": _future_date(60),
        }
        indicators = {
            "rsi": [55, 58, 60, 62, 65],
            "macd_histogram": [0.1, 0.2, 0.3, 0.2, 0.1],
            "ema_arrangement": "bullish",
            "close_prices": list(range(180, 190)),
            "high_prices": list(range(181, 191)),
            "obv": list(range(1000, 1100, 10)),
            "hv_20": 30.0,
            "hv_60": 30.0,
        }
        result = detector.analyze(position, indicators)
        assert result["overall_action"] == "EXIT"
        assert any(s["check"] == "HARD_STOP" for s in result["signals"])

    def test_multiple_signals_highest_wins(self, detector):
        """Trend break (REDUCE) + OBV divergence (WATCH) → overall REDUCE."""
        position = {
            "option_price": 5.50,
            "stop_loss": 2.50,
            "expiry": _future_date(60),
        }
        indicators = {
            "rsi": [60, 62, 63, 61, 60],
            "macd_histogram": [0.1, 0.2, 0.1, 0.05, 0.01],
            "ema_arrangement": "bearish",
            "close_prices": [100, 102, 104, 103, 105, 104, 106, 107, 108, 110],
            "high_prices": [101, 103, 105, 104, 106, 105, 107, 108, 109, 111],
            "obv": [1000, 1050, 1080, 1060, 1070, 1040, 1030, 1020, 1010, 1000],
            "hv_20": 20.0,
            "hv_60": 35.0,
        }
        result = detector.analyze(position, indicators)
        assert result["overall_action"] == "REDUCE"

    def test_custom_thresholds(self):
        thresholds = ExitThresholds(
            rsi_drop_threshold=5.0,  # more sensitive
            trailing_stop_pct=3.0,  # tighter stop
        )
        detector = ExitSignalDetector(thresholds=thresholds)

        position = {
            "option_price": 5.50,
            "stop_loss": 2.50,
            "expiry": _future_date(60),
        }
        indicators = {
            "rsi": [60, 65, 70, 68, 63],  # drop=7 > custom 5
            "macd_histogram": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "ema_arrangement": "bullish",
            "close_prices": list(range(180, 190)),
            "high_prices": list(range(181, 191)),
            "obv": list(range(1000, 1100, 10)),
            "hv_20": 30.0,
            "hv_60": 30.0,
        }
        result = detector.analyze(position, indicators)
        assert any(s["check"] == "MOMENTUM_FADE" for s in result["signals"])

    def test_empty_indicators(self, detector):
        position = {"option_price": 5.50, "expiry": _future_date(60)}
        indicators = {}
        result = detector.analyze(position, indicators)
        assert result["overall_action"] in ("HOLD", "WATCH", "REDUCE", "EXIT")
        assert result["checks_run"] == 8

    def test_minimal_position(self, detector):
        result = detector.analyze({}, {})
        assert result["checks_run"] == 8
