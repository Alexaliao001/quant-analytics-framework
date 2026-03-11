"""
Exit Signal Detector — 8-layer priority-ordered exit signal framework for options positions.

Checks are ordered by severity and urgency:
  1. HARD_STOP: Option price at or below stop loss
  2. TARGET_HIT: Option price at or above target levels
  3. MOMENTUM_FADE: RSI drop from peak + MACD histogram negative streak
  4. TREND_BREAK: EMA arrangement shifted from bullish
  5. OBV_DIVERGENCE: Price at highs but volume not confirming
  6. TRAILING_STOP: Pullback exceeds threshold from post-entry high
  7. TIME_DECAY: DTE below threshold (theta cliff approaching)
  8. IV_COLLAPSE: Historical volatility contracting (proxy for IV crush)

Each check returns a structured signal with:
  - check: which check triggered
  - action: EXIT / REDUCE / WATCH / HOLD
  - severity: CRITICAL / WARNING / TARGET / INFO
  - detail: human-readable explanation

The overall action for a position is the highest-severity triggered signal.

Design philosophy:
  - No external data fetching — accepts pre-computed indicator data
  - Configurable thresholds for different strategies
  - Priority ordering ensures critical checks are never masked
  - Each check is independently testable

Example:
    from risk import ExitSignalDetector

    detector = ExitSignalDetector()

    indicators = {
        "rsi": [65.2, 68.1, 70.3, 62.4, 55.1],
        "macd_histogram": [0.3, 0.1, -0.2, -0.4, -0.3],
        "ema_arrangement": "bearish",
        "close_prices": [180, 182, 185, 183, 179],
        "high_prices": [181, 184, 186, 184, 180],
        "obv": [1000, 1050, 1020, 980, 960],
        "hv_20": 25.0,
        "hv_60": 35.0,
    }

    position = {
        "stop_loss": 2.50,
        "target1": 8.00,
        "target2": 12.00,
        "option_price": 5.50,
        "entry_date": "2026-02-01",
        "expiry": "2026-06-20",
        "entry_stock_price": 175.0,
    }

    result = detector.analyze(position, indicators)
    print(f"Action: {result['overall_action']}")
    for sig in result['signals']:
        print(f"  [{sig['severity']}] {sig['check']}: {sig['detail']}")
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional


@dataclass
class ExitThresholds:
    """Configurable thresholds for exit signal detection.

    All thresholds have sensible defaults. Adjust based on your
    strategy's risk tolerance and typical holding period.
    """

    rsi_drop_threshold: float = 15.0  # RSI drop from peak
    macd_negative_bars: int = 3  # consecutive negative MACD histogram bars
    trailing_stop_pct: float = 7.0  # % pullback from post-entry high
    time_decay_dte: int = 14  # exit when DTE below this
    iv_contraction_ratio: float = 0.80  # HV20 < HV60 * ratio → contracting
    obv_near_high_pct: float = 0.98  # within 2% of high = "near high"


# ─── Action Priority (higher = more urgent) ────────────────────────────────

ACTION_PRIORITY = {"EXIT": 3, "REDUCE": 2, "WATCH": 1, "HOLD": 0}


# ─── Individual Exit Checks ────────────────────────────────────────────────


def check_hard_stop(
    option_price: float, stop_loss: float
) -> Optional[dict[str, str]]:
    """#1 (CRITICAL): Option price at or below stop loss.

    This is the most important check — capital preservation.
    Always checked first, regardless of other signals.
    """
    if not stop_loss or not option_price:
        return None
    if option_price <= stop_loss:
        return {
            "check": "HARD_STOP",
            "action": "EXIT",
            "severity": "CRITICAL",
            "detail": f"Option ${option_price:.2f} <= stop ${stop_loss:.2f}",
        }
    return None


def check_target_hit(
    option_price: float,
    target1: float = 0,
    target2: float = 0,
) -> Optional[dict[str, str]]:
    """#2 (TARGET): Option price at or above target levels.

    Target2 → full exit. Target1 → partial reduction.
    """
    if not option_price:
        return None

    if target2 and option_price >= target2:
        return {
            "check": "TARGET_HIT",
            "action": "EXIT",
            "severity": "TARGET",
            "detail": f"TARGET2 hit: ${option_price:.2f} >= ${target2:.2f}",
        }
    if target1 and option_price >= target1:
        return {
            "check": "TARGET_HIT",
            "action": "REDUCE",
            "severity": "TARGET",
            "detail": f"TARGET1 hit: ${option_price:.2f} >= ${target1:.2f}",
        }
    return None


def check_momentum_fade(
    rsi_values: list[float],
    macd_histogram: list[float],
    thresholds: ExitThresholds,
) -> Optional[dict[str, str]]:
    """#3 (WARNING): RSI drop from peak + MACD histogram negative streak.

    Detects fading momentum when both RSI and MACD confirm weakness.
    Requires BOTH conditions to avoid false signals from single indicators.

    Args:
        rsi_values: Recent RSI readings (oldest to newest, since entry)
        macd_histogram: Recent MACD histogram values
        thresholds: Configurable thresholds
    """
    if not rsi_values or len(rsi_values) < 3:
        return None
    if not macd_histogram or len(macd_histogram) < thresholds.macd_negative_bars:
        return None

    rsi_peak = max(rsi_values)
    rsi_current = rsi_values[-1]
    rsi_drop = rsi_peak - rsi_current

    # Check MACD histogram negative streak
    recent_hist = macd_histogram[-thresholds.macd_negative_bars :]
    hist_negative = all(h < 0 for h in recent_hist)

    if rsi_drop > thresholds.rsi_drop_threshold and hist_negative:
        return {
            "check": "MOMENTUM_FADE",
            "action": "REDUCE",
            "severity": "WARNING",
            "detail": (
                f"RSI dropped {rsi_drop:.0f} from peak "
                f"{rsi_peak:.0f}→{rsi_current:.0f}, "
                f"MACD hist negative {thresholds.macd_negative_bars}+ bars"
            ),
        }
    return None


def check_trend_break(
    ema_arrangement: str,
) -> Optional[dict[str, str]]:
    """#4 (WARNING): EMA arrangement no longer bullish.

    When weekly EMAs flip from bullish to bearish/mixed,
    the underlying trend that supported the entry is broken.

    Args:
        ema_arrangement: "bullish", "bearish", or "mixed"
    """
    if ema_arrangement in ("bearish", "mixed"):
        return {
            "check": "TREND_BREAK",
            "action": "REDUCE",
            "severity": "WARNING",
            "detail": f"EMA arrangement: {ema_arrangement} (not bullish)",
        }
    return None


def check_obv_divergence(
    close_prices: list[float],
    obv_values: list[float],
    thresholds: ExitThresholds,
) -> Optional[dict[str, str]]:
    """#5 (INFO): Price near highs but OBV not confirming (bearish divergence).

    Volume precedes price — when volume stops confirming new highs,
    the move may be exhausting. This is a warning, not an exit trigger.

    Args:
        close_prices: Recent close prices (last 10+ bars)
        obv_values: Corresponding OBV values (same length)
        thresholds: Configurable thresholds
    """
    if not close_prices or not obv_values or len(close_prices) < 10:
        return None
    if len(obv_values) < len(close_prices):
        return None

    # Check last 10 bars
    recent_close = close_prices[-10:]
    recent_obv = obv_values[-10:]

    threshold = thresholds.obv_near_high_pct

    price_at_high = close_prices[-1] >= max(recent_close) * threshold
    obv_at_high = obv_values[-1] >= max(recent_obv) * threshold

    if price_at_high and not obv_at_high:
        return {
            "check": "OBV_DIVERGENCE",
            "action": "WATCH",
            "severity": "INFO",
            "detail": "Price near high but OBV not confirming (bearish divergence)",
        }
    return None


def check_trailing_stop(
    high_prices: list[float],
    current_price: float,
    thresholds: ExitThresholds,
) -> Optional[dict[str, str]]:
    """#6 (WARNING): Pullback exceeds threshold from post-entry high.

    Protects profits by limiting drawdown from the high-water mark.

    Args:
        high_prices: Daily highs since entry
        current_price: Current stock price
        thresholds: Configurable thresholds
    """
    if not high_prices or len(high_prices) < 2:
        return None

    post_entry_high = max(high_prices)
    if post_entry_high <= 0:
        return None

    pullback_pct = (post_entry_high - current_price) / post_entry_high * 100

    if pullback_pct > thresholds.trailing_stop_pct:
        return {
            "check": "TRAILING_STOP",
            "action": "REDUCE",
            "severity": "WARNING",
            "detail": (
                f"Stock pulled back {pullback_pct:.1f}% from "
                f"post-entry high ${post_entry_high:.2f}"
            ),
        }
    return None


def check_time_decay(
    expiry: str, dte_threshold: int, today: Optional[date] = None
) -> Optional[dict[str, str]]:
    """#7 (WARNING): DTE below threshold — theta cliff approaching.

    Options lose premium at an accelerating rate as expiry approaches.
    Typically exit or reduce when DTE < 14 trading days unless near target.

    Args:
        expiry: Expiration date string "YYYY-MM-DD"
        dte_threshold: Exit when calendar DTE below this
        today: Override today's date (for testing)
    """
    if not expiry:
        return None

    today = today or date.today()
    try:
        expiry_date = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

    dte = (expiry_date - today).days
    if dte < 0:
        dte = 0

    if dte < dte_threshold:
        return {
            "check": "TIME_DECAY",
            "action": "EXIT",
            "severity": "WARNING",
            "detail": f"DTE={dte} days (< {dte_threshold} threshold), expiry {expiry}",
        }
    return None


def check_iv_collapse(
    hv_20: float, hv_60: float, thresholds: ExitThresholds
) -> Optional[dict[str, str]]:
    """#8 (INFO): Volatility contracting (proxy for IV collapse).

    When HV20 drops well below HV60, volatility is mean-reverting
    downward. For long options positions, this erodes premium even
    if the stock doesn't move.

    Args:
        hv_20: 20-day historical volatility (annualized %)
        hv_60: 60-day historical volatility (annualized %)
        thresholds: Configurable thresholds
    """
    if hv_20 is None or hv_60 is None or hv_60 <= 0:
        return None

    if hv_20 < hv_60 * thresholds.iv_contraction_ratio:
        return {
            "check": "IV_COLLAPSE",
            "action": "WATCH",
            "severity": "INFO",
            "detail": (
                f"HV20={hv_20:.1f}% < HV60={hv_60:.1f}% × "
                f"{thresholds.iv_contraction_ratio} — volatility contracting"
            ),
        }
    return None


# ─── Detector ──────────────────────────────────────────────────────────────


class ExitSignalDetector:
    """8-layer priority-ordered exit signal detector.

    Runs all 8 checks in priority order and determines overall action.
    The highest-severity triggered signal determines the position action.

    Args:
        thresholds: Configurable exit thresholds. Defaults to conservative values.
    """

    def __init__(self, thresholds: Optional[ExitThresholds] = None):
        self.thresholds = thresholds or ExitThresholds()

    def analyze(
        self,
        position: dict[str, Any],
        indicators: dict[str, Any],
        today: Optional[date] = None,
    ) -> dict[str, Any]:
        """Run all 8 exit checks on a position.

        Args:
            position: Dict with keys: option_price, stop_loss, target1, target2,
                      entry_date, expiry, entry_stock_price.
            indicators: Dict with keys: rsi (list), macd_histogram (list),
                       ema_arrangement (str), close_prices (list), high_prices (list),
                       obv (list), hv_20 (float), hv_60 (float).
            today: Override today's date (for testing).

        Returns:
            Dict with: overall_action, signals (list), checks_run (int).
        """
        signals = []

        # Check 1: Hard stop
        result = check_hard_stop(
            position.get("option_price", 0),
            position.get("stop_loss", 0),
        )
        if result:
            signals.append(result)

        # Check 2: Target hit
        result = check_target_hit(
            position.get("option_price", 0),
            position.get("target1", 0),
            position.get("target2", 0),
        )
        if result:
            signals.append(result)

        # Check 3: Momentum fade
        result = check_momentum_fade(
            indicators.get("rsi", []),
            indicators.get("macd_histogram", []),
            self.thresholds,
        )
        if result:
            signals.append(result)

        # Check 4: Trend break
        ema = indicators.get("ema_arrangement", "bullish")
        result = check_trend_break(ema)
        if result:
            signals.append(result)

        # Check 5: OBV divergence
        result = check_obv_divergence(
            indicators.get("close_prices", []),
            indicators.get("obv", []),
            self.thresholds,
        )
        if result:
            signals.append(result)

        # Check 6: Trailing stop
        result = check_trailing_stop(
            indicators.get("high_prices", []),
            indicators.get("close_prices", [-1])[-1] if indicators.get("close_prices") else 0,
            self.thresholds,
        )
        if result:
            signals.append(result)

        # Check 7: Time decay
        result = check_time_decay(
            position.get("expiry", ""),
            self.thresholds.time_decay_dte,
            today=today,
        )
        if result:
            signals.append(result)

        # Check 8: IV collapse
        result = check_iv_collapse(
            indicators.get("hv_20"),
            indicators.get("hv_60"),
            self.thresholds,
        )
        if result:
            signals.append(result)

        # Overall action = highest severity
        overall = "HOLD"
        for sig in signals:
            if ACTION_PRIORITY.get(sig["action"], 0) > ACTION_PRIORITY.get(overall, 0):
                overall = sig["action"]

        return {
            "overall_action": overall,
            "signals": signals,
            "checks_run": 8,
        }
