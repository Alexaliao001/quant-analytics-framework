"""Tests for Portfolio Risk Analyzer.

Tests cover:
- Greeks estimation (delta, theta, vega, premium remaining)
- Sector and single-name concentration
- HHI-based effective bets
- Scenario stress testing
- Position health scoring
- Full analyze() pipeline
- Edge cases (single position, expired options, zero DTE)
"""

import math
from datetime import date, timedelta

import pytest

from risk.portfolio_risk import (
    Position,
    PortfolioRiskAnalyzer,
    RiskConfig,
    estimate_daily_theta,
    estimate_delta,
    estimate_premium_remaining_pct,
    estimate_vega_impact,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _future_date(days: int) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y-%m-%d")


def _past_date(days: int) -> str:
    return (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")


def _make_position(**overrides) -> Position:
    defaults = {
        "ticker": "AAPL",
        "sector": "tech",
        "option_type": "call",
        "strike": 180.0,
        "stock_price": 185.0,
        "entry_price": 5.50,
        "expiry": _future_date(60),
        "entry_date": _past_date(20),
        "qty": 2,
        "total_cost": 1100.0,
        "stop_loss": 2.75,
        "target1": 8.25,
        "target2": 11.00,
    }
    defaults.update(overrides)
    return Position(**defaults)


# ─── Delta Estimation Tests ────────────────────────────────────────────────


class TestEstimateDelta:
    """Test logistic sigmoid delta approximation."""

    def test_deep_itm_call_high_delta(self):
        delta = estimate_delta(200.0, 150.0, 30, "call")
        assert delta > 0.8

    def test_deep_otm_call_low_delta(self):
        delta = estimate_delta(150.0, 200.0, 30, "call")
        assert delta < 0.2

    def test_atm_call_near_050(self):
        delta = estimate_delta(180.0, 180.0, 30, "call")
        assert 0.35 < delta < 0.65

    def test_put_delta_negative(self):
        delta = estimate_delta(180.0, 180.0, 30, "put")
        assert delta < 0

    def test_deep_itm_put(self):
        delta = estimate_delta(150.0, 200.0, 30, "put")
        assert delta < -0.8

    def test_zero_dte_call_itm(self):
        delta = estimate_delta(190.0, 180.0, 0, "call")
        assert delta == 1.0

    def test_zero_dte_call_otm(self):
        delta = estimate_delta(170.0, 180.0, 0, "call")
        assert delta == 0.0

    def test_zero_dte_put_itm(self):
        delta = estimate_delta(170.0, 180.0, 0, "put")
        assert delta == -1.0

    def test_longer_dte_wider_spread(self):
        """Longer DTE should give delta closer to 0.5 for ATM."""
        delta_short = estimate_delta(180.0, 175.0, 7, "call")
        delta_long = estimate_delta(180.0, 175.0, 90, "call")
        # Short DTE should be more extreme (further from 0.5)
        assert abs(delta_short - 0.5) > abs(delta_long - 0.5)

    def test_call_delta_bounded(self):
        delta = estimate_delta(500.0, 100.0, 30, "call")
        assert 0.01 <= delta <= 0.99


# ─── Theta Estimation Tests ────────────────────────────────────────────────


class TestEstimateTheta:
    """Test accelerated theta decay model."""

    def test_near_expiry_accelerated(self):
        theta_near = estimate_daily_theta(5.0, 10)
        theta_far = estimate_daily_theta(5.0, 60)
        assert theta_near > theta_far

    def test_zero_dte_full_premium(self):
        assert estimate_daily_theta(5.0, 0) == 5.0

    def test_acceleration_tiers(self):
        """DTE < 30 should have 1.5x acceleration."""
        theta_20 = estimate_daily_theta(5.0, 20)
        # Manual: (5.0 / 20) * 1.5 = 0.375
        assert abs(theta_20 - 0.375) < 0.001

    def test_normal_tier(self):
        """DTE 30-60 should have 1.0x."""
        theta_45 = estimate_daily_theta(5.0, 45)
        # Manual: (5.0 / 45) * 1.0 ≈ 0.111
        assert abs(theta_45 - 5.0 / 45) < 0.001

    def test_slow_tier(self):
        """DTE > 60 should have 0.7x."""
        theta_90 = estimate_daily_theta(5.0, 90)
        # Manual: (5.0 / 90) * 0.7 ≈ 0.0389
        expected = (5.0 / 90) * 0.7
        assert abs(theta_90 - expected) < 0.001


# ─── Vega Impact Tests ─────────────────────────────────────────────────────


class TestEstimateVega:
    """Test sqrt-scaled vega impact estimation."""

    def test_zero_dte_no_vega(self):
        assert estimate_vega_impact(5.0, 0) == 0.0

    def test_longer_dte_more_vega(self):
        v_30 = estimate_vega_impact(5.0, 30)
        v_90 = estimate_vega_impact(5.0, 90)
        assert v_90 > v_30

    def test_higher_iv_change_more_impact(self):
        v_10 = estimate_vega_impact(5.0, 30, iv_change_points=10)
        v_20 = estimate_vega_impact(5.0, 30, iv_change_points=20)
        assert v_20 == pytest.approx(v_10 * 2, rel=0.01)


# ─── Premium Remaining Tests ──────────────────────────────────────────────


class TestPremiumRemaining:
    """Test sqrt time decay premium estimation."""

    def test_full_time_100_pct(self):
        assert estimate_premium_remaining_pct(60, 60) == pytest.approx(100.0)

    def test_half_time_about_71_pct(self):
        pct = estimate_premium_remaining_pct(30, 60)
        assert 70 < pct < 72

    def test_zero_dte_zero_pct(self):
        assert estimate_premium_remaining_pct(0, 60) == 0.0

    def test_zero_total_dte(self):
        assert estimate_premium_remaining_pct(10, 0) == 0.0


# ─── Concentration Analysis Tests ─────────────────────────────────────────


class TestConcentration:
    """Test sector and single-name concentration analysis."""

    @pytest.fixture
    def analyzer(self):
        return PortfolioRiskAnalyzer(account_size=50000.0)

    def test_single_sector_high_concentration(self, analyzer):
        positions = [
            _make_position(ticker="AAPL", sector="tech", total_cost=5000),
            _make_position(ticker="MSFT", sector="tech", total_cost=5000),
        ]
        result = analyzer.analyze_concentration(positions)
        assert result["sector_weights"]["tech"]["pct"] == 20.0

    def test_sector_flag_triggered(self, analyzer):
        positions = [
            _make_position(ticker="AAPL", sector="tech", total_cost=10000),
        ]
        result = analyzer.analyze_concentration(positions)
        assert result["sector_weights"]["tech"]["flag"] is True
        assert len(result["sector_flags"]) == 1

    def test_name_flag_triggered(self, analyzer):
        positions = [
            _make_position(ticker="AAPL", total_cost=7000),
        ]
        result = analyzer.analyze_concentration(positions)
        # 7000/50000 = 14% > 12% threshold
        assert result["single_name_weights"]["AAPL"]["flag"] is True

    def test_diversified_no_flags(self, analyzer):
        positions = [
            _make_position(ticker="AAPL", sector="tech", total_cost=2000),
            _make_position(ticker="XOM", sector="energy", total_cost=2000),
            _make_position(ticker="JNJ", sector="healthcare", total_cost=2000),
        ]
        result = analyzer.analyze_concentration(positions)
        assert len(result["sector_flags"]) == 0
        assert len(result["name_flags"]) == 0

    def test_effective_bets(self, analyzer):
        """Two positions in same cluster should yield fewer effective bets."""
        same_cluster = [
            _make_position(ticker="AAPL", sector="tech", total_cost=5000),
            _make_position(ticker="NVDA", sector="semiconductor", total_cost=5000),
        ]
        diff_cluster = [
            _make_position(ticker="AAPL", sector="tech", total_cost=5000),
            _make_position(ticker="XOM", sector="energy", total_cost=5000),
        ]
        bets_same = analyzer.analyze_concentration(same_cluster)["effective_bets"]
        bets_diff = analyzer.analyze_concentration(diff_cluster)["effective_bets"]
        assert bets_diff > bets_same

    def test_directionality_counts(self, analyzer):
        positions = [
            _make_position(option_type="call"),
            _make_position(option_type="call"),
            _make_position(option_type="put"),
        ]
        result = analyzer.analyze_concentration(positions)
        assert result["directionality"]["calls"] == 2
        assert result["directionality"]["puts"] == 1


# ─── Theta Analysis Tests ─────────────────────────────────────────────────


class TestThetaAnalysis:
    """Test portfolio-level theta analysis."""

    @pytest.fixture
    def analyzer(self):
        return PortfolioRiskAnalyzer(account_size=50000.0)

    def test_danger_zone_detected(self, analyzer):
        positions = [
            _make_position(expiry=_future_date(7)),  # DTE=7 < 15
        ]
        result = analyzer.analyze_theta(positions)
        assert len(result["danger_positions"]) == 1

    def test_safe_dte_no_danger(self, analyzer):
        positions = [
            _make_position(expiry=_future_date(60)),
        ]
        result = analyzer.analyze_theta(positions)
        assert len(result["danger_positions"]) == 0

    def test_weekly_theta_approx_5x_daily(self, analyzer):
        positions = [_make_position()]
        result = analyzer.analyze_theta(positions)
        # Rounding happens independently, so allow small difference
        ratio = result["total_weekly_theta_usd"] / result["total_daily_theta_usd"]
        assert 4.8 < ratio < 5.2


# ─── Scenario Tests ────────────────────────────────────────────────────────


class TestScenarios:
    """Test scenario stress testing."""

    @pytest.fixture
    def analyzer(self):
        return PortfolioRiskAnalyzer(account_size=50000.0)

    def test_spy_down_negative_for_calls(self, analyzer):
        positions = [_make_position(option_type="call")]
        result = analyzer.analyze_scenarios(positions)
        assert result["spy_down_10pct"]["impact_usd"] < 0

    def test_spy_up_positive_for_calls(self, analyzer):
        positions = [_make_position(option_type="call")]
        result = analyzer.analyze_scenarios(positions)
        assert result["spy_up_10pct"]["impact_usd"] > 0

    def test_theta_bleed_negative(self, analyzer):
        positions = [_make_position()]
        result = analyzer.analyze_scenarios(positions)
        assert result["theta_5day_bleed"]["impact_usd"] < 0

    def test_vix_spike_positive_for_long_options(self, analyzer):
        positions = [_make_position()]
        result = analyzer.analyze_scenarios(positions)
        assert result["vix_up_20pts"]["impact_usd"] > 0

    def test_all_four_scenarios_present(self, analyzer):
        positions = [_make_position()]
        result = analyzer.analyze_scenarios(positions)
        expected = {"spy_up_10pct", "spy_down_10pct", "vix_up_20pts", "theta_5day_bleed"}
        assert set(result.keys()) == expected


# ─── Health Scoring Tests ──────────────────────────────────────────────────


class TestHealthScoring:
    """Test position health GREEN/YELLOW/RED scoring."""

    @pytest.fixture
    def analyzer(self):
        return PortfolioRiskAnalyzer(account_size=50000.0)

    def test_healthy_position_green(self, analyzer):
        pos = _make_position(
            expiry=_future_date(60),
            strike=180.0,
            stock_price=190.0,
        )
        result = analyzer.analyze_health([pos])
        assert result[0]["status"] == "GREEN"
        assert result[0]["score"] >= 70

    def test_near_expiry_yellow_or_red(self, analyzer):
        """DTE=3 should be YELLOW or RED (heavily penalized)."""
        pos = _make_position(expiry=_future_date(3))
        result = analyzer.analyze_health([pos])
        assert result[0]["status"] in ("YELLOW", "RED")
        assert result[0]["score"] <= 60

    def test_far_otm_deduction(self, analyzer):
        pos = _make_position(
            stock_price=150.0,
            strike=200.0,
            expiry=_future_date(60),
        )
        result = analyzer.analyze_health([pos])
        # Deep OTM should have delta deduction
        assert any("OTM" in r or "Delta" in r for r in result[0]["reasons"])

    def test_below_stop_deduction(self, analyzer):
        pos = _make_position(
            current_option_price=2.0,
            stop_loss=3.0,
        )
        result = analyzer.analyze_health([pos])
        assert any("stop" in r.lower() for r in result[0]["reasons"])

    def test_score_bounded_0_100(self, analyzer):
        # Worst case: everything bad
        pos = _make_position(
            expiry=_future_date(1),
            stock_price=100.0,
            strike=200.0,
            current_option_price=1.0,
            stop_loss=2.0,
        )
        result = analyzer.analyze_health([pos])
        assert 0 <= result[0]["score"] <= 100


# ─── Full Analysis Pipeline Tests ─────────────────────────────────────────


class TestFullAnalysis:
    """Test the complete analyze() pipeline."""

    @pytest.fixture
    def analyzer(self):
        return PortfolioRiskAnalyzer(account_size=50000.0)

    def test_report_has_all_sections(self, analyzer):
        positions = [_make_position()]
        report = analyzer.analyze(positions)
        assert "concentration" in report
        assert "theta" in report
        assert "scenarios" in report
        assert "health" in report
        assert "account_size" in report
        assert "num_positions" in report

    def test_empty_portfolio(self, analyzer):
        report = analyzer.analyze([])
        assert report["num_positions"] == 0

    def test_multi_position_portfolio(self, analyzer):
        positions = [
            _make_position(ticker="AAPL", sector="tech", total_cost=2000),
            _make_position(ticker="NVDA", sector="semiconductor", total_cost=3000),
            _make_position(ticker="XOM", sector="energy", total_cost=1500, option_type="put"),
        ]
        report = analyzer.analyze(positions)
        assert report["num_positions"] == 3
        assert len(report["health"]) == 3
        assert report["concentration"]["directionality"]["calls"] == 2
        assert report["concentration"]["directionality"]["puts"] == 1
