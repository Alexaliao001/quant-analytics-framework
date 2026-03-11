"""Tests for Bayesian Signal Calibration.

Tests cover:
- Wilson confidence interval correctness
- Kelly criterion weight calculation
- Bayesian shrinkage behavior (λ convergence)
- Prior strength scaling
- Drift detection and alert levels
- Edge cases (empty data, single observation, extreme returns)
- Full update pipeline across multiple cells
"""

import math

import pytest

from calibration.bayesian_weights import (
    BayesianCalibrator,
    compute_kelly_weight,
    compute_wilson_ci,
)


# ─── Wilson CI Tests ────────────────────────────────────────────────────────


class TestWilsonCI:
    """Test Wilson score confidence interval implementation."""

    def test_perfect_record(self):
        """10/10 wins should give high but not 100% CI."""
        lo, hi = compute_wilson_ci(10, 10)
        assert lo > 0.7
        assert hi <= 1.0

    def test_zero_record(self):
        """0/10 wins should give low CI."""
        lo, hi = compute_wilson_ci(0, 10)
        assert lo >= 0.0
        assert hi < 0.3

    def test_empty_sample(self):
        """0/0 should return full range."""
        lo, hi = compute_wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_50_50(self):
        """50/100 should center around 0.5."""
        lo, hi = compute_wilson_ci(50, 100)
        assert 0.4 < lo < 0.5
        assert 0.5 < hi < 0.6

    def test_large_sample_narrows_ci(self):
        """Larger sample should give narrower CI."""
        lo_small, hi_small = compute_wilson_ci(7, 10)
        lo_large, hi_large = compute_wilson_ci(700, 1000)
        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        assert width_large < width_small

    def test_higher_z_wider_ci(self):
        """95% CI should be wider than 90% CI."""
        lo_90, hi_90 = compute_wilson_ci(60, 100, z=1.645)
        lo_95, hi_95 = compute_wilson_ci(60, 100, z=1.96)
        assert (hi_95 - lo_95) > (hi_90 - lo_90)


# ─── Kelly Criterion Tests ──────────────────────────────────────────────────


class TestKellyCriterion:
    """Test Kelly criterion weight calculation."""

    def test_insufficient_data(self):
        """Should return None with < min_observations."""
        assert compute_kelly_weight([0.05, -0.02]) is None
        assert compute_kelly_weight([]) is None

    def test_all_winners(self):
        """All positive returns should give positive weight."""
        result = compute_kelly_weight([0.05, 0.03, 0.08, 0.02, 0.06])
        assert result is not None
        assert result["weight"] > 0
        assert result["win_pct"] == 1.0
        assert result["ev"] > 0

    def test_all_losers(self):
        """All negative returns should give zero weight."""
        result = compute_kelly_weight([-0.05, -0.03, -0.08, -0.02, -0.06])
        assert result is not None
        assert result["weight"] == 0.0
        assert result["win_pct"] == 0.0
        assert result["ev"] < 0

    def test_half_kelly_smaller(self):
        """Half Kelly should give smaller weight than full Kelly."""
        returns = [0.05, -0.02, 0.08, 0.03, -0.01, 0.06, -0.03]
        half = compute_kelly_weight(returns, half_kelly=True)
        full = compute_kelly_weight(returns, half_kelly=False)
        assert half["weight"] <= full["weight"]

    def test_ci_included(self):
        """Result should include Wilson CI for win rate."""
        result = compute_kelly_weight([0.05, -0.02, 0.08, 0.03, -0.01])
        assert "win_ci_90" in result
        lo, hi = result["win_ci_90"]
        assert lo <= result["win_pct"] <= hi

    def test_return_fields_complete(self):
        """Result should have all expected fields."""
        result = compute_kelly_weight([0.05, -0.02, 0.08, 0.03, -0.01])
        expected_fields = {
            "weight", "win_pct", "ev", "avg_win", "avg_loss",
            "kelly", "n", "n_wins", "win_ci_90",
        }
        assert set(result.keys()) == expected_fields


# ─── Bayesian Calibrator Tests ──────────────────────────────────────────────


class TestBayesianCalibrator:
    """Test the full Bayesian calibration system."""

    @pytest.fixture
    def sample_priors(self):
        """Example prior configuration for testing."""
        return {
            ("STRONG", "bull"): {
                "weight": 0.16, "n_backtest": 1000, "ev": 2.70, "win_pct": 0.616,
            },
            ("STRONG", "bear"): {
                "weight": 0.00, "n_backtest": 50, "ev": -1.71, "win_pct": 0.430,
            },
            ("MODERATE", "bull"): {
                "weight": 0.12, "n_backtest": 2000, "ev": 2.98, "win_pct": 0.585,
            },
            ("MODERATE", "bear"): {
                "weight": 0.00, "n_backtest": 30, "ev": -3.94, "win_pct": 0.380,
            },
        }

    @pytest.fixture
    def calibrator(self, sample_priors):
        return BayesianCalibrator(sample_priors)

    def test_no_live_data_returns_prior(self, calibrator):
        """With no live data, posterior should equal prior."""
        result = calibrator.compute_posterior("STRONG", "bull", [])
        assert result["posterior_weight"] == 0.16
        assert result["lambda"] == 0.0
        assert result["alert_level"] == "WAIT"

    def test_small_sample_stays_near_prior(self, calibrator):
        """With few live observations, posterior should stay close to prior."""
        live = [0.05, -0.02, 0.08, 0.03, -0.01, 0.06]  # 6 observations
        result = calibrator.compute_posterior("STRONG", "bull", live)

        # λ should be small (6 live vs ~95 prior strength for n=1000)
        assert result["lambda"] < 0.1
        # Posterior should be close to prior
        assert abs(result["posterior_weight"] - 0.16) < 0.05

    def test_large_sample_moves_toward_live(self, calibrator):
        """With many live observations, posterior should approach live weight."""
        # Generate 200 returns with ~60% win rate
        import random
        random.seed(42)
        live = [random.gauss(0.02, 0.05) for _ in range(200)]

        result = calibrator.compute_posterior("STRONG", "bull", live)

        # λ should be substantial
        assert result["lambda"] > 0.5
        # Posterior should be closer to live than to prior
        if result["live_weight"] is not None:
            dist_to_prior = abs(result["posterior_weight"] - result["prior_weight"])
            dist_to_live = abs(result["posterior_weight"] - result["live_weight"])
            assert dist_to_live <= dist_to_prior

    def test_prior_strength_scales_with_backtest(self, calibrator):
        """More backtest data should mean stronger prior resistance."""
        strength_1000 = calibrator.compute_prior_strength(1000)
        strength_50 = calibrator.compute_prior_strength(50)
        assert strength_1000 > strength_50

    def test_prior_strength_clamped(self, calibrator):
        """Prior strength should be within [min, max] bounds."""
        assert calibrator.compute_prior_strength(1) >= 30.0
        assert calibrator.compute_prior_strength(1000000) <= 100.0

    def test_drift_alert_ok(self, calibrator):
        """Tiny drift should be OK."""
        result = calibrator.compute_posterior("STRONG", "bull", [])
        result["drift"] = 0.01
        # Direct test of threshold logic
        assert 0.01 < calibrator.drift_thresholds["OK"]

    def test_full_update_all_cells(self, calibrator):
        """Full update should produce results for all (signal, regime) cells."""
        live = {
            ("STRONG", "bull"): [0.05, -0.02, 0.08, 0.03, -0.01],
            ("MODERATE", "bull"): [0.03, 0.01, -0.04, 0.02, 0.05],
        }
        results = calibrator.update(live)

        # Should have all 4 cells (2 signals × 2 regimes)
        assert len(results["cells"]) == 4
        assert "STRONG/bull" in results["cells"]
        assert "STRONG/bear" in results["cells"]
        assert "MODERATE/bull" in results["cells"]
        assert "MODERATE/bear" in results["cells"]

    def test_update_summary_fields(self, calibrator):
        """Update should include health score and data readiness."""
        results = calibrator.update({})
        assert "summary" in results
        assert "health_score" in results["summary"]
        assert "data_readiness" in results["summary"]
        assert "alert_counts" in results["summary"]

    def test_data_readiness_tiers(self, calibrator):
        """Data readiness should scale with total observations."""
        # No data
        results = calibrator.update({})
        assert results["summary"]["data_readiness"] == "INSUFFICIENT"

    def test_alerts_only_for_review_and_above(self, calibrator):
        """Alerts list should only contain REVIEW and ALERT level items."""
        live = {("STRONG", "bull"): [0.05, -0.02, 0.08, 0.03, -0.01]}
        results = calibrator.update(live)
        for alert in results["alerts"]:
            assert alert["level"] in ("REVIEW", "ALERT")

    def test_custom_drift_thresholds(self, sample_priors):
        """Custom thresholds should override defaults."""
        cal = BayesianCalibrator(
            sample_priors,
            drift_thresholds={"OK": 0.01, "WATCH": 0.02, "REVIEW": 0.03},
        )
        assert cal.drift_thresholds["OK"] == 0.01

    def test_lambda_convergence_formula(self, calibrator):
        """λ should follow the exact formula: n_live / (n_live + prior_strength)."""
        # STRONG/bull has n_backtest=1000 → prior_strength = min(100, √1000×3) ≈ 94.87
        prior_strength = calibrator.compute_prior_strength(1000)
        n_live = 50
        expected_lambda = n_live / (n_live + prior_strength)

        live = [0.01] * n_live  # All same return, doesn't matter for λ test
        result = calibrator.compute_posterior("STRONG", "bull", live)

        assert abs(result["lambda"] - expected_lambda) < 0.01

    def test_unknown_cell_uses_defaults(self, calibrator):
        """Unknown (signal, regime) pair should use safe defaults."""
        result = calibrator.compute_posterior("UNKNOWN", "unknown", [0.05, -0.02, 0.08, 0.03, -0.01])
        assert result["prior_weight"] == 0.0
        assert result["n_backtest"] == 0
