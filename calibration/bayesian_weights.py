"""
Bayesian Signal Calibration — Adaptive weight optimization with shrinkage estimation.

The core problem: you have a backtested trading system with optimized weights,
and now live data is coming in. How do you update the weights without:
  (a) Overreacting to small samples (overfitting to noise)
  (b) Ignoring genuine regime changes (underfitting to drift)

Solution: Bayesian shrinkage estimator that blends backtest priors with live
outcomes, controlled by an adaptive λ parameter.

Theory:
  posterior_weight = λ × live_weight + (1-λ) × prior_weight
  λ = n_live / (n_live + prior_strength)
  prior_strength = max(min_strength, min(max_strength, √n_backtest × 3))

  When live data is sparse:  λ → 0, posterior ≈ prior (trust backtest)
  When live data is abundant: λ → 1, posterior ≈ live (trust reality)

References:
  - Browne (1996): Bayesian Kelly Criterion
  - Black & Litterman (1992): Prior + views blending in portfolio optimization
  - Feng, Polson & Xu (2021): Hierarchical Bayesian factor investing

Example:
    # Define your backtest priors
    priors = {
        ("STRONG_BUY", "bull"):  {"weight": 0.16, "n_backtest": 1077, "ev": 2.70, "win_pct": 0.616},
        ("BUY", "bull"):         {"weight": 0.12, "n_backtest": 2448, "ev": 2.98, "win_pct": 0.585},
        ("HOLD", "neutral"):     {"weight": 0.05, "n_backtest": 226,  "ev": 0.77, "win_pct": 0.530},
    }

    calibrator = BayesianCalibrator(priors)

    # Feed live returns as they settle
    live_returns = {
        ("STRONG_BUY", "bull"): [0.05, -0.02, 0.08, 0.03, -0.01, 0.06, 0.04],
        ("BUY", "bull"):        [0.03, 0.01, -0.04, 0.02, 0.05],
    }

    results = calibrator.update(live_returns)

    for cell_key, cell in results["cells"].items():
        print(f"{cell_key}: prior={cell['prior_weight']:.3f} → "
              f"posterior={cell['posterior_weight']:.3f} "
              f"(λ={cell['lambda']:.2f}, drift={cell['drift']:.3f}, "
              f"{cell['alert_level']})")
"""

import math
from datetime import datetime
from typing import Any, Optional


# ─── Statistical Utilities ──────────────────────────────────────────────────


def compute_wilson_ci(
    successes: int, total: int, z: float = 1.645
) -> tuple[float, float]:
    """Wilson score confidence interval for proportions.

    More accurate than the normal approximation for small samples and
    extreme proportions. Essential for trading where sample sizes are
    often < 100 and win rates can be far from 50%.

    Args:
        successes: Number of successful outcomes (e.g., winning trades)
        total: Total number of observations
        z: Z-score for desired confidence level.
           1.645 = 90% CI, 1.96 = 95% CI, 2.576 = 99% CI

    Returns:
        (lower_bound, upper_bound) of the confidence interval

    Example:
        # 7 wins out of 10 trades — what's the true win rate?
        lo, hi = compute_wilson_ci(7, 10)
        # → (0.393, 0.903) — wide interval, small sample
        lo, hi = compute_wilson_ci(70, 100)
        # → (0.621, 0.769) — much tighter, larger sample
    """
    if total == 0:
        return (0.0, 1.0)
    n = total
    p_hat = successes / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def compute_kelly_weight(
    returns: list[float], half_kelly: bool = True, min_observations: int = 5
) -> Optional[dict[str, Any]]:
    """Compute Kelly criterion-based position weight from observed returns.

    The Kelly criterion maximizes long-term growth rate, but is notoriously
    aggressive. Half-Kelly (default) sacrifices ~25% of growth for ~50% less
    variance — the standard in practice.

    Args:
        returns: List of trade returns (e.g., [0.05, -0.02, 0.08, ...])
        half_kelly: Use half-Kelly (recommended). Full Kelly is too aggressive.
        min_observations: Minimum trades needed for a meaningful estimate.

    Returns:
        Dict with weight, win_pct, ev, avg_win, avg_loss, kelly, n, win_ci_90.
        None if insufficient observations.

    Example:
        returns = [0.05, -0.02, 0.08, 0.03, -0.01, 0.06, -0.03, 0.04]
        result = compute_kelly_weight(returns)
        print(f"Suggested weight: {result['weight']:.2%}")
        print(f"Win rate: {result['win_pct']:.0%} "
              f"(90% CI: {result['win_ci_90'][0]:.0%}-{result['win_ci_90'][1]:.0%})")
    """
    if len(returns) < min_observations:
        return None

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    n = len(returns)
    p = len(wins) / n

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(abs(l) for l in losses) / len(losses) if losses else 0.001

    # Kelly formula: f* = (bp - q) / b
    # where b = avg_win/avg_loss, p = win rate, q = 1 - p
    b = avg_win / avg_loss if avg_loss > 0 else 0.0
    kelly = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0
    weight = kelly * (0.5 if half_kelly else 1.0)
    ev = p * avg_win - (1 - p) * avg_loss

    ci_low, ci_high = compute_wilson_ci(len(wins), n)

    return {
        "weight": round(weight, 4),
        "win_pct": round(p, 4),
        "ev": round(ev, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "kelly": round(kelly, 4),
        "n": n,
        "n_wins": len(wins),
        "win_ci_90": [round(ci_low, 4), round(ci_high, 4)],
    }


# ─── Bayesian Calibrator ───────────────────────────────────────────────────


class BayesianCalibrator:
    """Bayesian shrinkage calibrator for signal weight optimization.

    Maintains a set of prior beliefs (from backtesting) and updates them
    incrementally as live outcomes arrive. Each "cell" is a (signal, regime)
    combination with its own weight.

    The key insight: prior_strength controls how resistant the system is
    to updating. A signal with 2,000 backtest observations has much stronger
    prior resistance than one with 50 observations — because we're more
    confident in the backtest result.

    Drift detection uses 4 alert levels:
    - OK (<2% drift): Prior confirmed by live data
    - WATCH (2-5%): Minor deviation, continue monitoring
    - REVIEW (5-8%): Significant deviation, consider manual review
    - ALERT (>8%): Major deviation, investigate immediately

    Args:
        priors: Dict mapping (signal, regime) tuples to prior statistics.
                Each value should have: weight, n_backtest, ev, win_pct.
        min_prior_strength: Minimum equivalent observations for prior (default: 30)
        max_prior_strength: Maximum equivalent observations for prior (default: 100)
        drift_thresholds: Dict with alert level thresholds (default: standard 4-tier)
    """

    # Default 4-tier alert thresholds
    DEFAULT_THRESHOLDS = {
        "OK": 0.02,      # < 2% absolute drift
        "WATCH": 0.05,   # 2-5%
        "REVIEW": 0.08,  # 5-8%
        # Anything above REVIEW threshold = ALERT
    }

    def __init__(
        self,
        priors: dict[tuple[str, str], dict[str, float]],
        min_prior_strength: float = 30.0,
        max_prior_strength: float = 100.0,
        drift_thresholds: Optional[dict[str, float]] = None,
    ):
        self.priors = priors
        self.min_prior_strength = min_prior_strength
        self.max_prior_strength = max_prior_strength
        self.drift_thresholds = drift_thresholds or self.DEFAULT_THRESHOLDS

        # Extract signal and regime labels from priors
        self.signals = sorted(set(k[0] for k in priors))
        self.regimes = sorted(set(k[1] for k in priors))

    def compute_prior_strength(self, n_backtest: int) -> float:
        """Compute prior strength (equivalent sample size) from backtest count.

        Larger backtest sample → stronger prior → slower adaptation.
        Formula: clamp(√n_backtest × 3, min_strength, max_strength)

        This means:
        - 100 backtest signals → prior_strength = 30 (minimum, adapts quickly)
        - 400 backtest signals → prior_strength = 60 (moderate resistance)
        - 1000 backtest signals → prior_strength = 95 (strong resistance)
        """
        return max(
            self.min_prior_strength,
            min(self.max_prior_strength, math.sqrt(n_backtest) * 3),
        )

    def compute_posterior(
        self, signal: str, regime: str, live_returns: list[float]
    ) -> dict[str, Any]:
        """Compute Bayesian posterior for a single (signal, regime) cell.

        Returns comprehensive analysis including prior, live, posterior weights,
        λ, drift magnitude, alert level, and recommendation.
        """
        key = (signal, regime)
        prior_data = self.priors.get(key, {})
        prior_weight = prior_data.get("weight", 0.0)
        n_backtest = prior_data.get("n_backtest", 0)
        prior_ev = prior_data.get("ev", 0.0)
        prior_win_pct = prior_data.get("win_pct", 0.5)

        prior_strength = self.compute_prior_strength(n_backtest)

        result = {
            "signal": signal,
            "regime": regime,
            "prior_weight": prior_weight,
            "prior_ev": prior_ev,
            "prior_win_pct": prior_win_pct,
            "prior_strength": round(prior_strength, 1),
            "n_backtest": n_backtest,
            "n_live": len(live_returns),
            "lambda": 0.0,
            "live_weight": None,
            "live_stats": None,
            "posterior_weight": prior_weight,
            "drift": 0.0,
            "drift_pct": 0.0,
            "alert_level": "WAIT",
            "recommendation": "Using static prior weight (insufficient live data)",
        }

        # Need minimum observations for meaningful Bayesian update
        settled = [r for r in live_returns if r is not None]
        if len(settled) < 5:
            result["recommendation"] = (
                f"Using static prior weight. Need {5 - len(settled)} more "
                f"settled observations for Bayesian update."
            )
            return result

        # Compute live statistics via Kelly
        live_stats = compute_kelly_weight(settled)
        if live_stats is None:
            return result

        result["live_stats"] = live_stats
        live_weight = live_stats["weight"]
        result["live_weight"] = live_weight

        # Core Bayesian shrinkage
        n_live = len(settled)
        lam = n_live / (n_live + prior_strength)
        result["lambda"] = round(lam, 4)

        posterior = lam * live_weight + (1 - lam) * prior_weight
        result["posterior_weight"] = round(posterior, 4)

        # Drift detection
        drift = abs(posterior - prior_weight)
        result["drift"] = round(drift, 4)
        result["drift_pct"] = round(
            drift / max(abs(prior_weight), 0.01) * 100, 1
        )

        # Alert level classification
        thresholds = self.drift_thresholds
        if drift < thresholds.get("OK", 0.02):
            result["alert_level"] = "OK"
            result["recommendation"] = "Prior weight confirmed by live data."
        elif drift < thresholds.get("WATCH", 0.05):
            result["alert_level"] = "WATCH"
            result["recommendation"] = (
                f"Minor drift ({drift:.3f}). Continue monitoring. "
                f"lambda={lam:.2f}."
            )
        elif drift < thresholds.get("REVIEW", 0.08):
            result["alert_level"] = "REVIEW"
            result["recommendation"] = (
                f"Significant drift ({drift:.3f}). Consider updating "
                f"weight from {prior_weight:.3f} to {posterior:.3f}. "
                f"n_live={n_live}, lambda={lam:.2f}."
            )
        else:
            result["alert_level"] = "ALERT"
            result["recommendation"] = (
                f"Major drift ({drift:.3f})! Prior {prior_weight:.3f} vs "
                f"posterior {posterior:.3f}. Live n={n_live}, "
                f"win={live_stats['win_pct']:.0%}, EV={live_stats['ev']:+.2%}. "
                f"Investigate immediately."
            )

        return result

    def update(
        self, live_returns: dict[tuple[str, str], list[float]]
    ) -> dict[str, Any]:
        """Run Bayesian update across all cells.

        Args:
            live_returns: Dict mapping (signal, regime) to list of returns.
                         Missing cells will use prior weight with no update.

        Returns:
            Comprehensive results dict with per-cell analysis, alerts, and
            overall system health score.

        Example:
            results = calibrator.update({
                ("STRONG_BUY", "bull"): [0.05, -0.02, 0.08, 0.03, -0.01],
                ("BUY", "bull"): [0.03, 0.01, -0.04],
            })
            print(f"Health: {results['summary']['health_score']}/100")
            for alert in results['alerts']:
                print(f"  [{alert['level']}] {alert['cell']}: {alert['recommendation']}")
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "cells": {},
            "alerts": [],
            "summary": {},
        }

        alert_counts = {"OK": 0, "WATCH": 0, "REVIEW": 0, "ALERT": 0, "WAIT": 0}
        total_live = sum(len(v) for v in live_returns.values())

        for signal in self.signals:
            for regime in self.regimes:
                key = (signal, regime)
                returns = live_returns.get(key, [])
                cell_result = self.compute_posterior(signal, regime, returns)

                cell_key = f"{signal}/{regime}"
                results["cells"][cell_key] = cell_result
                alert_counts[cell_result["alert_level"]] += 1

                if cell_result["alert_level"] in ("REVIEW", "ALERT"):
                    results["alerts"].append({
                        "cell": cell_key,
                        "level": cell_result["alert_level"],
                        "drift": cell_result["drift"],
                        "recommendation": cell_result["recommendation"],
                    })

        # System health score (weighted by severity)
        total_cells = len(self.signals) * len(self.regimes)
        if total_cells > 0:
            health_score = (
                alert_counts["OK"] * 100
                + alert_counts["WATCH"] * 75
                + alert_counts["REVIEW"] * 40
                + alert_counts["ALERT"] * 10
                + alert_counts["WAIT"] * 80  # no data = trust prior
            ) / total_cells
        else:
            health_score = 0.0

        # Data readiness tiers
        if total_live < 30:
            readiness = "INSUFFICIENT"
        elif total_live < 100:
            readiness = "PRELIMINARY"
        elif total_live < 300:
            readiness = "DEVELOPING"
        else:
            readiness = "RELIABLE"

        results["summary"] = {
            "health_score": round(health_score, 1),
            "alert_counts": alert_counts,
            "total_live_observations": total_live,
            "data_readiness": readiness,
        }

        return results
