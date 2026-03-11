"""Tests for the API Reliability Layer.

Tests cover:
- Connection pooling and session reuse
- Semaphore-based concurrency limiting
- Exponential backoff retry logic
- Thread-safe error tracking
- Structured error responses (never throws)
- MultiFallbackResolver priority chain
- DataQualityTracker reporting
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from core.api_reliability import (
    DataQualityTracker,
    MultiFallbackResolver,
    ReliableAPIClient,
)


# ─── ReliableAPIClient Tests ────────────────────────────────────────────────


class TestReliableAPIClient:
    """Test the core HTTP client with retry and concurrency control."""

    def test_successful_request(self):
        """Happy path: API returns 200 with valid JSON."""
        client = ReliableAPIClient(base_url="https://api.example.com", api_key="test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [1, 2, 3]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_response):
            result = client.get("/v2/data")

        assert result == {"results": [1, 2, 3]}
        assert not result.get("_failed")

    def test_retry_on_429(self):
        """Rate limited (429) should trigger retry with backoff."""
        client = ReliableAPIClient(
            base_url="https://api.example.com",
            api_key="test",
            max_retries=3,
            base_backoff=0.01,  # Fast backoff for tests
        )

        # First two calls return 429, third succeeds
        mock_429 = MagicMock()
        mock_429.status_code = 429

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"results": "ok"}
        mock_ok.raise_for_status = MagicMock()

        with patch.object(
            client._session, "get", side_effect=[mock_429, mock_429, mock_ok]
        ):
            result = client.get("/v2/data")

        assert result == {"results": "ok"}

    def test_retry_on_500(self):
        """Server error (500) should trigger retry."""
        client = ReliableAPIClient(
            base_url="https://api.example.com",
            api_key="test",
            max_retries=2,
            base_backoff=0.01,
        )

        mock_500 = MagicMock()
        mock_500.status_code = 500

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"ok": True}
        mock_ok.raise_for_status = MagicMock()

        with patch.object(
            client._session, "get", side_effect=[mock_500, mock_ok]
        ):
            result = client.get("/v2/data")

        assert result == {"ok": True}

    def test_all_retries_exhausted(self):
        """When all retries fail, return structured error (never throw)."""
        client = ReliableAPIClient(
            base_url="https://api.example.com",
            api_key="test",
            max_retries=2,
            base_backoff=0.01,
        )

        mock_429 = MagicMock()
        mock_429.status_code = 429

        with patch.object(
            client._session, "get", side_effect=[mock_429, mock_429]
        ):
            result = client.get("/v2/data")

        assert result["_failed"] is True
        assert "429" in result["error"]

    def test_error_tracking_thread_safe(self):
        """Error tracking should be thread-safe across concurrent requests."""
        client = ReliableAPIClient(
            base_url="https://api.example.com",
            api_key="test",
            max_retries=1,
            max_concurrent=4,
        )

        mock_fail = MagicMock()
        mock_fail.status_code = 429

        with patch.object(client._session, "get", return_value=mock_fail):
            threads = []
            for i in range(10):
                t = threading.Thread(target=client.get, args=(f"/path/{i}",))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        stats = client.get_stats()
        assert stats["total_calls"] == 10
        assert stats["total_errors"] == 10
        assert stats["error_rate"] == 1.0

    def test_stats_reset(self):
        """Stats reset should clear all counters."""
        client = ReliableAPIClient(base_url="https://api.example.com", api_key="test")
        client._call_count = 100
        client._errors = [{"path": "/x", "error": "test"}]

        client.reset_stats()
        stats = client.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_errors"] == 0

    def test_semaphore_limits_concurrency(self):
        """Semaphore should limit concurrent in-flight requests."""
        max_concurrent = 2
        client = ReliableAPIClient(
            base_url="https://api.example.com",
            api_key="test",
            max_concurrent=max_concurrent,
            max_retries=1,
        )

        peak_concurrent = 0
        current_concurrent = 0
        lock = threading.Lock()

        original_get = client._session.get

        def slow_get(*args, **kwargs):
            nonlocal peak_concurrent, current_concurrent
            with lock:
                current_concurrent += 1
                peak_concurrent = max(peak_concurrent, current_concurrent)
            time.sleep(0.05)
            with lock:
                current_concurrent -= 1
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = {"ok": True}
            mock.raise_for_status = MagicMock()
            return mock

        with patch.object(client._session, "get", side_effect=slow_get):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(client.get, f"/p/{i}") for i in range(10)]
                for f in futures:
                    f.result()

        # Peak concurrency should not exceed semaphore limit
        assert peak_concurrent <= max_concurrent

    def test_context_manager(self):
        """Client should work as context manager."""
        with ReliableAPIClient(
            base_url="https://api.example.com", api_key="test"
        ) as client:
            assert client is not None

    def test_params_passed_correctly(self):
        """Custom params should be merged with API key."""
        client = ReliableAPIClient(
            base_url="https://api.example.com",
            api_key="mykey123",
            api_key_param="token",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            client.get("/v2/data", {"limit": 100, "sort": "asc"})

        call_args = mock_get.call_args
        params = call_args[1]["params"] if "params" in call_args[1] else call_args[0][1]
        assert params["token"] == "mykey123"
        assert params["limit"] == 100
        assert params["sort"] == "asc"


# ─── MultiFallbackResolver Tests ────────────────────────────────────────────


class TestMultiFallbackResolver:
    """Test the multi-layer fallback resolution pattern."""

    @pytest.fixture
    def resolver(self):
        return MultiFallbackResolver(
            layers=[
                ("override", {"AAPL": "XLK", "NVDA": "SMH", "TSLA": "CARZ"}),
                ("sic_exact", {"3674": "SMH", "7372": "XLK", "2911": "XLE"}),
                ("sic_prefix", {"36": "SMH", "73": "XLK", "29": "XLE"}),
            ],
            default="XLK",
        )

    def test_primary_key_match(self, resolver):
        """Primary key should match first (highest priority)."""
        assert resolver.resolve("AAPL") == "XLK"
        assert resolver.resolve("NVDA") == "SMH"
        assert resolver.resolve("TSLA") == "CARZ"

    def test_secondary_key_exact_match(self, resolver):
        """When primary key misses, fall back to secondary exact match."""
        assert resolver.resolve("UNKNOWN", "3674") == "SMH"
        assert resolver.resolve("UNKNOWN", "2911") == "XLE"

    def test_secondary_key_prefix_match(self, resolver):
        """When exact secondary misses, try prefix match."""
        assert resolver.resolve("UNKNOWN", "3699") == "SMH"  # prefix "36" → SMH
        assert resolver.resolve("UNKNOWN", "2999") == "XLE"  # prefix "29" → XLE

    def test_default_when_nothing_matches(self, resolver):
        """Return default when no layer matches."""
        assert resolver.resolve("UNKNOWN", "9999") == "XLK"
        assert resolver.resolve("UNKNOWN") == "XLK"

    def test_primary_overrides_secondary(self, resolver):
        """Primary key should take precedence even if secondary would also match."""
        # AAPL → XLK via primary, even if secondary SIC would map elsewhere
        assert resolver.resolve("AAPL", "3674") == "XLK"


# ─── DataQualityTracker Tests ───────────────────────────────────────────────


class TestDataQualityTracker:
    """Test data quality monitoring and reporting."""

    def test_high_reliability(self):
        """All successes → high reliability."""
        tracker = DataQualityTracker()
        tracker.record_success("daily")
        tracker.record_success("weekly")
        tracker.record_success("monthly")

        report = tracker.get_report()
        assert report["reliability"] == "high"
        assert report["success_rate"] == 1.0
        assert len(report["sources_ok"]) == 3

    def test_medium_reliability(self):
        """Mix of successes and failures → medium reliability."""
        tracker = DataQualityTracker()
        tracker.record_success("daily")
        tracker.record_success("weekly")
        tracker.record_failure("monthly", "API timeout")

        report = tracker.get_report()
        assert report["reliability"] == "medium"
        assert 0.5 <= report["success_rate"] < 1.0

    def test_low_reliability(self):
        """Majority failures → low reliability."""
        tracker = DataQualityTracker()
        tracker.record_failure("daily", "timeout")
        tracker.record_failure("weekly", "429")
        tracker.record_success("monthly")

        report = tracker.get_report()
        assert report["reliability"] == "low"

    def test_warnings_tracked_separately(self):
        """Warnings don't affect success rate but are reported."""
        tracker = DataQualityTracker()
        tracker.record_success("daily")
        tracker.record_warning("daily", "Only 3 bars returned")

        report = tracker.get_report()
        assert report["reliability"] == "high"
        assert len(report["warnings"]) == 1
        assert report["warnings"][0]["msg"] == "Only 3 bars returned"

    def test_empty_tracker(self):
        """Empty tracker should report gracefully."""
        tracker = DataQualityTracker()
        report = tracker.get_report()
        assert report["reliability"] == "low"
        assert report["success_rate"] == 0.0
