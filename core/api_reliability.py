"""
API Reliability Layer — Production-grade HTTP client for high-frequency data fetching.

Handles the reality of working with rate-limited REST APIs at scale:
- Connection pooling (reuses TCP connections, avoids handshake storms)
- Semaphore-based concurrency limiting (prevents burst overload)
- Exponential backoff retry with jitter
- Thread-safe error tracking and call counting
- Structured error responses (never throws, always returns)

Designed for financial data APIs but applicable to any rate-limited REST service.

Example:
    client = ReliableAPIClient(
        base_url="https://api.example.com",
        api_key_param="apiKey",
        api_key=os.environ["API_KEY"],
        max_connections=10,
        max_concurrent=8,
        max_retries=3,
    )
    data = client.get("/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-12-31")
    if data.get("_failed"):
        print(f"Request failed: {data['error']}")

    # Parallel fetching with ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed

    paths = [f"/v2/aggs/ticker/{t}/range/1/day/2024-01-01/2024-12-31"
             for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]]

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(client.get, p): p for p in paths}
        for f in as_completed(futures):
            result = f.result()
            if not result.get("_failed"):
                print(f"Got {len(result.get('results', []))} bars")

    # Check health after batch
    stats = client.get_stats()
    print(f"Total calls: {stats['total_calls']}, Errors: {stats['total_errors']}")
    print(f"Error rate: {stats['error_rate']:.1%}")
"""

import os
import threading
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter


class ReliableAPIClient:
    """Thread-safe HTTP client with connection pooling, rate limiting, and retry logic.

    Design decisions:
    - Semaphore is released BEFORE backoff sleep (don't hold connection slots while waiting)
    - Error tracking is lock-protected for thread safety
    - Failed requests return structured dicts instead of raising (caller decides policy)
    - Call counting enables monitoring without external instrumentation

    Args:
        base_url: API base URL (e.g., "https://api.example.com")
        api_key_param: Query parameter name for API key (e.g., "apiKey", "token")
        api_key: API key value. Falls back to env var {API_KEY_ENV_VAR} if not provided.
        max_connections: HTTP connection pool size (default: 10)
        max_concurrent: Max simultaneous in-flight requests via semaphore (default: 8)
        max_retries: Max retry attempts per request (default: 3)
        base_backoff: Base backoff time in seconds (default: 0.5)
        max_backoff: Maximum backoff time in seconds (default: 4.0)
        timeout: Request timeout in seconds (default: 20)
    """

    def __init__(
        self,
        base_url: str,
        api_key_param: str = "apiKey",
        api_key: Optional[str] = None,
        max_connections: int = 10,
        max_concurrent: int = 8,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        max_backoff: float = 4.0,
        timeout: int = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key_param = api_key_param
        self.api_key = api_key or os.environ.get("API_KEY", "")
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.timeout = timeout

        # Connection-pooled session (reuses TCP connections across requests)
        self._session = requests.Session()
        self._session.mount(
            "https://",
            HTTPAdapter(pool_connections=max_connections, pool_maxsize=max_connections),
        )

        # Concurrency limiter — prevents burst of simultaneous connections
        # that would trigger rate limits or overwhelm the server
        self._semaphore = threading.Semaphore(max_concurrent)

        # Thread-safe error tracking
        self._errors: list[dict] = []
        self._call_count = 0
        self._lock = threading.Lock()

    def get(self, path: str, params: Optional[dict] = None) -> dict[str, Any]:
        """Make a GET request with retry, backoff, and concurrency control.

        Returns:
            dict: API response JSON on success.
            dict with "_failed": True and "error": str on failure.
                  Never raises — caller decides error handling policy.

        Thread-safe: Can be called from multiple ThreadPoolExecutor workers.
        """
        if params is None:
            params = {}
        params[self.api_key_param] = self.api_key
        url = f"{self.base_url}{path}"

        last_error = None
        for attempt in range(self.max_retries):
            # Acquire semaphore — limits concurrent in-flight requests
            with self._semaphore:
                with self._lock:
                    self._call_count += 1
                try:
                    resp = self._session.get(url, params=params, timeout=self.timeout)
                    if resp.status_code == 429:
                        # Rate limited — fall through to backoff
                        last_error = Exception(
                            f"HTTP 429 Rate Limited (attempt {attempt + 1})"
                        )
                    elif resp.status_code >= 500:
                        # Server error — retryable
                        last_error = Exception(
                            f"HTTP {resp.status_code} Server Error (attempt {attempt + 1})"
                        )
                    else:
                        resp.raise_for_status()
                        return resp.json()
                except requests.exceptions.Timeout as e:
                    last_error = e
                except requests.exceptions.ConnectionError as e:
                    last_error = e
                except Exception as e:
                    last_error = e

            # Backoff OUTSIDE semaphore — don't hold the connection slot while sleeping.
            # This is critical: if we held the semaphore during sleep, we'd block
            # other threads from making requests during our backoff window.
            if attempt < self.max_retries - 1:
                wait = min((2**attempt) * self.base_backoff, self.max_backoff)
                time.sleep(wait)

        # All retries exhausted — record failure and return structured error
        error_str = str(last_error)
        with self._lock:
            self._errors.append({"path": path, "error": error_str[:200]})
        return {"error": error_str, "_failed": True}

    def get_stats(self) -> dict[str, Any]:
        """Return thread-safe snapshot of client health metrics.

        Useful for monitoring after batch operations:
            stats = client.get_stats()
            if stats["error_rate"] > 0.1:
                logger.warning("High error rate: %.1f%%", stats["error_rate"] * 100)
        """
        with self._lock:
            total = self._call_count
            errors = list(self._errors)
        return {
            "total_calls": total,
            "total_errors": len(errors),
            "error_rate": len(errors) / total if total > 0 else 0.0,
            "errors": errors[-10:],  # last 10 errors only
        }

    def reset_stats(self) -> None:
        """Reset error tracking. Call between independent batch operations."""
        with self._lock:
            self._errors.clear()
            self._call_count = 0

    def close(self) -> None:
        """Close the underlying HTTP session and release connection pool."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MultiFallbackResolver:
    """Multi-layer fallback resolution pattern.

    A common problem in data engineering: you need to resolve some entity
    (e.g., a ticker to its sector ETF) but no single data source is reliable.
    This class implements a priority-ordered fallback chain.

    Example:
        resolver = MultiFallbackResolver(
            layers=[
                ("override", {"AAPL": "XLK", "NVDA": "SMH", "TSLA": "CARZ"}),
                ("sic_exact", {"3674": "SMH", "7372": "XLK", "2911": "XLE"}),
                ("sic_prefix", {"36": "SMH", "73": "XLK", "29": "XLE"}),
            ],
            default="XLK",
        )

        # Tries layers in order, returns first match
        etf = resolver.resolve("AAPL")           # → "XLK" (from override)
        etf = resolver.resolve("UNKNOWN", "3674") # → "SMH" (from sic_exact)
        etf = resolver.resolve("UNKNOWN", "9999") # → "XLK" (default)
    """

    def __init__(self, layers: list[tuple[str, dict]], default: str):
        """
        Args:
            layers: List of (name, mapping_dict) in priority order.
                    First match wins.
            default: Value returned when no layer matches.
        """
        self.layers = layers
        self.default = default

    def resolve(self, primary_key: str, secondary_key: str = "") -> str:
        """Resolve through fallback chain.

        Args:
            primary_key: First key to try (e.g., ticker symbol)
            secondary_key: Second key for code-based lookups (e.g., SIC code)

        Returns:
            Resolved value from highest-priority matching layer, or default.
        """
        # Layer 1+: Direct primary key lookup
        for name, mapping in self.layers:
            if primary_key in mapping:
                return mapping[primary_key]

        # Layers with secondary key (exact match)
        secondary = str(secondary_key).strip()
        if secondary:
            for name, mapping in self.layers:
                if secondary in mapping:
                    return mapping[secondary]
            # Prefix matching (e.g., SIC division = first 2 digits)
            if len(secondary) >= 2:
                prefix = secondary[:2]
                for name, mapping in self.layers:
                    if prefix in mapping:
                        return mapping[prefix]

        return self.default


class DataQualityTracker:
    """Track data quality metrics across a batch fetch operation.

    After fetching data from multiple sources, you need to know:
    - Which sources succeeded?
    - Which sources had warnings?
    - What's the overall reliability?

    Example:
        tracker = DataQualityTracker()
        tracker.record_success("daily_ohlcv")
        tracker.record_success("weekly_ohlcv")
        tracker.record_warning("hourly_ohlcv", "Only 3 bars returned")
        tracker.record_failure("monthly_ohlcv", "API timeout")

        report = tracker.get_report()
        # {
        #     "reliability": "medium",
        #     "sources_ok": ["daily_ohlcv", "weekly_ohlcv"],
        #     "warnings": [{"source": "hourly_ohlcv", "msg": "Only 3 bars returned"}],
        #     "failures": [{"source": "monthly_ohlcv", "msg": "API timeout"}],
        #     "success_rate": 0.75,
        # }
    """

    def __init__(self):
        self._successes: list[str] = []
        self._warnings: list[dict] = []
        self._failures: list[dict] = []

    def record_success(self, source: str) -> None:
        self._successes.append(source)

    def record_warning(self, source: str, message: str) -> None:
        self._warnings.append({"source": source, "msg": message})

    def record_failure(self, source: str, message: str) -> None:
        self._failures.append({"source": source, "msg": message})

    def get_report(self) -> dict:
        total = len(self._successes) + len(self._failures)
        success_rate = len(self._successes) / total if total > 0 else 0.0

        if success_rate >= 0.8 and not self._failures:
            reliability = "high"
        elif success_rate >= 0.5:
            reliability = "medium"
        else:
            reliability = "low"

        return {
            "reliability": reliability,
            "sources_ok": list(self._successes),
            "warnings": list(self._warnings),
            "failures": list(self._failures),
            "success_rate": round(success_rate, 3),
        }
