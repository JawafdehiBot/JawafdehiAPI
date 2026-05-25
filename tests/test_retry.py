"""Tests for retry module."""

import pytest

from cases.retry import retry_with_backoff, retryable_http_status, retryable_network_error


class TestRetryWithBackoff:
    def test_returns_on_success(self):
        result = retry_with_backoff(lambda x: x * 2, 5)
        assert result == 10

    def test_retries_on_failure(self):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient")
            return "ok"

        result = retry_with_backoff(flaky, max_retries=3, base_seconds=0.001, max_seconds=0.01, jitter=0)
        assert result == "ok"
        assert call_count[0] == 3

    def test_raises_after_exhausting_retries(self):
        def always_fails():
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            retry_with_backoff(always_fails, max_retries=2, base_seconds=0.001, max_seconds=0.01, jitter=0)

    def test_respects_retryable_filter(self):
        def raises_value_error():
            raise ValueError("bad argument")

        with pytest.raises(ValueError):
            retry_with_backoff(
                raises_value_error,
                max_retries=2,
                retryable_exceptions=(RuntimeError,),
                base_seconds=0.001,
                jitter=0,
            )


class TestRetryableHelpers:
    def test_retryable_http_status(self):
        assert retryable_http_status(429) is True
        assert retryable_http_status(503) is True
        assert retryable_http_status(200) is False
        assert retryable_http_status(500) is False

    def test_retryable_network_error(self):
        assert retryable_network_error(TimeoutError("timeout")) is True
        assert retryable_network_error(ConnectionError("refused")) is True
        assert retryable_network_error(ValueError("not a network error")) is False
