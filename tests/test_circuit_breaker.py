"""Tests for circuit breaker module."""

import time

import pytest

from cases.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError


class TestCircuitBreaker:
    def test_closed_on_init(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == "closed"

    def test_passes_successful_calls(self):
        cb = CircuitBreaker(name="test")
        result = cb.call(lambda x: x * 2, 21)
        assert result == 42
        assert cb.state == "closed"

    def test_trips_after_consecutive_failures(self):
        cb = CircuitBreaker(name="test")
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            except RuntimeError:
                pass
        assert cb.state == "open"

    def test_raises_open_error_when_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=1)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        except RuntimeError:
            pass
        assert cb.state == "open"
        with pytest.raises(CircuitBreakerOpenError, match="open"):
            cb.call(lambda: 42)

    def test_resets_after_cooldown(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0.01)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        except RuntimeError:
            pass
        assert cb.state == "open"
        time.sleep(0.02)
        result = cb.call(lambda: 99)
        assert result == 99
        assert cb.state == "closed"

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0.01)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        except RuntimeError:
            pass
        time.sleep(0.02)
        cb.call(lambda: 42)
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0.01)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        except RuntimeError:
            pass
        time.sleep(0.02)
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        except RuntimeError:
            pass
        assert cb.state == "open"
