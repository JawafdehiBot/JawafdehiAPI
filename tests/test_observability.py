"""Tests for observability module: metrics, circuit breaker, retry policies."""

import pytest

from cases.observability import (
    pipeline_duration,
    llm_call,
    cache_hit,
    quality_gate,
    section_confidence,
    likhit_failures,
    circuit_breaker_trips,
    null_source_type,
    nepali_script_coverage,
    record_llm_outcome,
    record_cache_lookup,
    record_quality_gate,
    record_confidence,
    record_likhit_failure,
    record_circuit_breaker_trip,
    update_null_source_type_ratio,
    update_nepali_script_coverage,
    track_pipeline_duration,
    export_textfile,
)


class TestHistogram:
    def test_pipeline_duration_observe(self):
        pipeline_duration.observe(5.0, labels={"tier": "source_llm", "command": "test"})
        snap = pipeline_duration.snapshot()
        assert snap["count"] == 1
        assert snap["sum"] == 5.0

    def test_section_confidence_bounds(self):
        record_confidence("bigo", 0.85)
        record_confidence("bigo", -0.5)  # clamped to 0
        record_confidence("bigo", 2.0)  # clamped to 1
        snap = section_confidence.snapshot()
        assert snap["count"] == 3

    def test_track_pipeline_duration_context(self):
        with track_pipeline_duration(tier="rule_based", command="test"):
            pass
        snap = pipeline_duration.snapshot()
        assert snap["count"] >= 1
        assert snap["sum"] >= 0


class TestCounter:
    def test_llm_call_counter(self):
        initial = llm_call.snapshot()["total"]
        record_llm_outcome(True, model="test-model", command="test")
        record_llm_outcome(False, model="test-model", command="test")
        snap = llm_call.snapshot()
        assert snap["total"] == initial + 2

    def test_cache_hit_counter(self):
        record_cache_lookup("mcp", True)
        record_cache_lookup("mcp", False)
        snap = cache_hit.snapshot()
        assert snap["total"] == 2

    def test_quality_gate_counter(self):
        record_quality_gate("bigo_validation", True)
        snap = quality_gate.snapshot()
        assert snap["total"] == 1

    def test_likhit_failure_counter(self):
        record_likhit_failure("pdf")
        snap = likhit_failures.snapshot()
        assert snap["total"] >= 1

    def test_circuit_breaker_trip_counter(self):
        record_circuit_breaker_trip("source_llm")
        snap = circuit_breaker_trips.snapshot()
        assert snap["total"] >= 1


class TestGauge:
    def test_null_source_type_ratio(self):
        update_null_source_type_ratio(0.3)
        assert null_source_type.snapshot() == 0.3
        update_null_source_type_ratio(1.5)  # clamped
        assert null_source_type.snapshot() == 1.0
        update_null_source_type_ratio(-0.5)  # clamped
        assert null_source_type.snapshot() == 0.0

    def test_nepali_script_coverage(self):
        update_nepali_script_coverage(0.75)
        assert nepali_script_coverage.snapshot() == 0.75


class TestExport:
    def test_export_textfile(self, tmp_path):
        path = tmp_path / "metrics.prom"
        export_textfile(str(path))
        content = path.read_text()
        assert "jawafdehi_enrichment_pipeline_duration_seconds" in content
        assert "jawafdehi_llm_call_total" in content
        assert "jawafdehi_null_source_type_ratio" in content
