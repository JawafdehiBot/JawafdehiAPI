"""Prometheus metrics for the case overview enrichment pipeline.

Exports 9 key metric families:
  - pipeline_duration_seconds (Histogram: p50/p95/p99)
  - llm_call_total (Counter: success/failure)
  - cache_hit_total (Counter: per cache layer)
  - quality_gate_total (Counter: pass/fail)
  - section_confidence (Histogram: per section)
  - likhit_conversion_failures_total (Counter)
  - circuit_breaker_trips_total (Counter)
  - null_source_type_ratio (Gauge)
  - nepali_script_coverage (Gauge)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# In-memory metrics backend (no external dependency required; a Prometheus
# client_textfile exporter can be added later with `prometheus-client`).
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()


@dataclass
class _Histogram:
    name: str
    help: str
    labelnames: list[str] = field(default_factory=list)
    _buckets: list[float] = field(default_factory=lambda: [0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300])
    _sum: float = 0.0
    _count: int = 0
    _bucket_counts: dict[str, int] = field(default_factory=dict)

    def observe(self, value: float, labels: dict | None = None) -> None:
        with _LOCK:
            self._sum += value
            self._count += 1
            for boundary in sorted(set(self._buckets)):
                if value <= boundary:
                    key = f"{_label_str(labels)}le={boundary}"
                    self._bucket_counts[key] = self._bucket_counts.get(key, 0) + 1
            key = f"{_label_str(labels)}le=+Inf"
            self._bucket_counts[key] = self._bucket_counts.get(key, 0) + 1

    def snapshot(self) -> dict:
        with _LOCK:
            return {"sum": self._sum, "count": self._count, "buckets": dict(self._bucket_counts)}


@dataclass
class _Counter:
    name: str
    help: str
    labelnames: list[str] = field(default_factory=list)
    _value: float = 0.0
    _labels: dict[str, float] = field(default_factory=dict)

    def inc(self, amount: float = 1, labels: dict | None = None) -> None:
        with _LOCK:
            self._value += amount
            key = _label_str(labels)
            self._labels[key] = self._labels.get(key, 0) + amount

    def snapshot(self) -> dict:
        with _LOCK:
            return {"total": self._value, "by_label": dict(self._labels)}


@dataclass
class _Gauge:
    name: str
    help: str
    labelnames: list[str] = field(default_factory=list)
    _value: float = 0.0

    def set(self, value: float) -> None:
        with _LOCK:
            self._value = value

    def inc(self, amount: float = 1) -> None:
        with _LOCK:
            self._value += amount

    def dec(self, amount: float = 1) -> None:
        with _LOCK:
            self._value -= amount

    def snapshot(self) -> float:
        with _LOCK:
            return self._value


def _label_str(labels: dict | None) -> str:
    if not labels:
        return ""
    parts = sorted(f'{k}="{v}"' for k, v in labels.items())
    return "{" + ",".join(parts) + "}"


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

pipeline_duration = _Histogram(
    name="jawafdehi_enrichment_pipeline_duration_seconds",
    help="End-to-end pipeline duration per case in seconds.",
    labelnames=["tier", "command"],
    _buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600],
)

llm_call = _Counter(
    name="jawafdehi_llm_call_total",
    help="LLM call count by outcome.",
    labelnames=["outcome", "model", "command"],
)

cache_hit = _Counter(
    name="jawafdehi_cache_hit_total",
    help="Cache hit count by layer.",
    labelnames=["layer", "hit"],
)

quality_gate = _Counter(
    name="jawafdehi_quality_gate_total",
    help="Quality gate check count by result.",
    labelnames=["gate", "result"],
)

section_confidence = _Histogram(
    name="jawafdehi_section_confidence",
    help="Per-section LLM extraction confidence (0-1).",
    labelnames=["section"],
    _buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

likhit_failures = _Counter(
    name="jawafdehi_likhit_conversion_failures_total",
    help="Likhit document conversion failures.",
    labelnames=["file_type"],
)

circuit_breaker_trips = _Counter(
    name="jawafdehi_circuit_breaker_trips_total",
    help="Circuit breaker trip count.",
    labelnames=["circuit"],
)

null_source_type = _Gauge(
    name="jawafdehi_null_source_type_ratio",
    help="Ratio of source documents with null source_type (0-1).",
)

nepali_script_coverage = _Gauge(
    name="jawafdehi_nepali_script_coverage",
    help="Proportion of cases with Devanagari text in evidence (0-1).",
)


# ---------------------------------------------------------------------------
# Textfile export (Prometheus node_exporter compatible)
# ---------------------------------------------------------------------------

_GAUGES: dict[str, _Gauge] = {
    "jawafdehi_null_source_type_ratio": null_source_type,
    "jawafdehi_nepali_script_coverage": nepali_script_coverage,
}


def export_textfile(path: str) -> None:
    """Write all current metric snapshots to *path* in Prometheus exposition format."""
    lines: list[str] = []

    for metric in [
        pipeline_duration,
        section_confidence,
    ]:
        snap = metric.snapshot()
        lines.append(f"# HELP {metric.name} {metric.help}")
        lines.append(f"# TYPE {metric.name} histogram")
        if snap["count"] > 0:
            lines.append(f"{metric.name}_sum {snap['sum']}")
            lines.append(f"{metric.name}_count {snap['count']}")
            for bucket_key, bucket_count in sorted(snap["buckets"].items()):
                lines.append(f"{metric.name}_bucket{{{bucket_key}}} {bucket_count}")

    for metric in [llm_call, cache_hit, quality_gate, likhit_failures, circuit_breaker_trips]:
        snap = metric.snapshot()
        lines.append(f"# HELP {metric.name} {metric.help}")
        lines.append(f"# TYPE {metric.name} counter")
        lines.append(f"{metric.name}_total {snap['total']}")
        for label_key, label_val in sorted(snap.get("by_label", {}).items()):
            clean_key = label_key.strip("{}") if label_key else ""
            lines.append(f"{metric.name}_total{{{clean_key}}} {label_val}")

    for name, gauge in _GAUGES.items():
        lines.append(f"# HELP {name} {gauge.help}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {gauge.snapshot()}")

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Convenience helpers for pipeline instrumentation
# ---------------------------------------------------------------------------

import time as _time
from contextlib import contextmanager


@contextmanager
def track_pipeline_duration(tier: str, command: str = "unknown"):
    """Context manager to record pipeline duration for a case."""
    start = _time.monotonic()
    try:
        yield
    finally:
        elapsed = _time.monotonic() - start
        pipeline_duration.observe(elapsed, labels={"tier": tier, "command": command})


def record_llm_outcome(success: bool, model: str = "unknown", command: str = "unknown") -> None:
    outcome = "success" if success else "failure"
    llm_call.inc(labels={"outcome": outcome, "model": model, "command": command})


def record_cache_lookup(layer: str, hit: bool) -> None:
    cache_hit.inc(labels={"layer": layer, "hit": "true" if hit else "false"})


def record_quality_gate(gate: str, passed: bool) -> None:
    quality_gate.inc(labels={"gate": gate, "result": "pass" if passed else "fail"})


def record_confidence(section: str, confidence: float) -> None:
    section_confidence.observe(max(0.0, min(1.0, confidence)), labels={"section": section})


def record_likhit_failure(file_type: str = "unknown") -> None:
    likhit_failures.inc(labels={"file_type": file_type})


def record_circuit_breaker_trip(circuit: str) -> None:
    circuit_breaker_trips.inc(labels={"circuit": circuit})


def update_null_source_type_ratio(ratio: float) -> None:
    null_source_type.set(max(0.0, min(1.0, ratio)))


def update_nepali_script_coverage(coverage: float) -> None:
    nepali_script_coverage.set(max(0.0, min(1.0, coverage)))
