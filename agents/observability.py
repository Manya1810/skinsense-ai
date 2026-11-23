"""SkinSense AI - Observability, Auditability & Data Provenance Tracing
Complies with Requirement 9:
OpenTelemetry-inspired hierarchical span tracing, cryptographic provenance hashes,
and exportable audit trail logs.
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid


class TraceSpan:
    """Represents an execution span within an agentic trace."""
    def __init__(self, name: str, span_id: str, parent_span_id: Optional[str] = None):
        self.name = name
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_time = time.perf_counter()
        self.start_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0
        self.status: str = "RUNNING"
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "event": event_name,
            "data": payload or {},
        })

    def finish(self, status: str = "OK") -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_iso,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class ObservabilityTracer:
    """Central observability tracer capturing provenance and audit trail for every run."""

    def __init__(self, case_id: str = "CASE-DEFAULT"):
        self.trace_id = f"TR-{uuid.uuid4().hex[:12].upper()}"
        self.case_id = case_id
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._spans: Dict[str, TraceSpan] = {}
        self._provenance: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "case_id": self.case_id,
            "frameworks": ["CrewAI 1.15.33", "Google ADK 2.13.0"],
            "image_sha256": None,
            "symptoms_sha256": None,
        }

    def record_provenance(self, image_bytes: Optional[bytes] = None, symptoms_text: Optional[str] = None) -> None:
        """Records cryptographic SHA-256 hashes of input artifacts for legal provenance."""
        if image_bytes:
            self._provenance["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        if symptoms_text:
            self._provenance["symptoms_sha256"] = hashlib.sha256(symptoms_text.encode("utf-8")).hexdigest()

    def start_span(self, name: str, parent_span_id: Optional[str] = None) -> str:
        """Starts a new trace span, returning span_id."""
        s_id = uuid.uuid4().hex[:8]
        span = TraceSpan(name=name, span_id=s_id, parent_span_id=parent_span_id)
        self._spans[s_id] = span
        return s_id

    def end_span(self, span_id: str, status: str = "OK", attributes: Optional[Dict[str, Any]] = None) -> None:
        """Completes an active trace span."""
        if span_id in self._spans:
            span = self._spans[span_id]
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            span.finish(status=status)

    def export_trace_log(self, filepath: str = "static/reports/audit_traces.jsonl") -> str:
        """Appends the trace session to a persistent JSON Lines audit log."""
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        trace_data = {
            "trace_id": self.trace_id,
            "case_id": self.case_id,
            "timestamp": self.timestamp,
            "provenance": self._provenance,
            "total_spans": len(self._spans),
            "spans": [s.to_dict() for s in self._spans.values()],
        }

        with open(out_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_data) + "\n")

        return str(out_p)

    def get_summary(self) -> Dict[str, Any]:
        """Returns in-memory snapshot of trace."""
        return {
            "trace_id": self.trace_id,
            "spans_count": len(self._spans),
            "provenance": self._provenance,
            "spans": [s.to_dict() for s in self._spans.values()],
        }

