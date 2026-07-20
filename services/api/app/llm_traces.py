"""In-memory trace store for LLM interactions.

# LESSON POINT: LLMOps Observability
# Unlike classical ML where you track accuracy/F1 per training run, LLM-based
# features need per-request tracing: what prompt was sent, which tools were called,
# how long inference took, how many tokens were consumed, and what it cost.
# These traces sit alongside (but separate from) the sklearn model metrics in MLflow.
"""

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMTrace:
    """A single trace record for one chat interaction."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = ""
    model: str = ""
    prompt: str = ""
    response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class TraceStore:
    """Simple in-memory trace store. Keeps the last N traces."""

    def __init__(self, max_size: int = 100):
        self._traces: list[LLMTrace] = []
        self._max_size = max_size

    def add(self, trace: LLMTrace) -> None:
        self._traces.append(trace)
        if len(self._traces) > self._max_size:
            self._traces = self._traces[-self._max_size :]

    def list_all(self) -> list[dict]:
        return [t.to_dict() for t in reversed(self._traces)]

    def get(self, trace_id: str) -> dict | None:
        for t in self._traces:
            if t.trace_id == trace_id:
                return t.to_dict()
        return None

    def clear(self) -> None:
        self._traces.clear()


# Singleton instance
trace_store = TraceStore()
