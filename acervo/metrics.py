"""Session metrics — tracks per-turn performance and graph growth.

Accumulates metrics during a session, exposes aggregates on demand.
Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class TurnMetric:
    """Metrics captured for a single conversation turn."""

    turn_number: int
    timestamp: str

    # Context tokens
    warm_tokens: int = 0
    hot_tokens: int = 0
    total_context_tokens: int = 0

    # Graph snapshot
    node_count: int = 0
    edge_count: int = 0
    nodes_activated: int = 0

    # Extraction results
    entities_extracted: int = 0
    facts_added: int = 0
    facts_deduped: int = 0

    # Pipeline decisions
    topic: str = ""
    plan_tool: str = ""
    context_hit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionMetrics:
    """Accumulates per-turn metrics and computes session aggregates."""

    session_id: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    turns: list[TurnMetric] = field(default_factory=list)

    def record_turn(self, **kwargs) -> TurnMetric:
        """Record a new turn metric. Returns the created TurnMetric."""
        turn = TurnMetric(
            turn_number=len(self.turns) + 1,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            **kwargs,
        )
        self.turns.append(turn)
        return turn

    def snapshot(self) -> TurnMetric | None:
        """Return the most recent turn metric, or None if no turns."""
        return self.turns[-1] if self.turns else None

    # ── Aggregates ──

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def avg_total_tokens(self) -> float:
        """Average total context tokens across all turns."""
        if not self.turns:
            return 0.0
        return sum(t.total_context_tokens for t in self.turns) / len(self.turns)

    @property
    def avg_warm_tokens(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.warm_tokens for t in self.turns) / len(self.turns)

    @property
    def context_hit_rate(self) -> float:
        """Fraction of turns where graph context was useful (0.0-1.0)."""
        if not self.turns:
            return 0.0
        hits = sum(1 for t in self.turns if t.context_hit)
        return hits / len(self.turns)

    @property
    def graph_growth_rate(self) -> float:
        """Average nodes added per turn."""
        if len(self.turns) < 2:
            return 0.0
        first = self.turns[0].node_count
        last = self.turns[-1].node_count
        return (last - first) / (len(self.turns) - 1)

    @property
    def fact_density(self) -> float:
        """Total facts added / total nodes (at last turn)."""
        if not self.turns:
            return 0.0
        total_facts = sum(t.facts_added for t in self.turns)
        last_nodes = self.turns[-1].node_count
        return total_facts / last_nodes if last_nodes > 0 else 0.0

    @property
    def total_entities_extracted(self) -> int:
        return sum(t.entities_extracted for t in self.turns)

    @property
    def total_facts_added(self) -> int:
        return sum(t.facts_added for t in self.turns)

    @property
    def total_facts_deduped(self) -> int:
        return sum(t.facts_deduped for t in self.turns)

    # ── Export ──

    def export_json(self) -> dict:
        """Export full session metrics as a JSON-serializable dict."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "turn_count": self.turn_count,
            "aggregates": {
                "avg_total_tokens": round(self.avg_total_tokens, 1),
                "avg_warm_tokens": round(self.avg_warm_tokens, 1),
                "context_hit_rate": round(self.context_hit_rate, 3),
                "graph_growth_rate": round(self.graph_growth_rate, 2),
                "fact_density": round(self.fact_density, 2),
                "total_entities_extracted": self.total_entities_extracted,
                "total_facts_added": self.total_facts_added,
                "total_facts_deduped": self.total_facts_deduped,
            },
            "turns": [t.to_dict() for t in self.turns],
        }

    def summary(self) -> str:
        """Human-readable session summary."""
        if not self.turns:
            return "No turns recorded."
        last = self.turns[-1]
        return (
            f"Session: {self.turn_count} turns | "
            f"Avg tokens: {self.avg_total_tokens:.0f} | "
            f"Context hit rate: {self.context_hit_rate:.0%} | "
            f"Graph: {last.node_count} nodes, {last.edge_count} edges | "
            f"Facts: {self.total_facts_added} added, {self.total_facts_deduped} deduped"
        )
