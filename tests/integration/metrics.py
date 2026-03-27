"""Data models for integration test metrics.

Captures per-turn and per-scenario results for both Acervo and baseline runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntityExpectation:
    """Expected entity from a YAML scenario."""

    label: str
    type: str = ""
    layer: str = ""


@dataclass
class RelationExpectation:
    """Expected relation from a YAML scenario."""

    source: str
    target: str
    relation: str


@dataclass
class FactExpectation:
    """Expected fact from a YAML scenario."""

    entity: str
    substring: str


@dataclass
class Checkpoint:
    """Context verification checkpoint from a YAML scenario."""

    should_have_context: bool = True
    context_should_mention: list[str] = field(default_factory=list)
    context_should_not_mention: list[str] = field(default_factory=list)


@dataclass
class ScenarioTurn:
    """One turn in a scripted scenario (loaded from YAML)."""

    user_msg: str
    assistant_msg: str = ""
    phase: str = ""
    description: str = ""
    expected_entities: list[EntityExpectation] = field(default_factory=list)
    expected_relations: list[RelationExpectation] = field(default_factory=list)
    expected_facts: list[FactExpectation] = field(default_factory=list)
    checkpoint: Checkpoint | None = None
    max_new_nodes: int | None = None


@dataclass
class Scenario:
    """A full test scenario loaded from YAML."""

    name: str
    category: str
    description: str
    turns: list[ScenarioTurn]
    system_prompt: str = ""
    persona: str = ""
    persona_role: str = ""
    narrative_hook: str = ""


@dataclass
class TurnResult:
    """Metrics captured for one turn of an Acervo run."""

    turn_number: int
    phase: str
    description: str = ""

    # Token metrics
    acervo_tokens: int = 0
    warm_tokens: int = 0
    hot_tokens: int = 0
    system_tokens: int = 0       # system prompt cost (constant per turn)
    user_tokens: int = 0         # current user message tokens
    overhead_tokens: int = 0     # framing markers + "Understood." acknowledgment
    baseline_tokens: int = 0     # what full history would cost (system + all messages)
    savings_pct: float = 0.0

    # Extraction quality
    entities_expected: int = 0
    entities_found: int = 0
    entities_missing: list[str] = field(default_factory=list)
    relations_expected: int = 0
    relations_found: int = 0
    facts_expected: int = 0
    facts_found: int = 0

    # Context quality
    context_hit: bool = False
    context_mentions_ok: list[str] = field(default_factory=list)
    context_mentions_missing: list[str] = field(default_factory=list)
    context_mentions_unwanted: list[str] = field(default_factory=list)

    # Graph state
    node_count: int = 0
    edge_count: int = 0

    # Timing
    prepare_ms: int = 0
    process_ms: int = 0

    # Detailed capture (for evidence/blog export)
    user_msg: str = ""
    assistant_msg: str = ""
    warm_context: str = ""          # what Acervo injected as context
    entities_extracted: list[str] = field(default_factory=list)  # labels from this turn
    topic: str = ""

    @property
    def entity_recall(self) -> float:
        if self.entities_expected == 0:
            return 1.0
        return self.entities_found / self.entities_expected

    @property
    def checkpoint_passed(self) -> bool:
        return (
            not self.context_mentions_missing
            and not self.context_mentions_unwanted
        )


@dataclass
class ScenarioResult:
    """Aggregated results for one scenario run."""

    name: str
    category: str
    turns: list[TurnResult] = field(default_factory=list)
    total_turns: int = 0

    # Graph final state
    final_node_count: int = 0
    final_edge_count: int = 0
    personal_nodes: int = 0
    universal_nodes: int = 0
    phantom_entities: list[str] = field(default_factory=list)

    # Graph node listing
    graph_nodes: list[dict] = field(default_factory=list)

    # Trace info
    trace_path: str = ""
    trace_lines: int = 0

    # Soft assertion tracking
    soft_failures: list[str] = field(default_factory=list)
    soft_total: int = 0

    @property
    def avg_acervo_tokens(self) -> float:
        if not self.turns:
            return 0
        return sum(t.acervo_tokens for t in self.turns) / len(self.turns)

    @property
    def avg_baseline_tokens(self) -> float:
        if not self.turns:
            return 0
        return sum(t.baseline_tokens for t in self.turns) / len(self.turns)

    @property
    def avg_savings_pct(self) -> float:
        total_acervo = sum(t.acervo_tokens for t in self.turns)
        total_baseline = sum(t.baseline_tokens for t in self.turns)
        if total_baseline == 0:
            return 0
        return (1 - total_acervo / total_baseline) * 100

    @property
    def avg_prepare_ms(self) -> float:
        if not self.turns:
            return 0
        return sum(t.prepare_ms for t in self.turns) / len(self.turns)

    @property
    def avg_process_ms(self) -> float:
        if not self.turns:
            return 0
        return sum(t.process_ms for t in self.turns) / len(self.turns)

    @property
    def entity_recall(self) -> float:
        expected = sum(t.entities_expected for t in self.turns)
        found = sum(t.entities_found for t in self.turns)
        if expected == 0:
            return 1.0
        return found / expected

    @property
    def checkpoint_pass_count(self) -> int:
        return sum(1 for t in self.turns if t.checkpoint_passed)

    @property
    def checkpoint_total(self) -> int:
        return sum(
            1 for t in self.turns
            if t.context_mentions_ok or t.context_mentions_missing or t.context_mentions_unwanted
        )

    @property
    def context_hit_rate(self) -> float:
        if not self.turns:
            return 0
        return sum(1 for t in self.turns if t.context_hit) / len(self.turns)

    @property
    def soft_pass_rate(self) -> float:
        if self.soft_total == 0:
            return 1.0
        return 1 - len(self.soft_failures) / self.soft_total
