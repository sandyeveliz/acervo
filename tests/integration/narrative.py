"""Story beat detection and narrative builder for integration reports.

Analyzes ScenarioResult data to detect compelling moments ("aha moments")
that demonstrate Acervo's value. Pure Python analysis — no LLM calls needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.integration.metrics import ScenarioResult, TurnResult


@dataclass
class StoryBeat:
    """A single narrative moment detected from turn data."""

    turn_number: int
    beat_type: str  # resurrection, cost_crossover, compression_climax, graph_milestone, small_talk_resilience
    headline: str
    detail: str
    metric_value: float = 0.0


@dataclass
class ScenarioNarrative:
    """Narrative wrapper for a scenario result."""

    persona_name: str
    persona_role: str
    hook: str
    key_beats: list[StoryBeat] = field(default_factory=list)
    headline_stat: str = ""
    total_turns: int = 0
    category: str = ""


def detect_beats(result: ScenarioResult) -> list[StoryBeat]:
    """Detect all story beats from a scenario result."""
    beats: list[StoryBeat] = []
    beats.extend(_detect_resurrections(result.turns))
    crossover = _detect_cost_crossover(result.turns)
    if crossover:
        beats.append(crossover)
    climax = _detect_compression_climax(result.turns)
    if climax:
        beats.append(climax)
    beats.extend(_detect_graph_milestones(result.turns))
    beats.extend(_detect_small_talk_resilience(result.turns))
    beats.sort(key=lambda b: b.turn_number)
    return beats


def build_narrative(
    result: ScenarioResult,
    persona: str = "",
    persona_role: str = "",
    hook: str = "",
) -> ScenarioNarrative:
    """Build a full narrative from a scenario result + metadata."""
    beats = detect_beats(result)

    # Headline stat: pick the most impressive number
    headline = f"{result.avg_savings_pct:.0f}% token savings"
    if result.total_turns >= 50:
        headline += f" over {result.total_turns} turns"

    return ScenarioNarrative(
        persona_name=persona or _infer_persona(result),
        persona_role=persona_role,
        hook=hook or result.category,
        key_beats=beats,
        headline_stat=headline,
        total_turns=result.total_turns,
        category=result.category,
    )


# ── Beat Detectors ──


def _detect_resurrections(turns: list[TurnResult]) -> list[StoryBeat]:
    """Find turns where context was restored after a topic was dormant."""
    topic_last_seen: dict[str, int] = {}
    beats: list[StoryBeat] = []
    min_gap = 5  # at least 5 turns of dormancy to count

    for t in turns:
        topic = t.topic.strip().lower() if t.topic else ""
        if not topic:
            continue

        if topic in topic_last_seen and t.context_hit:
            gap = t.turn_number - topic_last_seen[topic]
            if gap >= min_gap:
                beats.append(StoryBeat(
                    turn_number=t.turn_number,
                    beat_type="resurrection",
                    headline=f"Recalled '{topic}' after {gap} turns of silence",
                    detail=(
                        f"Turn {t.turn_number}: the user returned to '{topic}' "
                        f"(last discussed at turn {topic_last_seen[topic]}). "
                        f"Acervo injected the relevant context from its knowledge graph."
                    ),
                    metric_value=float(gap),
                ))

        topic_last_seen[topic] = t.turn_number

    return beats


def _detect_cost_crossover(turns: list[TurnResult]) -> StoryBeat | None:
    """Find the turn where cumulative Acervo cost drops below 50% of baseline."""
    cum_acervo = 0
    cum_baseline = 0

    for t in turns:
        cum_acervo += t.acervo_tokens
        cum_baseline += t.baseline_tokens
        if cum_baseline > 0 and cum_acervo / cum_baseline <= 0.50:
            savings = (1 - cum_acervo / cum_baseline) * 100
            return StoryBeat(
                turn_number=t.turn_number,
                beat_type="cost_crossover",
                headline=f"50% savings milestone at turn {t.turn_number}",
                detail=(
                    f"By turn {t.turn_number}, Acervo had used {cum_acervo:,} tokens "
                    f"vs {cum_baseline:,} for full history — {savings:.0f}% savings."
                ),
                metric_value=savings,
            )
    return None


def _detect_compression_climax(turns: list[TurnResult]) -> StoryBeat | None:
    """Find the turn with the highest token savings percentage."""
    if not turns:
        return None

    best = max(turns, key=lambda t: t.savings_pct)
    if best.savings_pct <= 0:
        return None

    return StoryBeat(
        turn_number=best.turn_number,
        beat_type="compression_climax",
        headline=f"Peak compression: {best.savings_pct:.0f}% savings at turn {best.turn_number}",
        detail=(
            f"Turn {best.turn_number}: Acervo used {best.acervo_tokens} tokens "
            f"vs {best.baseline_tokens} for full history — "
            f"a {best.savings_pct:.0f}% reduction."
        ),
        metric_value=best.savings_pct,
    )


def _detect_graph_milestones(turns: list[TurnResult]) -> list[StoryBeat]:
    """Detect when the knowledge graph crosses node count thresholds."""
    thresholds = [10, 25, 50, 100]
    crossed: set[int] = set()
    beats: list[StoryBeat] = []

    for t in turns:
        for th in thresholds:
            if t.node_count >= th and th not in crossed:
                crossed.add(th)
                beats.append(StoryBeat(
                    turn_number=t.turn_number,
                    beat_type="graph_milestone",
                    headline=f"Knowledge graph reached {th} entities",
                    detail=(
                        f"Turn {t.turn_number}: Acervo's knowledge graph "
                        f"now contains {t.node_count} entities and {t.edge_count} relations."
                    ),
                    metric_value=float(th),
                ))
    return beats


def _detect_small_talk_resilience(turns: list[TurnResult]) -> list[StoryBeat]:
    """Detect sequences of small talk where graph stayed clean."""
    small_talk_streak = 0
    streak_start = 0
    beats: list[StoryBeat] = []

    for t in turns:
        if t.phase in ("small_talk", "tangent", "personal"):
            if small_talk_streak == 0:
                streak_start = t.turn_number
            small_talk_streak += 1
        else:
            if small_talk_streak >= 3:
                beats.append(StoryBeat(
                    turn_number=streak_start,
                    beat_type="small_talk_resilience",
                    headline=f"Survived {small_talk_streak} turns of small talk without graph pollution",
                    detail=(
                        f"Turns {streak_start}-{streak_start + small_talk_streak - 1}: "
                        f"the conversation went off-topic, but Acervo kept "
                        f"the knowledge graph clean."
                    ),
                    metric_value=float(small_talk_streak),
                ))
            small_talk_streak = 0

    # Handle trailing streak
    if small_talk_streak >= 3:
        beats.append(StoryBeat(
            turn_number=streak_start,
            beat_type="small_talk_resilience",
            headline=f"Survived {small_talk_streak} turns of small talk without graph pollution",
            detail=(
                f"Turns {streak_start}-{streak_start + small_talk_streak - 1}: "
                f"the conversation went off-topic, but Acervo kept "
                f"the knowledge graph clean."
            ),
            metric_value=float(small_talk_streak),
        ))

    return beats


def _infer_persona(result: ScenarioResult) -> str:
    """Try to extract a persona name from the first few turns."""
    for t in result.turns[:5]:
        for entity in t.entities_extracted:
            # Heuristic: first extracted entity in intro phase is likely the persona
            if t.phase == "intro" and entity[0].isupper():
                return entity
    return result.name.replace("_", " ").title()
