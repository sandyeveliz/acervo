"""Tests for acervo.metrics — SessionMetrics and TurnMetric."""

from acervo.metrics import SessionMetrics, TurnMetric


class TestTurnMetric:
    def test_defaults(self):
        t = TurnMetric(turn_number=1, timestamp="2026-01-01T00:00:00")
        assert t.warm_tokens == 0
        assert t.context_hit is False

    def test_to_dict(self):
        t = TurnMetric(turn_number=1, timestamp="2026-01-01T00:00:00", warm_tokens=100)
        d = t.to_dict()
        assert d["turn_number"] == 1
        assert d["warm_tokens"] == 100
        assert isinstance(d, dict)


class TestSessionMetrics:
    def test_empty_session(self):
        m = SessionMetrics()
        assert m.turn_count == 0
        assert m.avg_total_tokens == 0.0
        assert m.context_hit_rate == 0.0
        assert m.snapshot() is None
        assert m.summary() == "No turns recorded."

    def test_record_turn(self):
        m = SessionMetrics(session_id="test")
        turn = m.record_turn(warm_tokens=100, hot_tokens=50, total_context_tokens=200)
        assert turn.turn_number == 1
        assert turn.warm_tokens == 100
        assert m.turn_count == 1
        assert m.snapshot() is turn

    def test_multiple_turns(self):
        m = SessionMetrics()
        m.record_turn(total_context_tokens=200, node_count=3, context_hit=True)
        m.record_turn(total_context_tokens=220, node_count=5, context_hit=False)
        m.record_turn(total_context_tokens=210, node_count=6, context_hit=True)

        assert m.turn_count == 3
        assert m.avg_total_tokens == 210.0
        assert abs(m.context_hit_rate - 2/3) < 0.01
        assert m.graph_growth_rate == 1.5  # (6-3) / 2

    def test_fact_density(self):
        m = SessionMetrics()
        m.record_turn(facts_added=2, node_count=4)
        m.record_turn(facts_added=3, node_count=5)
        # total_facts=5, last_nodes=5 -> density=1.0
        assert m.fact_density == 1.0

    def test_export_json(self):
        m = SessionMetrics(session_id="s_test")
        m.record_turn(warm_tokens=100, total_context_tokens=200, context_hit=True)
        export = m.export_json()

        assert export["session_id"] == "s_test"
        assert export["turn_count"] == 1
        assert "aggregates" in export
        assert export["aggregates"]["context_hit_rate"] == 1.0
        assert len(export["turns"]) == 1
        assert export["turns"][0]["warm_tokens"] == 100

    def test_summary(self):
        m = SessionMetrics()
        m.record_turn(
            total_context_tokens=200, node_count=5, edge_count=8,
            facts_added=3, facts_deduped=1, context_hit=True,
        )
        s = m.summary()
        assert "1 turns" in s
        assert "5 nodes" in s
        assert "3 added" in s
